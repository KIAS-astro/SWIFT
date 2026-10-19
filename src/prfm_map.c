/*******************************************************************************
 * This file is part of SWIFT.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

/* Config parameters. */
#include <config.h>

/* Some standard headers. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef WITH_MPI
#include <mpi.h>
#endif

/* This object's header. */
#include "prfm_map.h"

/* Local headers. */
#include "engine.h"
#include "error.h"
#include "hydro.h"
#include "parser.h"
#include "physical_constants.h"
#include "potential.h"
#include "space.h"
#include "timeline.h"
#include "wind.h"

/*! Fields per pixel: gas mass, star mass, star sum m v_z, star sum m v_z^2 */
#define PRFM_MAP_NFIELD 4

/**
 * @brief The map of the current step. One per rank; written by
 * prfm_map_build() before the tasks of a step are launched and only read by
 * the star formation tasks.
 */
static struct {

  /*! Pixel data, PRFM_MAP_NFIELD values per pixel, x index slowest */
  double *data;

  /*! Number of pixels per side */
  int npix;

  /*! Disc centre and half the map side (internal units) */
  double c[3], half;

  /*! Box size and periodicity, for wrapping separations */
  double dim[3];
  int periodic;

  /*! Pixel side, R_cyl (internal units) */
  double pixel, R_cyl;

  /*! Pixels to search on each side of a particle's pixel */
  int nk;

  /*! The external potential, for rho_d (NULL: rho_d = 0) */
  const struct external_potential *potential;

  /*! Has the map been built? */
  int ready;

} prfm_map;

/**
 * @brief Read the map parameters from the PRFMStarFormation section.
 *
 * @param mp (return) The parameters.
 * @param params The parsed parameter file.
 * @param phys_const The physical constants in internal units.
 */
void prfm_map_params_init(struct prfm_map_params *mp,
                          struct swift_params *params,
                          const struct phys_const *phys_const) {

  const double kpc = 1e3 * phys_const->const_parsec;

  double centre[3] = {0., 0., 0.};
  parser_get_opt_param_double_array(params, "PRFMStarFormation:disc_centre", 3,
                                    centre);
  for (int i = 0; i < 3; i++) mp->centre[i] = centre[i];
  mp->centre_is_absolute = parser_get_opt_param_int(
      params, "PRFMStarFormation:disc_centre_is_absolute", 0);

  mp->pixel =
      parser_get_opt_param_double(params, "PRFMStarFormation:pixel_kpc", 0.1) *
      kpc;
  mp->R_cyl = parser_get_opt_param_double(
                  params, "PRFMStarFormation:cylinder_radius_kpc", 0.5) *
              kpc;
  mp->half_height = parser_get_opt_param_double(
                        params, "PRFMStarFormation:half_height_kpc", 1.0) *
                    kpc;
  mp->half_width = parser_get_opt_param_double(
                       params, "PRFMStarFormation:map_half_width_kpc", 25.0) *
                   kpc;
  mp->npix = (int)ceil(2. * mp->half_width / mp->pixel - 1e-6);
  mp->use_dark_matter =
      parser_get_opt_param_int(params, "PRFMStarFormation:use_dark_matter", 1);

  if (mp->npix < 1) error("PRFM map: no pixels");
  if (mp->R_cyl <= 0. || mp->half_height <= 0.)
    error("PRFM map: cylinder radius and half-height must be > 0");

#if !defined(EXTERNAL_POTENTIAL_HERNQUIST) && !defined(EXTERNAL_POTENTIAL_NFW_DK)
  if (mp->use_dark_matter)
    error(
        "PRFMStarFormation:use_dark_matter needs the Hernquist or NFW-DK "
        "external potential (--with-ext-potential=hernquist or nfw-dk); set "
        "it to 0 otherwise.");
#endif
}

/**
 * @brief Print the map parameters.
 */
void prfm_map_params_print(const struct prfm_map_params *mp) {
  message(
      "PRFM face-on map: %d x %d pixels of %e, half-width %e, slab half-height "
      "%e, R_cyl %e (internal units); centre (%e, %e, %e) %s; rho_d %s",
      mp->npix, mp->npix, mp->pixel, mp->half_width, mp->half_height,
      mp->R_cyl, mp->centre[0], mp->centre[1], mp->centre[2],
      mp->centre_is_absolute ? "absolute" : "relative to the box centre",
      mp->use_dark_matter ? "from the external potential" : "= 0");
}

/**
 * @brief Separation from the centre along one axis, wrapped in a periodic box.
 */
static double prfm_map_wrap(double dx, const double dim, const int periodic) {
  if (periodic) {
    if (dx > 0.5 * dim) dx -= dim;
    if (dx < -0.5 * dim) dx += dim;
  }
  return dx;
}

/**
 * @brief Build the map from the gas and star particles of all ranks.
 *
 * @param e The #engine.
 * @param mp The map parameters.
 */
void prfm_map_build(struct engine *e, const struct prfm_map_params *mp) {

  const ticks tic = getticks();
  const struct space *s = e->s;
  const int npix = mp->npix;
  const size_t ndata = (size_t)PRFM_MAP_NFIELD * npix * npix;

  if (prfm_map.data == NULL || prfm_map.npix != npix) {
    free(prfm_map.data);
    prfm_map.data = (double *)malloc(ndata * sizeof(double));
    if (prfm_map.data == NULL) error("Failed to allocate the PRFM map");
    prfm_map.npix = npix;
  }
  double *restrict data = prfm_map.data;
  memset(data, 0, ndata * sizeof(double));

  double c[3];
  for (int i = 0; i < 3; i++)
    c[i] = mp->centre[i] + (mp->centre_is_absolute ? 0. : 0.5 * s->dim[i]);

  /* The map is a square of npix pixels centred on the disc centre */
  const double half = 0.5 * npix * mp->pixel;
  const double inv_pixel = 1. / mp->pixel;
  for (int i = 0; i < 3; i++) {
    prfm_map.c[i] = c[i];
    prfm_map.dim[i] = s->dim[i];
  }
  prfm_map.half = half;
  prfm_map.periodic = s->periodic;
  prfm_map.pixel = mp->pixel;
  prfm_map.R_cyl = mp->R_cyl;
  prfm_map.nk = (int)ceil(mp->R_cyl * inv_pixel);
  prfm_map.potential = mp->use_dark_matter ? e->external_potential : NULL;

  /* Gas */
  for (size_t k = 0; k < s->nr_parts; k++) {
    const struct part *p = &s->parts[k];
    if (p->time_bin == time_bin_inhibited ||
        p->time_bin == time_bin_not_created)
      continue;
    /* Decoupled wind particles are not part of the disc */
    if (part_is_decoupled(p)) continue;
    const double dz = prfm_map_wrap(p->x[2] - c[2], s->dim[2], s->periodic);
    if (fabs(dz) >= mp->half_height) continue;
    const double dx = prfm_map_wrap(p->x[0] - c[0], s->dim[0], s->periodic);
    const double dy = prfm_map_wrap(p->x[1] - c[1], s->dim[1], s->periodic);
    const int i = (int)floor((dx + half) * inv_pixel);
    const int j = (int)floor((dy + half) * inv_pixel);
    if (i < 0 || i >= npix || j < 0 || j >= npix) continue;
    data[PRFM_MAP_NFIELD * ((size_t)i * npix + j)] += hydro_get_mass(p);
  }

  /* Stars */
  for (size_t k = 0; k < s->nr_sparts; k++) {
    const struct spart *sp = &s->sparts[k];
    if (sp->time_bin == time_bin_inhibited ||
        sp->time_bin == time_bin_not_created)
      continue;
    const double dz = prfm_map_wrap(sp->x[2] - c[2], s->dim[2], s->periodic);
    if (fabs(dz) >= mp->half_height) continue;
    const double dx = prfm_map_wrap(sp->x[0] - c[0], s->dim[0], s->periodic);
    const double dy = prfm_map_wrap(sp->x[1] - c[1], s->dim[1], s->periodic);
    const int i = (int)floor((dx + half) * inv_pixel);
    const int j = (int)floor((dy + half) * inv_pixel);
    if (i < 0 || i >= npix || j < 0 || j >= npix) continue;
    double *pix = &data[PRFM_MAP_NFIELD * ((size_t)i * npix + j)];
    const double m = sp->mass;
    const double vz = sp->v[2];
    pix[1] += m;
    pix[2] += m * vz;
    pix[3] += m * vz * vz;
  }

#ifdef WITH_MPI
  if (MPI_Allreduce(MPI_IN_PLACE, data, (int)ndata, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD) != MPI_SUCCESS)
    error("Failed to reduce the PRFM map");
#endif

  prfm_map.ready = 1;

  if (e->verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}

/**
 * @brief 2D cubic spline of compact support 1, unnormalised.
 */
static double prfm_map_kernel(const double q) {
  if (q >= 1.) return 0.;
  if (q < 0.5) return 1. - 6. * q * q + 6. * q * q * q;
  const double y = 1. - q;
  return 2. * y * y * y;
}

/**
 * @brief Surface densities, stellar vertical dispersion and dark matter
 * density at a position (internal units).
 *
 * All four are 0 before the first map is built and for positions whose
 * cylinder contains no map pixel.
 *
 * @param x The position.
 * @param Sigma_g (return) Gas surface density.
 * @param Sigma_s (return) Stellar surface density.
 * @param sigma_sz (return) Stellar vertical velocity dispersion.
 * @param rho_d (return) Dark matter density of the Hernquist halo.
 */
void prfm_map_sample(const double x[3], double *Sigma_g, double *Sigma_s,
                     double *sigma_sz, double *rho_d) {

  *Sigma_g = 0.;
  *Sigma_s = 0.;
  *sigma_sz = 0.;
  *rho_d = 0.;
  if (!prfm_map.ready) return;

  const int npix = prfm_map.npix;
  const double pixel = prfm_map.pixel;
  const double inv_R = 1. / prfm_map.R_cyl;
  const double *restrict data = prfm_map.data;

  /* Position relative to the map corner, wrapped as in the build */
  const double px = prfm_map_wrap(x[0] - prfm_map.c[0], prfm_map.dim[0],
                                  prfm_map.periodic) +
                    prfm_map.half;
  const double py = prfm_map_wrap(x[1] - prfm_map.c[1], prfm_map.dim[1],
                                  prfm_map.periodic) +
                    prfm_map.half;
  const int i0 = (int)floor(px / pixel);
  const int j0 = (int)floor(py / pixel);
  const int nk = prfm_map.nk;

  double wsum = 0., mg = 0., ms = 0., mv = 0., mv2 = 0.;
  for (int i = i0 - nk; i <= i0 + nk; i++) {
    if (i < 0 || i >= npix) continue;
    const double dx = (i + 0.5) * pixel - px;
    for (int j = j0 - nk; j <= j0 + nk; j++) {
      if (j < 0 || j >= npix) continue;
      const double dy = (j + 0.5) * pixel - py;
      const double w = prfm_map_kernel(sqrt(dx * dx + dy * dy) * inv_R);
      if (w <= 0.) continue;
      const double *pix = &data[PRFM_MAP_NFIELD * ((size_t)i * npix + j)];
      wsum += w;
      mg += w * pix[0];
      ms += w * pix[1];
      mv += w * pix[2];
      mv2 += w * pix[3];
    }
  }

  if (wsum > 0.) {
    const double inv_area = 1. / (wsum * pixel * pixel);
    *Sigma_g = mg * inv_area;
    *Sigma_s = ms * inv_area;
    if (ms > 0.) {
      const double vmean = mv / ms;
      const double var = mv2 / ms - vmean * vmean;
      *sigma_sz = var > 0. ? sqrt(var) : 0.;
    }
  }

#ifdef EXTERNAL_POTENTIAL_HERNQUIST
  /* Hernquist (1990) eq. 2, rho = M a / (2 pi r (r + a)^3), with r softened
   * by the potential's epsilon so that it stays finite at the centre. */
  const struct external_potential *pot = prfm_map.potential;
  if (pot != NULL) {
    const double rx = x[0] - pot->x[0];
    const double ry = x[1] - pot->x[1];
    const double rz = x[2] - pot->x[2];
    const double r = sqrt(rx * rx + ry * ry + rz * rz + pot->epsilon2);
    const double ra = r + pot->al;
    *rho_d = pot->mass * pot->al / (2. * M_PI * r * ra * ra * ra);
  }
#elif defined(EXTERNAL_POTENTIAL_NFW_DK)
  /* NFW plus the Diemer & Kravtsov (2014) outer term, r softened by the
   * potential's epsilon as in the Hernquist case. */
  const struct external_potential *pot = prfm_map.potential;
  if (pot != NULL) {
    const double rx = x[0] - pot->x[0];
    const double ry = x[1] - pot->x[1];
    const double rz = x[2] - pot->x[2];
    const double r = sqrt(rx * rx + ry * ry + rz * rz + pot->eps * pot->eps);
    *rho_d = nfw_dk_density(pot, r);
  }
#endif
}

/**
 * @brief Free the map.
 */
void prfm_map_clean(void) {
  free(prfm_map.data);
  prfm_map.data = NULL;
  prfm_map.ready = 0;
}
