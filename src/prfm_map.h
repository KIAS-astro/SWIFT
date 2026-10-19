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
#ifndef SWIFT_PRFM_MAP_H
#define SWIFT_PRFM_MAP_H

/**
 * @file src/prfm_map.h
 * @brief Face-on map of an isolated disc for PRFM star formation.
 *
 * A 2D grid of square pixels in the plane z = z_c of the disc, centred on
 * (x_c, y_c), holding for particles with |z - z_c| < half_height the gas mass,
 * the stellar mass and the stellar sum m v_z and sum m v_z^2. It is rebuilt
 * once per step from the particles on each rank and summed over ranks. A gas
 * particle's Sigma_g, Sigma_* and sigma_*z are the averages of the map over
 * the pixels whose centres lie within R_cyl of it, weighted by the 2D cubic
 * spline W(d / R_cyl) and normalised by the sum of the weights of the pixels
 * that lie on the map. The disc normal is the z axis.
 *
 * Particle positions are those of the last drift, which for inactive
 * particles may lag the current time by their time-step; the resulting
 * displacement is small compared with a pixel.
 */

/* Config parameters. */
#include <config.h>

struct engine;
struct swift_params;
struct unit_system;
struct phys_const;

/**
 * @brief Parameters of the map (internal units).
 */
struct prfm_map_params {

  /*! Disc centre (internal units); relative to the box centre unless
   * centre_is_absolute */
  double centre[3];

  /*! Is the centre an absolute position? */
  int centre_is_absolute;

  /*! Pixel side */
  double pixel;

  /*! Radius of the averaging cylinder R_cyl */
  double R_cyl;

  /*! Half-height of the slab mapped */
  double half_height;

  /*! Half-width of the (square) map */
  double half_width;

  /*! Number of pixels per side */
  int npix;

  /*! Evaluate rho_d from the Hernquist potential (else rho_d = 0) */
  int use_dark_matter;
};

void prfm_map_params_init(struct prfm_map_params *mp,
                          struct swift_params *params,
                          const struct phys_const *phys_const);

void prfm_map_params_print(const struct prfm_map_params *mp);

void prfm_map_build(struct engine *e, const struct prfm_map_params *mp);

void prfm_map_sample(const double x[3], double *Sigma_g, double *Sigma_s,
                     double *sigma_sz, double *rho_d);

void prfm_map_clean(void);

#endif /* SWIFT_PRFM_MAP_H */
