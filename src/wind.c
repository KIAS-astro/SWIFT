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

/**
 * @file src/wind.c
 * @brief Decoupled wind: parameters, restart dump and global counters.
 */

/* Config parameters. */
#include <config.h>

/* This object's header. */
#include "wind_properties.h"

#ifdef WIND_TIGRESS

/* Local includes. */
#include "adiabatic_index.h"
#include "clocks.h"
#include "engine.h"
#include "error.h"
#include "hydro.h"
#include "parser.h"
#include "physical_constants.h"
#include "restart.h"
#include "space.h"
#include "units.h"
#include "wind.h"
#include "wind_physics.h"
#include "wind_recoupling_physics.h"

/* MPI headers. */
#ifdef WITH_MPI
#include <mpi.h>
#endif

#include <math.h>
#include <string.h>

/*! Launch and recoupling totals on this rank */
struct wind_global_stats wind_global_stats;

/**
 * @brief Read the Wind section of the parameter file.
 *
 * @param props The properties to fill.
 * @param phys_const The physical constants in internal units.
 * @param us The internal unit system.
 * @param params The parsed parameter file.
 */
void wind_props_init(struct wind_props *props,
                     const struct phys_const *phys_const,
                     const struct unit_system *us,
                     struct swift_params *params) {

  char model[32] = "TIGRESS";
  parser_get_opt_param_string(params, "Wind:loading_model", model, "TIGRESS");
  if (strcmp(model, "TIGRESS") == 0)
    props->model = wind_loading_TIGRESS;
  else if (strcmp(model, "constant") == 0)
    props->model = wind_loading_constant;
  else if (strcmp(model, "none") == 0)
    props->model = wind_loading_none;
  else
    error("Wind:loading_model must be TIGRESS, constant or none, got '%s'",
          model);

  props->eta_M = 0.;
  props->eta_E = 0.;
  if (props->model == wind_loading_constant) {
    props->eta_M = parser_get_param_double(params, "Wind:eta_M");
    props->eta_E = parser_get_param_double(params, "Wind:eta_E");
    if (props->eta_M <= 0.) error("Wind:eta_M must be positive");
  }

  props->eta_M_log10_norm = parser_get_opt_param_double(
      params, "Wind:eta_M_log10_norm", WIND_TIGRESS_ETA_M_LOG10_NORM);
  props->eta_M_slope = parser_get_opt_param_double(params, "Wind:eta_M_slope",
                                                   WIND_TIGRESS_ETA_M_SLOPE);
  props->eta_E_log10_norm = parser_get_opt_param_double(
      params, "Wind:eta_E_log10_norm", WIND_TIGRESS_ETA_E_LOG10_NORM);
  props->eta_E_slope = parser_get_opt_param_double(params, "Wind:eta_E_slope",
                                                   WIND_TIGRESS_ETA_E_SLOPE);

  const double energy_cgs = units_cgs_conversion_factor(us, UNIT_CONV_ENERGY);
  props->u_star = 1e51 / energy_cgs / (95.5 * phys_const->const_solar_mass);

  /* Wind:energy_convention: what eta_E counts at launch.
   *   enthalpy (default for loading_model TIGRESS): Delta v^2/2 + gamma u_w
   *     = eta_E u_* / eta_M, the K20b convention (kinetic + enthalpy flux);
   *   internal (default for loading_model constant and none): Delta v^2/2 +
   *     u_w = eta_E u_* / eta_M, the Arkenstone convention.
   * Wind:kinetic_fraction = (Delta v^2/2) / (eta_E u_* / eta_M); its default
   * is the Mach-1 value, (gamma-1)/(gamma+1) = 1/4 for enthalpy and
   * gamma(gamma-1)/(2+gamma(gamma-1)) = 5/14 for internal (gamma = 5/3). */
  char conv[32];
  parser_get_opt_param_string(
      params, "Wind:energy_convention", conv,
      props->model == wind_loading_TIGRESS ? "enthalpy" : "internal");
  if (strcmp(conv, "enthalpy") == 0)
    props->energy_convention = wind_energy_enthalpy;
  else if (strcmp(conv, "internal") == 0)
    props->energy_convention = wind_energy_internal;
  else
    error("Wind:energy_convention must be enthalpy or internal, got '%s'",
          conv);

  props->f_kin = parser_get_opt_param_double(
      params, "Wind:kinetic_fraction",
      wind_kinetic_fraction_mach1(props->energy_convention, hydro_gamma));
  if (props->f_kin < 0. || props->f_kin > 1.)
    error("Wind:kinetic_fraction must be in [0, 1]");

  double n[3] = {0., 0., 1.};
  parser_get_opt_param_double_array(params, "Wind:launch_direction", 3, n);
  const double norm = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (norm == 0.) error("Wind:launch_direction must not be zero");
  for (int k = 0; k < 3; k++) props->normal[k] = n[k] / norm;

  const double kpc = 1e3 * phys_const->const_parsec;
  const double Myr = 1e6 * phys_const->const_year;
  props->d_rec =
      parser_get_opt_param_double(params, "Wind:recoupling_distance_kpc", 2.) *
      kpc;
  props->d_rec_kernel_factor = parser_get_opt_param_double(
      params, "Wind:recoupling_kernel_factor", 2.);
  props->t_rec =
      parser_get_opt_param_double(params, "Wind:recoupling_time_Myr", 25.) *
      Myr;
  props->dt_max =
      parser_get_opt_param_double(params, "Wind:max_timestep_Myr", 1.) * Myr;
  props->kpc = kpc;

  /* Experiment A (wind-design.md Sec. 9) */
  char rec[32];
  parser_get_opt_param_string(params, "Wind:recoupling", rec, "distance");
  if (strcmp(rec, "distance") == 0)
    props->recoupling = wind_recoupling_by_distance;
  else if (strcmp(rec, "pressure") == 0)
    props->recoupling = wind_recoupling_by_pressure;
  else
    error("Wind:recoupling must be distance or pressure, got '%s'", rec);

  char lh[32];
  parser_get_opt_param_string(params, "Wind:loading_height", lh, "H");
  if (strcmp(lh, "H") == 0)
    props->loading_height_recoupling = 0;
  else if (strcmp(lh, "recoupling") == 0)
    props->loading_height_recoupling = 1;
  else
    error("Wind:loading_height must be H or recoupling, got '%s'", lh);
  if (props->loading_height_recoupling &&
      props->recoupling != wind_recoupling_by_distance)
    error("Wind:loading_height = recoupling needs Wind:recoupling = distance");

  props->cc85_R =
      parser_get_opt_param_double(
          params, "Wind:cc85_launch_radius_kpc",
          parser_get_opt_param_double(
              params, "PRFMStarFormation:cylinder_radius_kpc", 0.5)) *
      kpc;
  props->d_min = parser_get_opt_param_double(
                     params, "Wind:recoupling_min_distance_kpc", 1.) *
                 kpc;
  props->Sigma_SFR_to_internal =
      phys_const->const_solar_mass / (kpc * kpc * phys_const->const_year);

  double c[3] = {0., 0., 0.};
  parser_get_opt_param_double_array(params, "Wind:ambient_centre", 3, c);
  for (int k = 0; k < 3; k++) props->ambient_centre[k] = c[k];
  props->ambient_centre_is_absolute =
      parser_get_opt_param_int(params, "Wind:ambient_centre_is_absolute", 0);
  /* rho = n_H m_H / X with the primordial X = 0.752; n_H per cm^3 times the
   * internal volume unit in cm^3 is n_H per internal volume */
  props->ambient_rho_max =
      parser_get_opt_param_double(params, "Wind:ambient_max_nH_cm3", 0.1) *
      units_cgs_conversion_factor(us, UNIT_CONV_VOLUME) *
      phys_const->const_proton_mass / 0.752;
  props->ambient_r_min = 0.1 * kpc;
  props->ambient_r_max =
      parser_get_opt_param_double(params, "Wind:ambient_max_radius_kpc",
                                  1000.) *
      kpc;

  bzero(&wind_global_stats, sizeof(struct wind_global_stats));
}

/**
 * @brief Print the wind parameters.
 */
void wind_props_print(const struct wind_props *props) {
  const char *names[3] = {"none", "constant", "TIGRESS"};
  message("Decoupled wind: loadings '%s'", names[props->model]);
  if (props->model == wind_loading_constant)
    message("Wind: eta_M = %.4g, eta_E = %.4g", props->eta_M, props->eta_E);
  if (props->model == wind_loading_TIGRESS)
    message(
        "Wind: eta_M = 10^%.3g Sigma_SFR^%.3g, eta_E = 10^%.3g Sigma_SFR^%.3g "
        "(Sigma_SFR in Msun/kpc^2/yr, clipped to [1e-4, 1])",
        props->eta_M_log10_norm, props->eta_M_slope, props->eta_E_log10_norm,
        props->eta_E_slope);
  message(
      "Wind: eta_E counts kinetic + %s, kinetic fraction %.4g, launch "
      "direction +-(%.3g, %.3g, %.3g), u_* = %.4g (internal units)",
      props->energy_convention == wind_energy_enthalpy ? "enthalpy"
                                                       : "internal energy",
      props->f_kin, props->normal[0], props->normal[1], props->normal[2],
      props->u_star);
  message(
      "Wind: recoupling at d = max(%.4g, %.3g kernel supports at launch) or "
      "t = %.4g (internal units); time-step cap %.4g",
      props->d_rec, props->d_rec_kernel_factor, props->t_rec, props->dt_max);
  if (props->loading_height_recoupling)
    message(
        "Wind: loadings times the K20b factors eta(z = d_rec) / eta(H) "
        "(500 pc and 1 kpc fits)");
  if (props->recoupling == wind_recoupling_by_pressure)
    message(
        "Wind: pressure recoupling: CC85 R = %.4g, d_min = max(%.4g, kernel "
        "factor); d_rec above is the backstop; ambient pressure from gas "
        "with rho < %.4g in log shells %.4g-%.4g about (%.4g, %.4g, %.4g)%s "
        "(internal units)",
        props->cc85_R, props->d_min, props->ambient_rho_max,
        props->ambient_r_min, props->ambient_r_max, props->ambient_centre[0],
        props->ambient_centre[1], props->ambient_centre[2],
        props->ambient_centre_is_absolute ? "" : " + box centre");
}

/*! Number of shells of the ambient-pressure profile */
#define WIND_AMBIENT_NBIN 128

/**
 * @brief The ambient-pressure profile of the current step: one per rank,
 * built before the tasks of a step and only read by them.
 */
static struct {
  double P[WIND_AMBIENT_NBIN];
  double c[3], dim[3];
  double log_r_min, inv_dlog;
  int periodic, ready;
} wind_ambient;

/**
 * @brief Separation along one axis, wrapped if the box is periodic.
 */
static double wind_ambient_wrap(double dx, const double dim,
                                const int periodic) {
  if (periodic) {
    if (dx > 0.5 * dim) dx -= dim;
    if (dx < -0.5 * dim) dx += dim;
  }
  return dx;
}

/**
 * @brief Build the volume-weighted pressure of the coupled, non-ISM gas in
 * log spherical shells about the ambient centre (wind-design.md Sec. 9.2).
 *
 * Every rank sums V = m / rho and V P over its particles; the sums are
 * reduced over ranks. Empty shells take the nearest non-empty one.
 *
 * @param e The #engine.
 */
void wind_ambient_build(struct engine *e) {

  const struct wind_props *props = e->wind_props;
  if (props == NULL || props->recoupling != wind_recoupling_by_pressure) return;

  const ticks tic = getticks();
  const struct space *s = e->s;
  for (int k = 0; k < 3; k++) {
    wind_ambient.c[k] = props->ambient_centre[k] +
                        (props->ambient_centre_is_absolute ? 0. : 0.5 * s->dim[k]);
    wind_ambient.dim[k] = s->dim[k];
  }
  wind_ambient.periodic = s->periodic;
  wind_ambient.log_r_min = log(props->ambient_r_min);
  wind_ambient.inv_dlog =
      WIND_AMBIENT_NBIN / (log(props->ambient_r_max) - wind_ambient.log_r_min);

  double sums[2 * WIND_AMBIENT_NBIN];
  memset(sums, 0, sizeof(sums));

  for (size_t k = 0; k < s->nr_parts; k++) {
    const struct part *p = &s->parts[k];
    if (p->time_bin == time_bin_inhibited ||
        p->time_bin == time_bin_not_created)
      continue;
    if (part_is_decoupled(p)) continue;
    const double rho = hydro_get_comoving_density(p);
    if (rho <= 0. || rho >= props->ambient_rho_max) continue;
    double r2 = 0.;
    for (int i = 0; i < 3; i++) {
      const double dx = wind_ambient_wrap(p->x[i] - wind_ambient.c[i],
                                          s->dim[i], s->periodic);
      r2 += dx * dx;
    }
    const int b = (int)floor((0.5 * log(fmax(r2, 1e-300)) -
                              wind_ambient.log_r_min) *
                             wind_ambient.inv_dlog);
    if (b < 0 || b >= WIND_AMBIENT_NBIN) continue;
    const double V = hydro_get_mass(p) / rho;
    sums[2 * b] += V;
    sums[2 * b + 1] += V * hydro_get_comoving_pressure(p);
  }

#ifdef WITH_MPI
  if (MPI_Allreduce(MPI_IN_PLACE, sums, 2 * WIND_AMBIENT_NBIN, MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS)
    error("Failed to reduce the ambient-pressure profile");
#endif

  int any = 0;
  for (int b = 0; b < WIND_AMBIENT_NBIN; b++) {
    wind_ambient.P[b] = sums[2 * b] > 0. ? sums[2 * b + 1] / sums[2 * b] : -1.;
    if (sums[2 * b] > 0.) any = 1;
  }
  /* Fill empty shells from the nearest non-empty one (inner on a tie) */
  if (any) {
    double filled[WIND_AMBIENT_NBIN];
    for (int b = 0; b < WIND_AMBIENT_NBIN; b++) {
      if (wind_ambient.P[b] >= 0.) {
        filled[b] = wind_ambient.P[b];
        continue;
      }
      for (int d = 1; d < WIND_AMBIENT_NBIN; d++) {
        if (b - d >= 0 && wind_ambient.P[b - d] >= 0.) {
          filled[b] = wind_ambient.P[b - d];
          break;
        }
        if (b + d < WIND_AMBIENT_NBIN && wind_ambient.P[b + d] >= 0.) {
          filled[b] = wind_ambient.P[b + d];
          break;
        }
      }
    }
    memcpy(wind_ambient.P, filled, sizeof(filled));
  } else {
    for (int b = 0; b < WIND_AMBIENT_NBIN; b++) wind_ambient.P[b] = 0.;
  }
  wind_ambient.ready = 1;

  if (e->verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}

/**
 * @brief Ambient pressure at a position: the profile's value in the shell
 * of the position (inside r_min the first shell, outside r_max the last);
 * 0 before the first build.
 */
double wind_ambient_pressure(const double x[3]) {
  if (!wind_ambient.ready) return 0.;
  double r2 = 0.;
  for (int i = 0; i < 3; i++) {
    const double dx = wind_ambient_wrap(x[i] - wind_ambient.c[i],
                                        wind_ambient.dim[i],
                                        wind_ambient.periodic);
    r2 += dx * dx;
  }
  int b = (int)floor((0.5 * log(fmax(r2, 1e-300)) - wind_ambient.log_r_min) *
                     wind_ambient.inv_dlog);
  if (b < 0) b = 0;
  if (b >= WIND_AMBIENT_NBIN) b = WIND_AMBIENT_NBIN - 1;
  return wind_ambient.P[b];
}

/**
 * @brief Write the properties and the counters to a restart file.
 */
void wind_struct_dump(const struct wind_props *props, FILE *stream) {
  restart_write_blocks((void *)props, sizeof(struct wind_props), 1, stream,
                       "wind_props", "wind properties");
  restart_write_blocks((void *)&wind_global_stats,
                       sizeof(struct wind_global_stats), 1, stream,
                       "wind_stats", "wind counters");
}

/**
 * @brief Read the properties and the counters from a restart file.
 */
void wind_struct_restore(struct wind_props *props, FILE *stream) {
  restart_read_blocks((void *)props, sizeof(struct wind_props), 1, stream,
                      NULL, "wind properties");
  restart_read_blocks((void *)&wind_global_stats,
                      sizeof(struct wind_global_stats), 1, stream, NULL,
                      "wind counters");
}

/**
 * @brief Print the totals of all ranks as one log line.
 */
void wind_print_stats(const struct engine *e) {
  double s[8] = {wind_global_stats.n_launch,     wind_global_stats.m_launch,
                 wind_global_stats.E_kin_launch, wind_global_stats.E_th_launch,
                 wind_global_stats.n_recouple,   wind_global_stats.m_recouple,
                 wind_global_stats.E_kin_recouple,
                 wind_global_stats.E_th_recouple};
#ifdef WITH_MPI
  double r[8];
  MPI_Reduce(s, r, 8, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  memcpy(s, r, sizeof(s));
#endif
  if (e->nodeID == 0)
    message(
        "Wind totals at t=%.6e: launched N=%.0f M=%.6e E_kin=%.6e E_th=%.6e; "
        "recoupled N=%.0f M=%.6e E_kin=%.6e E_th=%.6e; launched "
        "E_kin+E_th=%.6e E_kin+gamma*E_th=%.6e (internal units)",
        e->time, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[2] + s[3],
        s[2] + hydro_gamma * s[3]);
}

#endif /* WIND_TIGRESS */
