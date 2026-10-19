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
#include "engine.h"
#include "error.h"
#include "lock.h"
#include "parser.h"
#include "physical_constants.h"
#include "restart.h"
#include "units.h"
#include "wind_physics.h"

/* MPI headers. */
#ifdef WITH_MPI
#include <mpi.h>
#endif

#include <math.h>
#include <string.h>

/*! Launch and recoupling totals on this rank */
struct wind_global_stats wind_global_stats;

/*! 1 when the run uses wind carriers */
int wind_carriers_on = 0;

/*! One line per carrier deposit (wind_carriers.txt, or
 * wind_carriers_<rank>.txt with MPI) */
static FILE *wind_carrier_file = NULL;
static swift_lock_type wind_carrier_file_lock;

/**
 * @brief Append one line to the carrier event file, opening it on first use.
 */
void wind_carrier_log_event(const char *line) {
  lock_lock(&wind_carrier_file_lock);
  if (wind_carrier_file == NULL) {
    char name[64] = "wind_carriers.txt";
#ifdef WITH_MPI
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    sprintf(name, "wind_carriers_%d.txt", rank);
#endif
    wind_carrier_file = fopen(name, "a");
    if (wind_carrier_file == NULL) error("Could not open %s", name);
    fprintf(wind_carrier_file,
            "# id t_launch t_deposit criterion(1=distance,2=time) "
            "x_launch[3] x_deposit[3] distance d_rec mass v[3] u h n_ngb "
            "weight_sum (internal units)\n");
  }
  fputs(line, wind_carrier_file);
  fflush(wind_carrier_file);
  if (lock_unlock(&wind_carrier_file_lock) != 0)
    error("Failed to unlock the wind carrier file.");
}

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

  props->f_kin = parser_get_opt_param_double(params, "Wind:kinetic_fraction",
                                             WIND_KINETIC_FRACTION_MACH1);

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

  char mode[32] = "particle";
  parser_get_opt_param_string(params, "Wind:mode", mode, "particle");
  if (strcmp(mode, "particle") == 0)
    props->carriers = 0;
  else if (strcmp(mode, "carrier") == 0)
    props->carriers = 1;
  else
    error("Wind:mode must be particle or carrier, got '%s'", mode);
  props->carriers_per_particle =
      parser_get_opt_param_double(params, "Wind:carriers_per_particle", 1.);
  if (props->carriers_per_particle < 1.)
    error("Wind:carriers_per_particle must be >= 1");

#if !defined(FEEDBACK_NONE)
  if (props->carriers)
    error(
        "Wind:mode = carrier needs --with-feedback=none (the carriers use its "
        "loops)");
#endif

  wind_carriers_on = props->carriers;
  lock_init(&wind_carrier_file_lock);
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
      "Wind: kinetic fraction %.4g, launch direction +-(%.3g, %.3g, %.3g), "
      "u_* = %.4g (internal units)",
      props->f_kin, props->normal[0], props->normal[1], props->normal[2],
      props->u_star);
  message(
      "Wind: recoupling at d = max(%.4g, %.3g kernel supports at launch) or "
      "t = %.4g (internal units); time-step cap %.4g",
      props->d_rec, props->d_rec_kernel_factor, props->t_rec, props->dt_max);
  if (props->carriers)
    message(
        "Wind: carried by wind carriers, N_c = %.4g carriers per gas particle "
        "mass",
        props->carriers_per_particle);
  else
    message("Wind: carried by whole gas particles");
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
  wind_carriers_on = props->carriers;
  lock_init(&wind_carrier_file_lock);
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
        "recoupled N=%.0f M=%.6e E_kin=%.6e E_th=%.6e (internal units)",
        e->time, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]);

  if (!e->wind_props->carriers) return;
  const struct wind_global_stats *g = &wind_global_stats;
  double c[22] = {g->c_n_launch,      g->c_m_launch,      g->c_p_launch[0],
                  g->c_p_launch[1],   g->c_p_launch[2],   g->c_E_kin_launch,
                  g->c_E_th_launch,   g->c_E_launch_net,  g->c_Z_launch,
                  g->c_n_failed,      g->c_n_deposit,     g->c_n_by_time,
                  g->c_m_deposit,     g->c_p_deposit[0],  g->c_p_deposit[1],
                  g->c_p_deposit[2],  g->c_E_kin_deposit, g->c_E_th_deposit,
                  g->c_E_pot_deposit, g->c_E_floor,       g->c_Z_deposit,
                  g->c_n_retry};
#ifdef WITH_MPI
  double cr[22];
  MPI_Reduce(c, cr, 22, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  memcpy(c, cr, sizeof(c));
#endif
  if (e->nodeID == 0)
    message(
        "Wind carrier totals at t=%.6e: launched N=%.0f M=%.6e "
        "P=(%.6e,%.6e,%.6e) E_kin=%.6e E_th=%.6e E_net=%.6e M_Z=%.6e "
        "failed=%.0f; deposited N=%.0f by_time=%.0f M=%.6e "
        "P=(%.6e,%.6e,%.6e) E_kin=%.6e E_th=%.6e E_pot=%.6e E_floor=%.6e "
        "M_Z=%.6e retries=%.0f (internal units)",
        e->time, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9],
        c[10], c[11], c[12], c[13], c[14], c[15], c[16], c[17], c[18], c[19],
        c[20], c[21]);
}

#endif /* WIND_TIGRESS */
