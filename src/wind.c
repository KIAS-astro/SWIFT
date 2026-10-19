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
#include "engine.h"
#include "error.h"
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

  /* Wind:loading_set (wind-design.md Sec. 10.3) sets the defaults of the
   * hot and warm norms and of the CR energy loading: K20 (no CRs) or
   * TIGRESSpp (K26 with CRs at 2 kpc) */
  char set[32];
  parser_get_opt_param_string(params, "Wind:loading_set", set, "K20");
  int tigresspp = 0;
  if (strcmp(set, "TIGRESSpp") == 0)
    tigresspp = 1;
  else if (strcmp(set, "K20") != 0)
    error("Wind:loading_set must be K20 or TIGRESSpp, got '%s'", set);

  props->warm_phase = parser_get_opt_param_int(params, "Wind:warm_phase", 0);

  props->eta_M = 0.;
  props->eta_E = 0.;
  if (props->model == wind_loading_constant) {
    props->eta_M = parser_get_param_double(params, "Wind:eta_M");
    props->eta_E = parser_get_param_double(params, "Wind:eta_E");
    if (props->eta_M < 0. || (props->eta_M == 0. && !props->warm_phase))
      error("Wind:eta_M must be positive (or 0 with the warm phase on)");
  }

  props->eta_M_log10_norm = parser_get_opt_param_double(
      params, "Wind:eta_M_log10_norm",
      WIND_TIGRESS_ETA_M_LOG10_NORM +
          (tigresspp ? log10(WIND_TIGRESSPP_HOT_ETA_M_FACTOR) : 0.));
  props->eta_M_slope = parser_get_opt_param_double(params, "Wind:eta_M_slope",
                                                   WIND_TIGRESS_ETA_M_SLOPE);
  props->eta_E_log10_norm = parser_get_opt_param_double(
      params, "Wind:eta_E_log10_norm",
      WIND_TIGRESS_ETA_E_LOG10_NORM +
          (tigresspp ? log10(WIND_TIGRESSPP_HOT_ETA_E_FACTOR) : 0.));
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

  /* Warm channel: loading, kick, temperature (mu = 0.6), recoupling */
  props->warm_eta_M =
      parser_get_opt_param_double(params, "Wind:warm_eta_M", -1.);
  props->warm_eta_M_log10_norm = parser_get_opt_param_double(
      params, "Wind:warm_eta_M_log10_norm",
      tigresspp ? WIND_WARM_ETA_M_LOG10_NORM_TIGRESSPP
                : WIND_WARM_ETA_M_LOG10_NORM_K20);
  props->warm_eta_M_slope = parser_get_opt_param_double(
      params, "Wind:warm_eta_M_slope", WIND_WARM_ETA_M_SLOPE);
  const double kms = 1e5 / units_cgs_conversion_factor(us, UNIT_CONV_VELOCITY);
  props->warm_v =
      parser_get_opt_param_double(params, "Wind:warm_launch_speed_kms", 100.) *
      kms;
  const double T_warm =
      parser_get_opt_param_double(params, "Wind:warm_temperature_K", 1e4);
  props->warm_u = phys_const->const_boltzmann_k * T_warm /
                  (hydro_gamma_minus_one * 0.6 * phys_const->const_proton_mass);
  props->warm_d_rec = parser_get_opt_param_double(
                          params, "Wind:warm_recoupling_distance_kpc", 1.5) *
                      kpc;

  /* CR energy carried by the wind particles */
  props->cr_eta_E = parser_get_opt_param_double(
      params, "Wind:cr_eta_E", tigresspp ? WIND_TIGRESSPP_CR_ETA_E : 0.);
  char carriers[32];
  parser_get_opt_param_string(params, "Wind:cr_carriers", carriers, "hot");
  if (strcmp(carriers, "hot") == 0)
    props->cr_carriers_both = 0;
  else if (strcmp(carriers, "both") == 0)
    props->cr_carriers_both = 1;
  else
    error("Wind:cr_carriers must be hot or both, got '%s'", carriers);
  char deposit[32];
  parser_get_opt_param_string(params, "Wind:cr_deposit", deposit, "thermal");
  if (strcmp(deposit, "thermal") == 0)
    props->cr_deposit_thermal = 1;
  else if (strcmp(deposit, "none") == 0)
    props->cr_deposit_thermal = 0;
  else
    error("Wind:cr_deposit must be thermal or none, got '%s'", deposit);
  if (props->cr_eta_E > 0. && props->model == wind_loading_constant &&
      props->eta_M == 0. && !props->cr_carriers_both)
    error("Wind:cr_eta_E > 0 needs a hot phase or cr_carriers: both");

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
  if (props->warm_phase) {
    if (props->warm_eta_M > 0.)
      message("Wind: warm phase, eta_M,warm = %.4g", props->warm_eta_M);
    else
      message("Wind: warm phase, eta_M,warm = 10^%.3g Sigma_SFR^%.3g",
              props->warm_eta_M_log10_norm, props->warm_eta_M_slope);
    message(
        "Wind: warm kick %.4g, u_warm %.4g, recoupling at max(%.4g, kernel) "
        "or t_rec or the apex (internal units)",
        props->warm_v, props->warm_u, props->warm_d_rec);
  }
  if (props->cr_eta_E > 0.)
    message("Wind: CR energy loading %.4g carried by %s, deposit %s",
            props->cr_eta_E, props->cr_carriers_both ? "hot and warm" : "hot",
            props->cr_deposit_thermal ? "thermal" : "none");
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

  double q[10] = {wind_global_stats.n_launch_hot,
                  wind_global_stats.m_launch_hot,
                  wind_global_stats.n_launch_warm,
                  wind_global_stats.m_launch_warm,
                  wind_global_stats.E_cr_launch,
                  wind_global_stats.E_cr_release,
                  wind_global_stats.E_cr_deposit,
                  wind_global_stats.n_recouple_apex,
                  wind_global_stats.m_over_eta_hot,
                  wind_global_stats.m_over_eta_warm};
#ifdef WITH_MPI
  double rq[10];
  MPI_Reduce(q, rq, 10, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  memcpy(q, rq, sizeof(q));
#endif
  if (e->nodeID == 0)
    message(
        "Wind phases at t=%.6e: hot N=%.0f M=%.6e; warm N=%.0f M=%.6e; CR "
        "launched=%.17e released=%.17e deposited=%.17e; apex recouplings "
        "N=%.0f; sum m/eta hot=%.6e warm=%.6e (internal units)",
        e->time, q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9]);
}

#endif /* WIND_TIGRESS */
