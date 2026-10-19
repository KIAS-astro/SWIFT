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
#include "hydro.h"
#include "parser.h"
#include "physical_constants.h"
#include "restart.h"
#include "units.h"
#include "wind_physics.h"

/* MPI headers. */
#ifdef WITH_MPI
#include <mpi.h>
#endif

#include <float.h>
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

  /* Wind:mode: particle (whole gas particles launched by the star formation
   * draw) or light (light particles spawned at rebuilds from reservoirs) */
  char mode[32];
  parser_get_opt_param_string(params, "Wind:mode", mode, "particle");
  if (strcmp(mode, "particle") == 0)
    props->light = 0;
  else if (strcmp(mode, "light") == 0)
    props->light = 1;
  else
    error("Wind:mode must be particle or light, got '%s'", mode);

  props->light_m_g = 0.;
  props->light_m_l = 0.;
  props->light_N = 1.;
  props->light_max_mass_ratio = parser_get_opt_param_double(
      params, "Wind:light_max_mass_ratio", 10.);
  props->light_R200 =
      parser_get_opt_param_double(params, "Wind:light_R200_kpc", 97.9) * kpc;
  props->light_t0 =
      parser_get_opt_param_double(params, "Wind:light_t0_Myr", 10.) * Myr;
  props->light_k = parser_get_opt_param_double(params, "Wind:light_k", 0.25);
  props->light_fw_thresh =
      parser_get_opt_param_double(params, "Wind:light_fw_thresh", 1000.);
  props->light_fw_ref =
      parser_get_opt_param_double(params, "Wind:light_fw_ref", 0.1);
  props->light_rebuild_interval =
      parser_get_opt_param_double(params, "Wind:light_rebuild_interval_Myr",
                                  1.) *
      Myr;
  props->light_last_rebuild = -FLT_MAX;
  props->source_radius = 0.;
  props->source_Mdot = 0.;
  props->source_Edot = 0.;
  props->source_m_carry = 0.;
  for (int k = 0; k < 3; k++) props->source_centre[k] = 0.;

  if (props->light) {
    props->light_m_g =
        parser_get_param_double(params, "Wind:light_gas_mass_Msun") *
        phys_const->const_solar_mass;
    props->light_N = parser_get_param_double(params, "Wind:light_N");
    if (props->light_N < 1.) error("Wind:light_N must be at least 1");
    if (props->light_N > props->light_max_mass_ratio)
      error(
          "Wind:light_N = %g exceeds Wind:light_max_mass_ratio = %g: SPH and "
          "MFM errors grow with the mass ratio of neighbours",
          props->light_N, props->light_max_mass_ratio);
    props->light_m_l = props->light_m_g / props->light_N;

    /* Optional central source (Chevalier & Clegg 1985 test) */
    props->source_Mdot =
        parser_get_opt_param_double(params, "Wind:light_source_Mdot_Msun_p_yr",
                                    0.) *
        phys_const->const_solar_mass / phys_const->const_year;
    props->source_Edot =
        parser_get_opt_param_double(params, "Wind:light_source_Edot_erg_p_s",
                                    0.) /
        energy_cgs * units_cgs_conversion_factor(us, UNIT_CONV_TIME);
    props->source_radius =
        parser_get_opt_param_double(params, "Wind:light_source_radius_kpc",
                                    0.) *
        kpc;
    parser_get_opt_param_double_array(params, "Wind:light_source_centre", 3,
                                      props->source_centre);
    if (props->source_Mdot > 0. && props->source_Edot <= 0.)
      error("Wind:light_source_Edot_erg_p_s must be positive with a source");
  }

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
  if (props->light) {
    message(
        "Wind: light particles, m_g = %.6g, N = %g, m_l = %.6g (internal "
        "units), mass ratio cap %g, rebuild every %.4g",
        props->light_m_g, props->light_N, props->light_m_l,
        props->light_max_mass_ratio, props->light_rebuild_interval);
    message(
        "Wind: light clock R_200 = %.4g, t_0 = %.4g, k = %g, f_w,thresh = %g, "
        "f_w,ref = %g",
        props->light_R200, props->light_t0, props->light_k,
        props->light_fw_thresh, props->light_fw_ref);
    if (props->source_Mdot > 0.)
      message(
          "Wind: light source at (%.4g, %.4g, %.4g), radius %.4g, Mdot = "
          "%.6g, Edot = %.6g (internal units)",
          props->source_centre[0], props->source_centre[1],
          props->source_centre[2], props->source_radius, props->source_Mdot,
          props->source_Edot);
  }
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

  if (!e->wind_props->light) return;

  /* Light mode: reservoirs still held, and the light population */
  double held[4] = {0., 0., 0., 0.};
  const struct space *sp = e->s;
  for (size_t i = 0; i < sp->nr_parts; i++) {
    const struct part *p = &sp->parts[i];
    if (p->time_bin == time_bin_inhibited ||
        p->time_bin == time_bin_not_created)
      continue;
    held[0] += p->wind_data.m_pending;
    if (p->wind_data.light) {
      held[1] += 1.;
      held[2] += hydro_get_mass(p);
      if (p->wind_data.decoupled) held[3] += 1.;
    }
  }
  const struct wind_global_stats *g = &wind_global_stats;
  double l[17] = {g->m_pending_added, g->E_pending_added,
                  g->m_pending_to_stars, g->m_rounding,
                  g->n_spawn,         g->m_spawn,
                  g->m_source,        g->n_merge,
                  g->m_merge,         g->E_merge_heat,
                  held[0],            held[1],
                  held[2],            held[3],
                  g->P_launch[0],     g->P_launch[1],
                  g->P_launch[2]};
#ifdef WITH_MPI
  double lr[17];
  MPI_Reduce(l, lr, 17, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  memcpy(l, lr, sizeof(l));
#endif
  if (e->nodeID == 0)
    message(
        "Light totals at t=%.6e: pending added M=%.10e E=%.10e; to stars "
        "M=%.10e; rounding M=%.10e; spawned N=%.0f M=%.10e; source M=%.10e; "
        "merged N=%.0f M=%.10e E_heat=%.10e; pending held M=%.10e; light "
        "alive N=%.0f M=%.10e decoupled N=%.0f; launch kicks Px=%.10e "
        "Py=%.10e Pz=%.10e (internal units)",
        e->time, l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8], l[9],
        l[10], l[11], l[12], l[13], l[14], l[15], l[16]);
}

#endif /* WIND_TIGRESS */
