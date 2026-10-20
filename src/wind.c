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
#include "cell.h"
#include "engine.h"
#include "error.h"
#include "hydro.h"
#include "kernel_hydro.h"
#include "parser.h"
#include "physical_constants.h"
#include "restart.h"
#include "space.h"
#include "timestep_sync_part.h"
#include "units.h"
#include "wind.h"
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
  props->overshoot_fraction = parser_get_opt_param_double(
      params, "Wind:recoupling_overshoot_fraction", 0.02);

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
  char speed_pdf[32];
  parser_get_opt_param_string(params, "Wind:warm_launch_speed_pdf", speed_pdf,
                              "fixed");
  if (strcmp(speed_pdf, "fixed") == 0)
    props->warm_speed_pdf = 0;
  else if (strcmp(speed_pdf, "K20a") == 0)
    props->warm_speed_pdf = 1;
  else
    error("Wind:warm_launch_speed_pdf must be fixed or K20a, got '%s'",
          speed_pdf);
  props->warm_pdf_v0 =
      parser_get_opt_param_double(params, "Wind:warm_pdf_v0_kms", 60.) * kms;
  props->warm_pdf_slope =
      parser_get_opt_param_double(params, "Wind:warm_pdf_slope", 0.23);
  props->warm_pdf_floor =
      parser_get_opt_param_double(params, "Wind:warm_pdf_floor_kms", 3.) * kms;
  props->warm_climb_v =
      parser_get_opt_param_double(params, "Wind:warm_climb_speed_kms", 0.) *
      kms;
  if (props->warm_pdf_v0 <= 0. || props->warm_pdf_floor < 0. ||
      props->warm_climb_v < 0.)
    error(
        "Wind:warm_pdf_v0_kms must be positive, warm_pdf_floor_kms and "
        "warm_climb_speed_kms >= 0");

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

  char mode[32] = "distance";
  parser_get_opt_param_string(params, "Wind:recoupling_mode", mode,
                              "distance");
  if (strcmp(mode, "distance") == 0)
    props->mode = wind_recouple_distance;
  else if (strcmp(mode, "density") == 0)
    props->mode = wind_recouple_density;
  else
    error("Wind:recoupling_mode must be distance or density, got '%s'", mode);

  char rec_deposit[32] = "self";
  parser_get_opt_param_string(params, "Wind:recoupling_deposit", rec_deposit,
                              "self");
  if (strcmp(rec_deposit, "self") == 0)
    props->deposit = wind_deposit_self;
  else if (strcmp(rec_deposit, "host") == 0)
    props->deposit = wind_deposit_host;
  else
    error("Wind:recoupling_deposit must be self or host, got '%s'",
          rec_deposit);

  /* n_H threshold to mass density with the primordial hydrogen fraction
   * 0.752 (Arkenstone: 0.1 rho_SF) */
  const double nH_rec = parser_get_opt_param_double(
      params, "Wind:recoupling_density_H_p_cm3", 0.013);
  const double rho_cgs = nH_rec * 1.67262192e-24 / 0.752;
  props->rho_rec =
      rho_cgs / units_cgs_conversion_factor(us, UNIT_CONV_DENSITY);

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
  message("Wind: decoupled steps overshoot the recoupling distance by at most "
          "%.3g of it",
          props->overshoot_fraction);
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
    if (props->warm_speed_pdf)
      message(
          "Wind: warm kick drawn from the K20a PDF, v_L^2 = v_ref^2 + "
          "v_climb^2, v_ref ~ Gamma(2, v0), v0 = %.4g Sigma_SFR^%.3g + %.4g, "
          "v_climb = %.4g (internal units; replaces the fixed warm kick)",
          props->warm_pdf_v0, props->warm_pdf_slope, props->warm_pdf_floor,
          props->warm_climb_v);
  }
  if (props->cr_eta_E > 0.)
    message("Wind: CR energy loading %.4g carried by %s, deposit %s",
            props->cr_eta_E, props->cr_carriers_both ? "hot and warm" : "hot",
            props->cr_deposit_thermal ? "thermal" : "none");
  message(
      "Wind: hot recoupling mode '%s' (density threshold %.4g internal "
      "units), hot deposit '%s'; the warm phase rejoins as itself",
      props->mode == wind_recouple_density ? "density" : "distance",
      props->rho_rec, props->deposit == wind_deposit_host ? "host" : "self");
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
  double s[12] = {wind_global_stats.n_launch,     wind_global_stats.m_launch,
                  wind_global_stats.E_kin_launch, wind_global_stats.E_th_launch,
                  wind_global_stats.n_recouple,   wind_global_stats.m_recouple,
                  wind_global_stats.E_kin_recouple,
                  wind_global_stats.E_th_recouple,
                  wind_global_stats.n_reason[wind_reason_distance],
                  wind_global_stats.n_reason[wind_reason_density],
                  wind_global_stats.n_reason[wind_reason_time],
                  wind_global_stats.n_merged};
#ifdef WITH_MPI
  double r[12];
  MPI_Reduce(s, r, 12, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  memcpy(s, r, sizeof(s));
#endif
  if (e->nodeID == 0) {
    message(
        "Wind totals at t=%.6e: launched N=%.0f M=%.6e E_kin=%.6e E_th=%.6e; "
        "recoupled N=%.0f M=%.6e E_kin=%.6e E_th=%.6e; launched "
        "E_kin+E_th=%.6e E_kin+gamma*E_th=%.6e (internal units)",
        e->time, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[2] + s[3],
        s[2] + hydro_gamma * s[3]);
    message(
        "Wind recouplings by criterion: distance %.0f density %.0f time %.0f; "
        "merged into a host %.0f",
        s[8], s[9], s[10], s[11]);
  }

  double q[16] = {wind_global_stats.n_launch_hot,
                  wind_global_stats.m_launch_hot,
                  wind_global_stats.n_launch_warm,
                  wind_global_stats.m_launch_warm,
                  wind_global_stats.E_cr_launch,
                  wind_global_stats.E_cr_release,
                  wind_global_stats.E_cr_deposit,
                  wind_global_stats.n_recouple_apex,
                  wind_global_stats.m_over_eta_hot,
                  wind_global_stats.m_over_eta_warm,
                  wind_global_stats.E_launch_hot,
                  wind_global_stats.E_launch_warm,
                  wind_global_stats.p_launch_hot,
                  wind_global_stats.p_launch_warm,
                  wind_global_stats.E_launch_warm_expected,
                  wind_global_stats.E_launch_warm_variance};
#ifdef WITH_MPI
  double rq[16];
  MPI_Reduce(q, rq, 16, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  memcpy(q, rq, sizeof(q));
#endif
  if (e->nodeID == 0)
    message(
        "Wind phases at t=%.6e: hot N=%.0f M=%.6e; warm N=%.0f M=%.6e; CR "
        "launched=%.17e released=%.17e deposited=%.17e; apex recouplings "
        "N=%.0f; sum m/eta hot=%.6e warm=%.6e; launched energy hot=%.10e "
        "warm=%.10e; momentum hot=%.10e warm=%.10e; warm K20a expected "
        "energy=%.10e variance=%.10e (internal units)",
        e->time, q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9],
        q[10], q[11], q[12], q[13], q[14], q[15]);
}

/**
 * @brief Running state of the neighbour walk for one decoupled particle.
 */
struct wind_gather {
  double x[3];
  double rho;
  double r2_min;
  struct part *host;
  struct cell *host_cell;
};

/**
 * @brief Separation along one axis, nearest image in a periodic box.
 */
static double wind_wrap(double dx, const double dim, const int periodic) {
  if (periodic) {
    if (dx > 0.5 * dim) dx -= dim;
    if (dx < -0.5 * dim) dx += dim;
  }
  return dx;
}

/**
 * @brief Walk the cell tree: add every non-decoupled gas particle whose kernel
 * contains g->x to the scatter density, and keep the nearest one.
 */
static void wind_gather_recurse(struct cell *c, const struct space *s,
                                struct wind_gather *g) {
  if (c->hydro.count == 0) return;

  /* Distance from the point to the cell's box */
  double d2 = 0.;
  for (int k = 0; k < 3; k++) {
    const double centre = c->loc[k] + 0.5 * c->width[k];
    const double dx =
        fabs(wind_wrap(g->x[k] - centre, s->dim[k], s->periodic)) -
        0.5 * c->width[k];
    if (dx > 0.) d2 += dx * dx;
  }
  const double reach = kernel_gamma * c->hydro.h_max;
  if (d2 > reach * reach && d2 > g->r2_min) return;

  if (c->split) {
    for (int k = 0; k < 8; k++)
      if (c->progeny[k] != NULL) wind_gather_recurse(c->progeny[k], s, g);
    return;
  }

  for (int i = 0; i < c->hydro.count; i++) {
    struct part *pj = &c->hydro.parts[i];
    if (pj->time_bin == time_bin_inhibited ||
        pj->time_bin == time_bin_not_created)
      continue;
    if (part_is_decoupled(pj)) continue;
    double r2 = 0.;
    for (int k = 0; k < 3; k++) {
      const double dx = wind_wrap(g->x[k] - pj->x[k], s->dim[k], s->periodic);
      r2 += dx * dx;
    }
    const float hj = pj->h;
    if (r2 < kernel_gamma2 * hj * hj) {
      const float hj_inv = 1.f / hj;
      float w;
      kernel_eval(sqrtf((float)r2) * hj_inv, &w);
      g->rho += hydro_get_mass(pj) * w * pow_dimension(hj_inv);
    }
    if (r2 < g->r2_min) {
      g->r2_min = r2;
      g->host = pj;
      g->host_cell = c;
    }
  }
}

/**
 * @brief Merge a wind particle into its host and remove it: mass, momentum
 * and total (kinetic plus internal) energy are conserved; the host takes the
 * mass-weighted mean velocity and the internal energy that closes the energy
 * budget, so the relative kinetic energy becomes heat. This is the SPH
 * counterpart of Arkenstone's standard recoupling into the host cell (Smith
 * et al. 2024, Sec. 3.4.1).
 */
static void wind_merge_into_host(struct engine *e, struct part *p,
                                 struct xpart *xp, struct part *h,
                                 struct xpart *hx, struct cell *hc) {

  const struct cosmology *cosmo = e->cosmology;
  const double mw = hydro_get_mass(p), mh = hydro_get_mass(h);
  const double m = mw + mh;
  const double uw = hydro_get_physical_internal_energy(p, xp, cosmo);
  const double uh = hydro_get_physical_internal_energy(h, hx, cosmo);

  double v[3], ekw = 0., ekh = 0., ek = 0., dv2 = 0.;
  for (int k = 0; k < 3; k++) {
    v[k] = (mw * xp->v_full[k] + mh * hx->v_full[k]) / m;
    ekw += 0.5 * xp->v_full[k] * xp->v_full[k];
    ekh += 0.5 * hx->v_full[k] * hx->v_full[k];
    ek += 0.5 * v[k] * v[k];
    dv2 += (v[k] - hx->v_full[k]) * (v[k] - hx->v_full[k]);
  }
  /* CR energy carried by the particle is released into the host, as heat
   * with Wind:cr_deposit thermal */
  const double E_cr = p->wind_data.cr_energy;
  const double E_cr_heat = e->wind_props->cr_deposit_thermal ? E_cr : 0.;
  if (E_cr > 0.) {
    wind_global_stats.E_cr_release += E_cr;
    wind_global_stats.E_cr_deposit += E_cr_heat;
    p->wind_data.cr_energy = 0.;
  }
  const double u =
      (mw * (uw + ekw) + mh * (uh + ekh) + E_cr_heat) / m - ek;

  /* Record the flight on the particle before it is removed */
  p->wind_data.recouple_time = e->time;
  p->wind_data.recouple_distance = wind_recoupling_measure(p, e->wind_props);
  p->wind_data.recouple_e_kin = ekw;
  p->wind_data.recouple_e_th = uw;
  const int reason = p->wind_data.recouple_pending;
  wind_global_stats.n_reason[reason] += 1.;
  wind_global_stats.n_recouple += 1.;
  wind_global_stats.m_recouple += mw;
  wind_global_stats.E_kin_recouple += mw * ekw;
  wind_global_stats.E_th_recouple += mw * uw;
  wind_global_stats.n_merged += 1.;

  /* An active host may be between hydro_init_part and its density loop (at a
   * rebuild or with --drift-all, everything is drifted before this pass): its
   * density sums, and in SPHENIX the force fields sharing a union with them,
   * are then accumulators for this step and must not be written. */
  const int host_active = h->time_bin <= e->max_active_bin;

  /* Metal mass fractions: mass-weighted mean. The smoothed fractions of an
   * active host are density-loop sums (chemistry_init_part zeroes them), so
   * they are left to the loop; writing a mean into the sum multiplied the
   * hydrogen fraction seen by the cooling by about 1e4. */
#if defined(CHEMISTRY_EAGLE)
  for (int k = 0; k < chemistry_element_count; k++) {
    h->chemistry_data.metal_mass_fraction[k] =
        (mw * p->chemistry_data.metal_mass_fraction[k] +
         mh * h->chemistry_data.metal_mass_fraction[k]) /
        m;
    if (!host_active)
      h->chemistry_data.smoothed_metal_mass_fraction[k] =
          (mw * p->chemistry_data.smoothed_metal_mass_fraction[k] +
           mh * h->chemistry_data.smoothed_metal_mass_fraction[k]) /
          m;
  }
  h->chemistry_data.metal_mass_fraction_total =
      (mw * p->chemistry_data.metal_mass_fraction_total +
       mh * h->chemistry_data.metal_mass_fraction_total) /
      m;
  if (!host_active)
    h->chemistry_data.smoothed_metal_mass_fraction_total =
        (mw * p->chemistry_data.smoothed_metal_mass_fraction_total +
         mh * h->chemistry_data.smoothed_metal_mass_fraction_total) /
        m;
#endif

  hydro_set_mass(h, m);
  if (h->gpart != NULL) h->gpart->mass = m;
  for (int k = 0; k < 3; k++) {
    hx->v_full[k] = v[k];
    h->v[k] = v[k];
    if (h->gpart != NULL) h->gpart->v_full[k] = v[k];
  }
  hydro_set_physical_internal_energy(h, hx, cosmo, u);

  if (host_active) {
    /* In SPHENIX the force fields (pressure, sound speed) share a union with
     * the density sums, so the feedback setters below would write into those
     * sums (with rho = 0 they write NaN). Set only the drifted energy; the
     * ghost recomputes pressure, sound speed and v_sig this step. */
#if defined(SPHENIX_SPH)
    hydro_set_init_internal_energy(h, u / cosmo->a_factor_internal_energy);
#else
    error("Wind: merging into an active host is implemented for SPHENIX only");
#endif
  } else {
    hydro_set_drifted_physical_internal_energy(h, cosmo, NULL, u);
    hydro_set_v_sig_based_on_velocity_kick(h, cosmo, sqrt(dv2));

    /* Put the inactive host on the current step */
    timestep_sync_part(h);
    cell_activate_sync_part(hc, &e->sched);
  }

  /* The particle's mass and gravity are now the host's */
  if (p->gpart != NULL) p->gpart->mass = 0.f;
  cell_remove_part(e, hc, p, xp);
}

/**
 * @brief Before the tasks of a step: for every decoupled particle, the
 * ambient gas density (scatter estimate from the non-decoupled neighbours
 * whose kernels contain it) and, if it met its criterion with deposit into a
 * host, the merge into its nearest gas neighbour.
 *
 * A serial walk of the cell tree per decoupled particle; decoupled particles
 * are rare (tens to hundreds), so the cost is small. Positions are those of
 * the last drift. Single rank only.
 *
 * @param e The #engine.
 */
void wind_engine_pass(struct engine *e) {

  const struct wind_props *props = e->wind_props;
  if (props->mode == wind_recouple_distance &&
      props->deposit == wind_deposit_self)
    return;
  if (e->nr_nodes > 1)
    error("Wind: density recoupling and host deposit need a single rank");
  if (props->deposit == wind_deposit_host &&
      !(e->policy & engine_policy_timestep_sync))
    error("Wind: recoupling_deposit host needs the time-step sync (--sync)");

  struct space *s = e->s;
  for (size_t k = 0; k < s->nr_parts; k++) {
    struct part *p = &s->parts[k];
    struct xpart *xp = &s->xparts[k];
    if (p->time_bin == time_bin_inhibited ||
        p->time_bin == time_bin_not_created)
      continue;
    if (!part_is_decoupled(p)) continue;

    struct wind_gather g;
    for (int i = 0; i < 3; i++) g.x[i] = p->x[i];
    g.rho = 0.;
    g.r2_min = FLT_MAX;
    g.host = NULL;
    g.host_cell = NULL;
    for (int c = 0; c < s->nr_cells; c++)
      wind_gather_recurse(&s->cells_top[c], s, &g);
    p->wind_data.rho_ambient = g.rho;

    if (p->wind_data.recouple_pending == wind_reason_none) continue;
    if (g.host != NULL) {
      wind_merge_into_host(e, p, xp, g.host, &s->xparts[g.host - s->parts],
                           g.host_cell);
    } else {
      /* No gas within reach (vacuum): rejoin as itself; with no neighbours
       * there is nothing for the limiter to wake */
      const int reason = p->wind_data.recouple_pending;
      wind_recouple(p, xp, props, e->cosmology, e->time, reason);
    }
  }
}

#endif /* WIND_TIGRESS */
