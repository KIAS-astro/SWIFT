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

  char mode[32] = "distance";
  parser_get_opt_param_string(params, "Wind:recoupling_mode", mode,
                              "distance");
  if (strcmp(mode, "distance") == 0)
    props->mode = wind_recouple_distance;
  else if (strcmp(mode, "density") == 0)
    props->mode = wind_recouple_density;
  else
    error("Wind:recoupling_mode must be distance or density, got '%s'", mode);

  char deposit[32] = "self";
  parser_get_opt_param_string(params, "Wind:recoupling_deposit", deposit,
                              "self");
  if (strcmp(deposit, "self") == 0)
    props->deposit = wind_deposit_self;
  else if (strcmp(deposit, "host") == 0)
    props->deposit = wind_deposit_host;
  else
    error("Wind:recoupling_deposit must be self or host, got '%s'", deposit);

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
  message(
      "Wind: recoupling mode '%s' (density threshold %.4g internal units), "
      "deposit '%s'",
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
#define WIND_DEBUG_NLOG 4096
static struct {
  long long host, wind;
  int step, bin;
  double m, v, u, hh, u0, rho0, f0, vsig0, alpha0;
  int mab;
} wind_debug_log[WIND_DEBUG_NLOG];
static int wind_debug_nlog = 0;

/* isfinite() is folded to 1 under -ffast-math, so test the exponent bits */
static int wind_debug_bad(const double x) {
  unsigned long long b;
  memcpy(&b, &x, sizeof(b));
  return ((b >> 52) & 0x7ffULL) == 0x7ffULL;
}

/**
 * @brief DEBUG: after the tasks of a step, every gas and gravity particle must
 * be finite; on failure print the offender and this step's merges.
 */
void wind_debug_check(struct engine *e) {
  const struct space *s = e->s;
  int nbad = 0;
  for (size_t k = 0; k < s->nr_parts; k++) {
    const struct part *p = &s->parts[k];
    const struct xpart *xp = &s->xparts[k];
    if (p->time_bin == time_bin_inhibited ||
        p->time_bin == time_bin_not_created)
      continue;
    if (wind_debug_bad(p->x[0] + p->x[1] + p->x[2]) ||
        wind_debug_bad((double)xp->v_full[0] + xp->v_full[1] + xp->v_full[2]) ||
        wind_debug_bad(xp->u_full) || wind_debug_bad(p->h) ||
        wind_debug_bad(p->rho) || wind_debug_bad(p->mass)) {
      message(
          "DEBUG non-finite part after tasks %lld step %d: x %g %g %g v %g %g %g "
          "u %g h %g rho %g m %g bin %d dec %d launches %d a_hydro %g %g %g",
          p->id, e->step, p->x[0], p->x[1], p->x[2], xp->v_full[0],
          xp->v_full[1], xp->v_full[2], xp->u_full, p->h, p->rho, p->mass,
          p->time_bin, p->wind_data.decoupled, p->wind_data.launch_count,
          p->a_hydro[0], p->a_hydro[1], p->a_hydro[2]);
      message(
          "DEBUG  force: p->u %g u_dt %g f %g P %g cs %g balsara %g alpha %g "
          "v_sig %g div_v %g div_v_prev %g alpha_diff %g laplace_u %g wcount %g "
          "wcount_dh %g rho_dh %g alpha_max_ngb %g max_active_bin %d wakeup %d sync %d",
          p->u, p->u_dt, p->force.f, p->force.pressure, p->force.soundspeed,
          p->force.balsara, p->viscosity.alpha, p->viscosity.v_sig,
          p->viscosity.div_v, p->viscosity.div_v_previous_step,
          p->diffusion.alpha, p->diffusion.laplace_u, p->density.wcount,
          p->density.wcount_dh, p->density.rho_dh, p->force.alpha_visc_max_ngb,
          e->max_active_bin, p->limiter_data.wakeup,
          p->limiter_data.to_be_synchronized);
      nbad++;
      for (int i = 0; i < min(wind_debug_nlog, WIND_DEBUG_NLOG); i++)
        if (wind_debug_log[i].host == p->id)
          message("DEBUG  host at step %d: wind %lld bin %d m %g v %g u %g; before: "
                  "u %g rho %g h %g f %g v_sig %g alpha %g max_active_bin %d",
                  wind_debug_log[i].step, wind_debug_log[i].wind,
                  wind_debug_log[i].bin, wind_debug_log[i].m,
                  wind_debug_log[i].v, wind_debug_log[i].u, wind_debug_log[i].u0,
                  wind_debug_log[i].rho0, wind_debug_log[i].hh,
                  wind_debug_log[i].f0, wind_debug_log[i].vsig0,
                  wind_debug_log[i].alpha0, wind_debug_log[i].mab);
    }
  }
  for (size_t k = 0; k < s->nr_gparts; k++) {
    const struct gpart *g = &s->gparts[k];
    if (g->time_bin == time_bin_inhibited || g->time_bin == time_bin_not_created)
      continue;
    if (!wind_debug_bad(g->x[0] + g->x[1] + g->x[2]) &&
        !wind_debug_bad((double)g->v_full[0] + g->v_full[1] + g->v_full[2]) &&
        !wind_debug_bad((double)g->a_grav[0] + g->a_grav[1] + g->a_grav[2]) &&
        !wind_debug_bad(g->mass))
      continue;
    long long pid = -1;
    if (g->type == swift_type_gas) pid = s->parts[-g->id_or_neg_offset].id;
    message(
        "DEBUG non-finite gpart step %d type %d part id %lld x %g %g %g v %g %g "
        "%g a %g %g %g m %g bin %d",
        e->step, g->type, pid, g->x[0], g->x[1], g->x[2], g->v_full[0],
        g->v_full[1], g->v_full[2], g->a_grav[0], g->a_grav[1], g->a_grav[2],
        g->mass, g->time_bin);
    if (g->type == swift_type_gas) {
      const struct part *p = &s->parts[-g->id_or_neg_offset];
      const struct xpart *xp = &s->xparts[-g->id_or_neg_offset];
      message("DEBUG  part u %g rho %g h %g a_hydro %g %g %g dec %d", xp->u_full,
              p->rho, p->h, p->a_hydro[0], p->a_hydro[1], p->a_hydro[2],
              p->wind_data.decoupled);
    }
    for (int i = 0; i < min(wind_debug_nlog, WIND_DEBUG_NLOG); i++)
      if (wind_debug_log[i].step >= e->step - 3) message("DEBUG  merge %d step %d host %lld wind %lld bin %d m %g v %g u %g h %g",
              i, wind_debug_log[i].step, wind_debug_log[i].host,
              wind_debug_log[i].wind, wind_debug_log[i].bin, wind_debug_log[i].m,
              wind_debug_log[i].v, wind_debug_log[i].u, wind_debug_log[i].hh);
    error("DEBUG non-finite particle after the tasks of step %d", e->step);
  }
  if (nbad) error("DEBUG %d non-finite gas particles after step %d", nbad, e->step);
}

static void wind_merge_into_host(struct engine *e, struct part *p,
                                 struct xpart *xp, struct part *h,
                                 struct xpart *hx, struct cell *hc) {

  const struct cosmology *cosmo = e->cosmology;
  const double mw = hydro_get_mass(p), mh = hydro_get_mass(h);
  const double m = mw + mh;
  const double uw = hydro_get_physical_internal_energy(p, xp, cosmo);
  const double uh = hydro_get_physical_internal_energy(h, hx, cosmo);
  {
    static long long n_merge = 0, n_lag = 0, n_active = 0;
    n_merge++;
    if (hc->hydro.ti_old_part != e->ti_current) n_lag++;
    if (h->time_bin <= e->max_active_bin) n_active++;
    if (n_merge % 200 == 0)
      message("DEBUG merges %lld: host cell not drifted to now %lld, host active %lld",
              n_merge, n_lag, n_active);
    if (!(uw > 0. && uh > 0.) || wind_debug_bad(uw) || wind_debug_bad(uh) ||
        h->h <= 0.f)
      message("DEBUG bad merge: wind %lld u %g host %lld u %g h %g bin %d", p->id,
              uw, h->id, uh, h->h, h->time_bin);
  }

  double v[3], ekw = 0., ekh = 0., ek = 0., dv2 = 0.;
  for (int k = 0; k < 3; k++) {
    v[k] = (mw * xp->v_full[k] + mh * hx->v_full[k]) / m;
    ekw += 0.5 * xp->v_full[k] * xp->v_full[k];
    ekh += 0.5 * hx->v_full[k] * hx->v_full[k];
    ek += 0.5 * v[k] * v[k];
    dv2 += (v[k] - hx->v_full[k]) * (v[k] - hx->v_full[k]);
  }
  const double u = (mw * (uw + ekw) + mh * (uh + ekh)) / m - ek;
  {
    const int i = wind_debug_nlog % WIND_DEBUG_NLOG;
    wind_debug_log[i].host = h->id;
    wind_debug_log[i].wind = p->id;
    wind_debug_log[i].step = e->step;
    wind_debug_log[i].bin = h->time_bin;
    wind_debug_log[i].m = m;
    wind_debug_log[i].v = sqrt(2. * ek);
    wind_debug_log[i].u = u;
    wind_debug_log[i].hh = h->h;
    wind_debug_log[i].u0 = uh;
    wind_debug_log[i].rho0 = h->rho;
    wind_debug_log[i].f0 = h->force.f;
    wind_debug_log[i].vsig0 = h->viscosity.v_sig;
    wind_debug_log[i].alpha0 = h->viscosity.alpha;
    wind_debug_log[i].mab = e->max_active_bin;
    wind_debug_nlog++;
  }

  /* Record the flight on the particle before it is removed */
  p->wind_data.recouple_time = e->time;
  p->wind_data.recouple_distance = wind_distance(p);
  p->wind_data.recouple_e_kin = ekw;
  p->wind_data.recouple_e_th = uw;
  const int reason = p->wind_data.recouple_pending;
  wind_global_stats.n_reason[reason] += 1.;
  wind_global_stats.n_recouple += 1.;
  wind_global_stats.m_recouple += mw;
  wind_global_stats.E_kin_recouple += mw * ekw;
  wind_global_stats.E_th_recouple += mw * uw;
  wind_global_stats.n_merged += 1.;

  /* Metal mass fractions: mass-weighted mean */
#if defined(CHEMISTRY_EAGLE)
  for (int k = 0; k < chemistry_element_count; k++) {
    h->chemistry_data.metal_mass_fraction[k] =
        (mw * p->chemistry_data.metal_mass_fraction[k] +
         mh * h->chemistry_data.metal_mass_fraction[k]) /
        m;
    h->chemistry_data.smoothed_metal_mass_fraction[k] =
        (mw * p->chemistry_data.smoothed_metal_mass_fraction[k] +
         mh * h->chemistry_data.smoothed_metal_mass_fraction[k]) /
        m;
  }
  h->chemistry_data.metal_mass_fraction_total =
      (mw * p->chemistry_data.metal_mass_fraction_total +
       mh * h->chemistry_data.metal_mass_fraction_total) /
      m;
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

  if (h->time_bin <= e->max_active_bin) {
    /* An active host may be between hydro_init_part (at a rebuild or with
     * --drift-all, everything is drifted before this pass) and its density
     * loop: in
     * SPHENIX the force fields (pressure, sound speed) share a union with the
     * density sums, so the feedback setters below would write into those sums
     * (with rho = 0 they write NaN). Set only the drifted energy; the ghost
     * recomputes pressure, sound speed and v_sig this step. */
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
    const struct part *p = &s->parts[k];
    const struct xpart *xp = &s->xparts[k];
    if (p->time_bin == time_bin_inhibited ||
        p->time_bin == time_bin_not_created)
      continue;
    if (wind_debug_bad(p->x[0] + p->x[1] + p->x[2]) ||
        wind_debug_bad((double)xp->v_full[0] + xp->v_full[1] + xp->v_full[2]) ||
        wind_debug_bad(xp->u_full) || wind_debug_bad(p->h) || p->h <= 0.f ||
        wind_debug_bad(p->rho) || wind_debug_bad(p->mass))
      error(
          "DEBUG non-finite part %lld step %d: x %g %g %g v %g %g %g u %g h %g "
          "rho %g m %g bin %d dec %d launches %d",
          p->id, e->step, p->x[0], p->x[1], p->x[2], xp->v_full[0],
          xp->v_full[1], xp->v_full[2], xp->u_full, p->h, p->rho, p->mass,
          p->time_bin, p->wind_data.decoupled, p->wind_data.launch_count);
  }
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
