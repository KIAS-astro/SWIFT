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
#ifndef SWIFT_WIND_TIGRESS_H
#define SWIFT_WIND_TIGRESS_H

/**
 * @file src/wind/TIGRESS/wind.h
 * @brief Decoupled galactic wind launched from PRFM star formation with the
 * TIGRESS hot-phase loadings (design: wind-design.md, WD1-WD7).
 *
 * A star-forming gas particle forms a star, launches, or stays, from one
 * uniform number (src/wind_physics.h). A launched particle gets a kick
 * Delta v along +-n and internal energy u_w, and is then skipped by every
 * hydrodynamical interaction, by cooling, star formation and the entropy
 * floor; it feels gravity only, with a time-step cap. It recouples after
 * travelling d_rec or after t_rec and rejoins the hydrodynamics with its
 * velocity and internal energy.
 *
 * Non-cosmological only: the kick and the energies are physical and
 * time is e->time.
 */

#include "atomic.h"
#include "cosmology.h"
#include "hydro.h"
#include "kernel_hydro.h"
#include "part.h"
#include "random.h"
#include "timeline.h"
#include "timestep_sync_part.h"
#include "wind_physics.h"
#include "wind_properties.h"

#include <stdio.h>

/**
 * @brief Is this gas particle decoupled from the hydrodynamics?
 */
__attribute__((always_inline)) INLINE static int part_is_decoupled(
    const struct part *p) {
  return p->wind_data.decoupled;
}

/**
 * @brief Specific kinetic energy of a particle (physical, non-cosmological).
 */
__attribute__((always_inline)) INLINE static float wind_specific_kinetic(
    const struct xpart *xp) {
  return 0.5f * (xp->v_full[0] * xp->v_full[0] +
                 xp->v_full[1] * xp->v_full[1] +
                 xp->v_full[2] * xp->v_full[2]);
}

/**
 * @brief Mass and energy loadings for a particle.
 *
 * @param props The wind properties.
 * @param xp The #xpart, carrying Sigma_SFR [Msun kpc^-2 yr^-1] from PRFM.
 * @param eta_M (return) Mass loading.
 * @param eta_E (return) Energy loading.
 */
INLINE static void wind_loadings(const struct wind_props *props,
                                 const struct xpart *xp, double *eta_M,
                                 double *eta_E) {
  switch (props->model) {
    case wind_loading_constant:
      *eta_M = props->eta_M;
      *eta_E = props->eta_E;
      break;
    case wind_loading_TIGRESS:
      *eta_M = wind_powerlaw_loading(props->eta_M_log10_norm,
                                     props->eta_M_slope, xp->sf_data.Sigma_SFR);
      *eta_E = wind_powerlaw_loading(props->eta_E_log10_norm,
                                     props->eta_E_slope, xp->sf_data.Sigma_SFR);
      break;
    default:
      *eta_M = 0.;
      *eta_E = 0.;
  }
}

/**
 * @brief Recoupling distance of a particle launched now: at least
 * d_rec_kernel_factor kernel supports, so that it leaves its launch kernel.
 */
INLINE static float wind_recoupling_distance(const struct part *p,
                                             const struct wind_props *props) {
  return fmax(props->d_rec, props->d_rec_kernel_factor * kernel_gamma * p->h);
}

/**
 * @brief Set the wind fields of a particle at the start of a run.
 *
 * A particle flagged as decoupled in the initial conditions (field
 * WindDecoupled) is launched at the start time with its initial velocity and
 * internal energy: this is how the tests decouple particles by hand.
 */
INLINE static void wind_first_init_part(struct part *p, struct xpart *xp,
                                        const struct wind_props *props,
                                        const double time) {

  const char decoupled = p->wind_data.decoupled;
  for (int k = 0; k < 3; k++) {
    p->wind_data.dx[k] = 0.;
    p->wind_data.launch_x[k] = p->x[k];
  }
  p->wind_data.launch_time = -1.f;
  p->wind_data.launch_e_kin = -1.f;
  p->wind_data.launch_e_th = -1.f;
  p->wind_data.recouple_time = -1.f;
  p->wind_data.recouple_e_kin = -1.f;
  p->wind_data.recouple_e_th = -1.f;
  p->wind_data.recouple_distance = -1.f;
  p->wind_data.d_rec = wind_recoupling_distance(p, props);
  p->wind_data.launch_count = 0;
  p->wind_data.decoupled = decoupled ? 1 : 0;

  if (decoupled) {
    p->wind_data.launch_time = time;
    p->wind_data.launch_e_kin = wind_specific_kinetic(xp);
    p->wind_data.launch_e_th = xp->u_full;
    p->wind_data.launch_count = 1;
    hydro_reset_acceleration(p);
    /* Never computed for a decoupled particle; any positive value keeps the
     * pressure and sound speed set in the kick finite */
    if (p->rho <= 0.f) p->rho = hydro_get_mass(p) / (p->h * p->h * p->h);
  }
}

/**
 * @brief Star, wind or nothing for a star-forming particle.
 *
 * @param props The wind properties.
 * @param p The #part.
 * @param xp The #xpart.
 * @param p_star Probability of forming a star in this step.
 * @param u The uniform number of the star formation draw.
 * @return 1 for a star, 2 for a launch, 0 otherwise.
 */
__attribute__((always_inline)) INLINE static int wind_star_formation_draw(
    const struct wind_props *props, const struct part *p,
    const struct xpart *xp, const double p_star, const double u) {
  /* Carriers are drawn independently of the star */
  if (props->carriers) return wind_draw(u, p_star, 0.);
  double eta_M, eta_E;
  wind_loadings(props, xp, &eta_M, &eta_E);
  return wind_draw(u, p_star, eta_M * p_star);
}

/**
 * @brief Should a star-forming particle that did not form a star launch?
 *
 * Repeats the draw of the star formation module with the same uniform
 * number, so that star and wind exclude each other.
 */
INLINE static int wind_should_launch(const struct part *p,
                                     const struct xpart *xp,
                                     const struct wind_props *props,
                                     const integertime_t ti_current,
                                     const double dt_star) {
  if (props->model == wind_loading_none || props->carriers) return 0;
  if (xp->sf_data.SFR <= 0.f) return 0;
  const double p_star = xp->sf_data.SFR * dt_star / hydro_get_mass(p);
  const double u =
      random_unit_interval(p->id, ti_current, random_number_star_formation);
  return wind_star_formation_draw(props, p, xp, p_star, u) == wind_draw_wind;
}

/**
 * @brief Launch a particle (WD3): kick along +-n, set u = u_w, decouple.
 */
INLINE static void wind_launch(struct part *p, struct xpart *xp,
                               const struct wind_props *props,
                               const struct cosmology *cosmo,
                               const double time,
                               const integertime_t ti_current) {

  double eta_M, eta_E, dv, u_w;
  wind_loadings(props, xp, &eta_M, &eta_E);
  wind_launch_kinematics(props->u_star, eta_M, eta_E, props->f_kin, &dv,
                         &u_w);

  const double sign =
      random_unit_interval(p->id, ti_current, random_number_wind_direction) <
              0.5
          ? -1.
          : 1.;
  for (int k = 0; k < 3; k++) {
    const float dv_k = sign * dv * props->normal[k];
    xp->v_full[k] += dv_k;
    p->v[k] += dv_k;
    if (p->gpart != NULL) p->gpart->v_full[k] = xp->v_full[k];
  }

  hydro_set_physical_internal_energy(p, xp, cosmo, u_w);
  hydro_set_drifted_physical_internal_energy(p, cosmo, NULL, u_w);

  /* No hydrodynamical acceleration or heating while decoupled */
  hydro_reset_acceleration(p);

  p->wind_data.decoupled = 1;
  p->wind_data.launch_count++;
  p->wind_data.launch_time = time;
  for (int k = 0; k < 3; k++) {
    p->wind_data.dx[k] = 0.;
    p->wind_data.launch_x[k] = p->x[k];
  }
  p->wind_data.launch_e_kin = wind_specific_kinetic(xp);
  p->wind_data.launch_e_th = xp->u_full;
  p->wind_data.d_rec = wind_recoupling_distance(p, props);

  const double m = hydro_get_mass(p);
  atomic_add_d(&wind_global_stats.n_launch, 1.);
  atomic_add_d(&wind_global_stats.m_launch, m);
  atomic_add_d(&wind_global_stats.E_kin_launch, m * p->wind_data.launch_e_kin);
  atomic_add_d(&wind_global_stats.E_th_launch, m * p->wind_data.launch_e_th);
}

/**
 * @brief Distance travelled since the launch.
 */
__attribute__((always_inline)) INLINE static double wind_distance(
    const struct part *p) {
  const double *dx = p->wind_data.dx;
  return sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
}

/**
 * @brief Should a decoupled particle recouple (WD5)?
 */
INLINE static int wind_should_recouple(const struct part *p,
                                       const struct wind_props *props,
                                       const double time) {
  if (!p->wind_data.decoupled) return 0;
  return wind_distance(p) >= p->wind_data.d_rec ||
         time - p->wind_data.launch_time >= props->t_rec;
}

/**
 * @brief Recouple a particle (WD6): it keeps its velocity and internal
 * energy; record them.
 */
INLINE static void wind_recouple(struct part *p, struct xpart *xp,
                                 const struct wind_props *props,
                                 const struct cosmology *cosmo,
                                 const double time) {

  p->wind_data.decoupled = 0;
  p->wind_data.recouple_time = time;
  p->wind_data.recouple_distance = wind_distance(p);
  p->wind_data.recouple_e_kin = wind_specific_kinetic(xp);
  p->wind_data.recouple_e_th = xp->u_full;
  hydro_reset_acceleration(p);

  /* The signal velocity of the first CFL time-step must know that the
   * particle arrives at its speed in gas that is, to first order, at rest */
  hydro_set_v_sig_based_on_velocity_kick(
      p, cosmo, sqrtf(2.f * p->wind_data.recouple_e_kin));

  const double m = hydro_get_mass(p);
  atomic_add_d(&wind_global_stats.n_recouple, 1.);
  atomic_add_d(&wind_global_stats.m_recouple, m);
  atomic_add_d(&wind_global_stats.E_kin_recouple,
               m * p->wind_data.recouple_e_kin);
  atomic_add_d(&wind_global_stats.E_th_recouple,
               m * p->wind_data.recouple_e_th);
}

/**
 * @brief Accumulate the displacement of a decoupled particle.
 */
__attribute__((always_inline)) INLINE static void wind_drift_part(
    struct part *p, const struct xpart *xp, const double dt_drift) {
  if (!p->wind_data.decoupled) return;
  for (int k = 0; k < 3; k++) p->wind_data.dx[k] += xp->v_full[k] * dt_drift;
}

/**
 * @brief Time-step cap of a decoupled particle.
 */
INLINE static float wind_timestep(const struct wind_props *props) {
  return props->dt_max;
}

/*
 * Wind carriers (wind-carrier-design.md): star particles of mass m_g / N_c
 * that carry the wind's mass, momentum, energy and metals from star-forming
 * gas to where they recouple, and deposit them there into the gas inside
 * their kernel. They use the stars loops of the feedback/none module
 * (density for h, prep2 for the kernel-weight sum, feedback for the
 * deposit) and are removed in the time-step task.
 */

/**
 * @brief Is this star particle a wind carrier?
 */
__attribute__((always_inline)) INLINE static int spart_is_wind_carrier(
    const struct spart *sp) {
  return sp->wind_data.carrier;
}

/**
 * @brief Is this star particle active in the stars density and feedback
 * loops? In a run with carriers: a carrier on the step it deposits, never an
 * ordinary star (with PRFM the wind is the only feedback). Otherwise
 * every star.
 */
__attribute__((always_inline)) INLINE static int wind_feedback_is_active(
    const struct spart *sp) {
  if (!wind_carriers_on) return 1;
  return sp->wind_data.carrier && sp->wind_data.deposit;
}

/**
 * @brief Set the carrier fields of a star particle at the start of a run.
 *
 * A star particle flagged WindCarrier in the initial conditions is a carrier
 * launched at the start time with its initial velocity, mass and the
 * internal energy WindCarrierInternalEnergies: this is how the tests place
 * carriers by hand.
 */
INLINE static void wind_first_init_spart(struct spart *sp,
                                         const struct wind_props *props,
                                         const double time) {
  const char carrier = sp->wind_data.carrier;
  const float u = sp->wind_data.u;
  bzero(&sp->wind_data, sizeof(struct wind_spart_data));
  if (!carrier) return;
  sp->wind_data.carrier = 1;
  sp->wind_data.u = u;
  sp->wind_data.launch_time = time;
  for (int k = 0; k < 3; k++) sp->wind_data.launch_x[k] = sp->x[k];
  sp->wind_data.d_rec =
      fmax(props->d_rec, props->d_rec_kernel_factor * kernel_gamma * sp->h);
  sp->birth_time = time;

  double v2 = 0.;
  for (int k = 0; k < 3; k++) {
    v2 += sp->v[k] * sp->v[k];
    atomic_add_d(&wind_global_stats.c_p_launch[k], sp->mass * sp->v[k]);
  }
  atomic_add_d(&wind_global_stats.c_n_launch, 1.);
  atomic_add_d(&wind_global_stats.c_m_launch, sp->mass);
  atomic_add_d(&wind_global_stats.c_E_kin_launch, 0.5 * sp->mass * v2);
  atomic_add_d(&wind_global_stats.c_E_th_launch, sp->mass * u);
}

/**
 * @brief Number of carriers a star-forming particle launches in this step.
 *
 * p_c = eta_M mdot_* dt / m_c with m_c = m_g / N_c; floor(p_c) carriers and
 * one more with probability p_c - floor(p_c), from a random stream of its
 * own (independent of the star draw). If what would be left of the particle
 * is less than a tenth of a carrier, the last carrier takes the whole
 * particle (*whole = 1), so that N_c = 1 launches whole particles.
 *
 * @param m_c (return) The carrier mass.
 * @param whole (return) 1 if the last carrier converts the gas particle.
 */
INLINE static int wind_number_of_carriers(const struct part *p,
                                          const struct xpart *xp,
                                          const struct wind_props *props,
                                          const integertime_t ti_current,
                                          const double dt_star, double *m_c,
                                          int *whole) {
  *whole = 0;
  *m_c = 0.;
  if (!props->carriers || props->model == wind_loading_none) return 0;
  if (xp->sf_data.SFR <= 0.f) return 0;

  double eta_M, eta_E;
  wind_loadings(props, xp, &eta_M, &eta_E);
  const double m_g = hydro_get_mass(p);
  *m_c = m_g / props->carriers_per_particle;
  const double p_c = eta_M * xp->sf_data.SFR * dt_star / *m_c;
  const double u =
      random_unit_interval(p->id, ti_current, random_number_wind_carrier);
  int n = wind_carrier_count(p_c, u);
  if (n == 0) return 0;

  const int n_max = (int)floor(m_g / *m_c + 1e-6);
  if (n > n_max) n = n_max;
  if (m_g - n * *m_c < 0.1 * *m_c) *whole = 1;
  return n;
}

/**
 * @brief Give a new star particle the properties of a carrier launched from
 * a gas particle, and take its mass from the gas.
 *
 * The carrier starts at the gas position with v_gas + Delta v n (random sign
 * along the disc normal), internal energy u_w (WD3) and the gas metal
 * fractions. The gas keeps its velocity and specific internal energy and
 * loses m_c; if whole, the gas particle has been converted into the carrier
 * and the carrier takes all of its mass.
 *
 * @param sp The new #spart (from cell_spawn_new_spart_from_part or
 * cell_convert_part_to_spart).
 * @param p The gas #part.
 * @param xp The gas #xpart.
 * @param m_c The carrier mass (ignored if whole).
 * @param whole 1 if the gas particle was converted.
 * @param index Number of this carrier among those launched by p this step.
 */
INLINE static void wind_carrier_launch(
    struct spart *sp, struct part *p, struct xpart *xp,
    const struct wind_props *props, const struct cosmology *cosmo,
    double m_c, const int whole, const double time,
    const integertime_t ti_current, const int index) {

  double eta_M, eta_E, dv, u_w;
  wind_loadings(props, xp, &eta_M, &eta_E);
  wind_launch_kinematics(props->u_star, eta_M, eta_E, props->f_kin, &dv,
                         &u_w);
  const double sign =
      random_unit_interval_part_ID_and_index(
          p->id, index, ti_current, random_number_wind_direction) < 0.5
          ? -1.
          : 1.;

  const double m_g = hydro_get_mass(p);
  if (whole) m_c = m_g;
  const double u_g = hydro_get_physical_internal_energy(p, xp, cosmo);

  double v2_g = 0., v2_c = 0.;
  for (int k = 0; k < 3; k++) {
    sp->v[k] = xp->v_full[k] + sign * dv * props->normal[k];
    sp->gpart->v_full[k] = sp->v[k];
    v2_g += xp->v_full[k] * xp->v_full[k];
    v2_c += sp->v[k] * sp->v[k];
  }
  sp->mass = m_c;
  sp->mass_init = m_c;
  sp->gpart->mass = m_c;
  sp->h = p->h;
  sp->birth_time = time;
  sp->last_enrichment_time = time;
  sp->count_since_last_enrichment = -1;
  sp->chemistry_data = p->chemistry_data;

  bzero(&sp->wind_data, sizeof(struct wind_spart_data));
  sp->wind_data.carrier = 1;
  sp->wind_data.u = u_w;
  sp->wind_data.launch_time = time;
  for (int k = 0; k < 3; k++) {
    sp->wind_data.launch_x[k] = p->x[k];
    sp->wind_data.a_grav[k] = sp->gpart->a_grav[k];
  }
  sp->wind_data.d_rec = wind_recoupling_distance(p, props);

  if (!whole) {
    hydro_set_mass(p, m_g - m_c);
    p->gpart->mass = m_g - m_c;
    /* mdot_* is proportional to the mass */
    xp->sf_data.SFR *= (m_g - m_c) / m_g;
  }

  atomic_add_d(&wind_global_stats.c_n_launch, 1.);
  atomic_add_d(&wind_global_stats.c_m_launch, m_c);
  for (int k = 0; k < 3; k++)
    atomic_add_d(&wind_global_stats.c_p_launch[k], m_c * sp->v[k]);
  atomic_add_d(&wind_global_stats.c_E_kin_launch, 0.5 * m_c * v2_c);
  atomic_add_d(&wind_global_stats.c_E_th_launch, m_c * u_w);
  atomic_add_d(&wind_global_stats.c_E_launch_net,
               m_c * (0.5 * v2_c + u_w - 0.5 * v2_g - u_g));
  atomic_add_d(&wind_global_stats.c_Z_launch,
               m_c * p->chemistry_data.metal_mass_fraction_total);
}

/**
 * @brief Count a carrier that could not be created (no free slot).
 */
INLINE static void wind_carrier_launch_failed(void) {
  atomic_add_d(&wind_global_stats.c_n_failed, 1.);
}

/**
 * @brief Accumulate the displacement of a carrier and decide whether it
 * deposits: after travelling d_rec or after t_rec.
 */
__attribute__((always_inline)) INLINE static void wind_carrier_drift(
    struct spart *sp, const double dt_drift, const struct wind_props *props,
    const double time) {
  if (!sp->wind_data.carrier) return;
  double *dx = sp->wind_data.dx;
  for (int k = 0; k < 3; k++) dx[k] += sp->v[k] * dt_drift;
  if (sp->wind_data.deposit) return;
  if (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2] >=
      (double)sp->wind_data.d_rec * sp->wind_data.d_rec)
    sp->wind_data.deposit = wind_carrier_distance;
  else if (time - sp->wind_data.launch_time >= props->t_rec)
    sp->wind_data.deposit = wind_carrier_time;
}

/**
 * @brief Reset the kernel-weight sum at the start of a step (and of each
 * smoothing-length iteration; the sum is taken after the last one).
 */
__attribute__((always_inline)) INLINE static void wind_carrier_init_spart(
    struct spart *sp) {
  sp->wind_data.weight_sum = 0.f;
  sp->wind_data.ngb_count = 0;
}

/**
 * @brief Stars prep2 loop: sum the kernel weights of the gas neighbours of
 * a depositing carrier, at its converged h and over the neighbours the
 * feedback loop will visit.
 */
__attribute__((always_inline)) INLINE static void wind_carrier_iact_density(
    const float r2, const float hi, struct spart *si) {
  if (!si->wind_data.carrier || !si->wind_data.deposit) return;
  float wi;
  kernel_eval(sqrtf(r2) / hi, &wi);
  si->wind_data.weight_sum += wi;
  si->wind_data.ngb_count++;
}

/**
 * @brief Mix a carrier's metal mass fractions into a gas particle: each
 * fraction becomes (m f + dm f_c) / (m + dm), which conserves metal mass.
 */
INLINE static void wind_carrier_mix_metals(
    struct chemistry_part_data *restrict cj,
    const struct chemistry_part_data *restrict cc, const double m,
    const double dm) {
#define WIND_MIX(f) cj->f = wind_carrier_mix(m, cj->f, dm, cc->f)
  for (int i = 0; i < chemistry_element_count; i++) {
    WIND_MIX(metal_mass_fraction[i]);
    WIND_MIX(smoothed_metal_mass_fraction[i]);
  }
  WIND_MIX(metal_mass_fraction_total);
  WIND_MIX(smoothed_metal_mass_fraction_total);
  WIND_MIX(metal_mass_fraction_from_SNIa);
  WIND_MIX(metal_mass_fraction_from_AGB);
  WIND_MIX(metal_mass_fraction_from_SNII);
  WIND_MIX(iron_mass_fraction_from_SNIa);
  WIND_MIX(smoothed_iron_mass_fraction_from_SNIa);
#undef WIND_MIX
}

/**
 * @brief Stars feedback loop: give a gas neighbour j its share of a
 * depositing carrier.
 *
 * With w_j = W(r_j, h_c) / sum W and dm = w_j m_c: momentum
 * m v_j + dm v_c, metals mixed by mass, and thermal energy
 *   m u_j + dm (v_c^2 / 2 + u_c) + m v_j^2 / 2 - (m + dm) v_j'^2 / 2
 *   - dm a_c . (x_c - x_j),
 * the last term being the potential energy gained by the moved mass (exact
 * for a uniform field, first order otherwise). Total energy is therefore
 * conserved pair by pair, unless the internal-energy floor applies.
 *
 * @param dx Vector from the gas particle to the carrier (x_c - x_j).
 */
INLINE static void wind_carrier_iact_apply(
    const float r2, const float dx[3], const float hi, const struct spart *si,
    struct part *pj, struct xpart *xpj, const struct cosmology *cosmo,
    const struct hydro_props *hydro_props) {

  if (!si->wind_data.carrier || !si->wind_data.deposit) return;
  if (si->wind_data.weight_sum <= 0.f) return;

  float wi;
  kernel_eval(sqrtf(r2) / hi, &wi);
  const double dm =
      si->mass * wind_carrier_share(wi, si->wind_data.weight_sum);
  const double m = hydro_get_mass(pj);
  const double m_new = m + dm;
  const double u_j = hydro_get_physical_internal_energy(pj, xpj, cosmo);

  double v_j[3], v_c[3], v_new[3];
  double v2_c = 0., a_dx = 0.;
  for (int k = 0; k < 3; k++) {
    v_j[k] = xpj->v_full[k];
    v_c[k] = si->v[k];
    v2_c += v_c[k] * v_c[k];
    a_dx += (double)si->wind_data.a_grav[k] * dx[k];
  }
  const double E_pot = dm * a_dx;
  double E_th = wind_carrier_deposit(m, v_j, u_j, dm, v_c, si->wind_data.u,
                                     a_dx, v_new);
  const double E_min = hydro_props->minimal_internal_energy * m_new;
  double E_floor = 0.;
  if (E_th < E_min) {
    E_floor = E_min - E_th;
    E_th = E_min;
  }

  for (int k = 0; k < 3; k++) {
    pj->v[k] += v_new[k] - xpj->v_full[k];
    xpj->v_full[k] = v_new[k];
    if (pj->gpart != NULL) pj->gpart->v_full[k] = v_new[k];
  }
  wind_carrier_mix_metals(&pj->chemistry_data, &si->chemistry_data, m, dm);
  hydro_set_mass(pj, m_new);
  if (pj->gpart != NULL) pj->gpart->mass = m_new;
  const double u_new = E_th / m_new;
  hydro_set_physical_internal_energy(pj, xpj, cosmo, u_new);
  hydro_set_drifted_physical_internal_energy(pj, cosmo, NULL, u_new);
  timestep_sync_part(pj);

  atomic_add_d(&wind_global_stats.c_m_deposit, dm);
  for (int k = 0; k < 3; k++)
    atomic_add_d(&wind_global_stats.c_p_deposit[k], dm * si->v[k]);
  atomic_add_d(&wind_global_stats.c_E_kin_deposit, 0.5 * dm * v2_c);
  atomic_add_d(&wind_global_stats.c_E_th_deposit, dm * si->wind_data.u);
  atomic_add_d(&wind_global_stats.c_E_pot_deposit, E_pot);
  if (E_floor > 0.) atomic_add_d(&wind_global_stats.c_E_floor, E_floor);
  atomic_add_d(&wind_global_stats.c_Z_deposit,
               dm * si->chemistry_data.metal_mass_fraction_total);
}

/**
 * @brief Called in the time-step task for an active star particle, after
 * its feedback loops. Returns 1 if the particle is a carrier that has
 * deposited and must now be removed (and records the deposit); otherwise
 * stores the gravitational acceleration of a carrier for its next deposit
 * and returns 0. A carrier that found no gas inside h_max keeps flying and
 * tries again on its next step.
 */
INLINE static int wind_carrier_end_step(struct spart *sp, const double time) {
  if (!sp->wind_data.carrier) return 0;
  if (sp->wind_data.deposit && sp->wind_data.weight_sum > 0.f) {
    atomic_add_d(&wind_global_stats.c_n_deposit, 1.);
    if (sp->wind_data.deposit == wind_carrier_time)
      atomic_add_d(&wind_global_stats.c_n_by_time, 1.);
    const double *dx = sp->wind_data.dx;
    char line[512];
    snprintf(line, sizeof(line),
             "%lld %.9e %.9e %d %.9e %.9e %.9e %.9e %.9e %.9e %.9e %.9e "
             "%.9e %.9e %.9e %.9e %.9e %.9e %d %.9e\n",
             sp->id, sp->wind_data.launch_time, time, sp->wind_data.deposit,
             sp->wind_data.launch_x[0], sp->wind_data.launch_x[1],
             sp->wind_data.launch_x[2], sp->x[0], sp->x[1], sp->x[2],
             sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]),
             sp->wind_data.d_rec, sp->mass, sp->v[0], sp->v[1], sp->v[2],
             sp->wind_data.u, sp->h, sp->wind_data.ngb_count,
             sp->wind_data.weight_sum);
    wind_carrier_log_event(line);
    return 1;
  }
  if (sp->wind_data.deposit)
    atomic_add_d(&wind_global_stats.c_n_retry, 1.);
  for (int k = 0; k < 3; k++)
    sp->wind_data.a_grav[k] = sp->gpart->a_grav[k];
  return 0;
}

#endif /* SWIFT_WIND_TIGRESS_H */
