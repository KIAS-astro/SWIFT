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
#include "light_physics.h"
#include "part.h"
#include "random.h"
#include "timeline.h"
#include "wind_physics.h"
#include "wind_properties.h"

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
  p->wind_data.m_pending = 0.;
  p->wind_data.E_pending = 0.;
  p->wind_data.light_merges = 0;
  if (!p->wind_data.light) {
    p->wind_data.dye = 0.f;
    p->wind_data.dye_time = -1.f;
  } else {
    /* Light particles in the initial conditions carry WindDye, which holds
     * from the start time (or from recoupling if decoupled) */
    p->wind_data.dye_time = decoupled ? -1.f : time;
  }

  if (decoupled) {
    p->wind_data.launch_time = time;
    p->wind_data.launch_e_kin = wind_specific_kinetic(xp);
    p->wind_data.launch_e_th = hydro_get_comoving_internal_energy(p, xp);
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
  /* Light mode: the wind is drawn from the reservoir, not from this number */
  if (props->light) return u < p_star ? wind_draw_star : wind_draw_nothing;
  double eta_M, eta_E;
  wind_loadings(props, xp, &eta_M, &eta_E);
  return wind_draw(u, p_star, eta_M * p_star);
}

/**
 * @brief Light mode: add this step's wind to the particle's reservoir.
 *
 * @param p The #part.
 * @param xp The #xpart, with the SFR of this step.
 * @param props The wind properties.
 * @param dt_star The step over which the SFR acts.
 */
INLINE static void wind_light_accumulate(struct part *p,
                                         const struct xpart *xp,
                                         const struct wind_props *props,
                                         const double dt_star) {
  if (!props->light || props->model == wind_loading_none) return;
  if (xp->sf_data.SFR <= 0.f) return;
  double eta_M, eta_E;
  wind_loadings(props, xp, &eta_M, &eta_E);
  const double m_star = xp->sf_data.SFR * dt_star;
  const double dm = eta_M * m_star;
  const double dE = eta_E * props->u_star * m_star;
  p->wind_data.m_pending += dm;
  p->wind_data.E_pending += dE;
  atomic_add_d(&wind_global_stats.m_pending_added, dm);
  atomic_add_d(&wind_global_stats.E_pending_added, dE);
}

/**
 * @brief Light mode: a particle about to become a star takes its reservoir
 * with it; count that mass. A negative reservoir (wind spawned ahead of the
 * star formation that owes it) is cancelled the same way.
 */
INLINE static void wind_part_becomes_star(struct part *p) {
  if (p->wind_data.m_pending == 0.) return;
  atomic_add_d(&wind_global_stats.m_pending_to_stars, p->wind_data.m_pending);
  p->wind_data.m_pending = 0.;
  p->wind_data.E_pending = 0.;
}

/**
 * @brief Light mode: dye of a particle at a time (Bennett et al. 2025,
 * eq. 6), frozen while decoupled.
 */
INLINE static double wind_light_dye(const struct part *p,
                                    const struct wind_props *props,
                                    const double time) {
  if (p->wind_data.dye_time < 0.f) return p->wind_data.dye;
  return p->wind_data.dye *
         exp(-(time - p->wind_data.dye_time) / props->light_t0);
}

/**
 * @brief Light mode: target mass of a light particle at a time.
 */
INLINE static double wind_light_target_mass(const struct part *p,
                                            const struct wind_props *props,
                                            const double time) {
  return light_target_mass(wind_light_dye(p, props, time), props->light_m_l,
                           props->light_m_g, props->light_fw_thresh,
                           props->light_fw_ref);
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
  if (props->model == wind_loading_none || props->light) return 0;
  if (xp->sf_data.SFR <= 0.f) return 0;
  const double p_star = xp->sf_data.SFR * dt_star / hydro_get_mass(p);
  const double u =
      random_unit_interval(p->id, ti_current, random_number_star_formation);
  return wind_star_formation_draw(props, p, xp, p_star, u) == wind_draw_wind;
}

/**
 * @brief Kick a particle by dv along the unit vector dir, set its internal
 * energy to u_w and decouple it.
 */
INLINE static void wind_launch_kick(struct part *p, struct xpart *xp,
                                    const struct wind_props *props,
                                    const struct cosmology *cosmo,
                                    const double time, const double dir[3],
                                    const double dv, const double u_w) {

  const double m_kick = hydro_get_mass(p);
  for (int k = 0; k < 3; k++) {
    const float dv_k = dv * dir[k];
    xp->v_full[k] += dv_k;
    p->v[k] += dv_k;
    if (p->gpart != NULL) p->gpart->v_full[k] = xp->v_full[k];
    atomic_add_d(&wind_global_stats.P_launch[k], m_kick * dv_k);
  }

  /* A particle launched at a tree rebuild may have had its density reset by
   * the drift (active particles are re-initialised for the density loop).
   * The density of a decoupled particle is never recomputed; any positive
   * value keeps the pressure and sound speed finite */
  if (p->rho <= 0.f) p->rho = hydro_get_mass(p) / (p->h * p->h * p->h);
#if defined(GIZMO_MFM_SPH)
  /* Likewise the GIZMO volume, which sets the first CFL step at recoupling */
  if (p->geometry.volume <= 0.f)
    p->geometry.volume = hydro_get_mass(p) / p->rho;
#endif

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
  p->wind_data.launch_e_th = hydro_get_physical_internal_energy(p, xp, cosmo);
  p->wind_data.d_rec = wind_recoupling_distance(p, props);

  const double m = hydro_get_mass(p);
  atomic_add_d(&wind_global_stats.n_launch, 1.);
  atomic_add_d(&wind_global_stats.m_launch, m);
  atomic_add_d(&wind_global_stats.E_kin_launch, m * p->wind_data.launch_e_kin);
  atomic_add_d(&wind_global_stats.E_th_launch, m * p->wind_data.launch_e_th);
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
  wind_launch_kinematics(props->u_star, eta_M, eta_E, props->f_kin,
                         props->energy_convention, hydro_gamma, &dv, &u_w);

  const double sign =
      random_unit_interval(p->id, ti_current, random_number_wind_direction) <
              0.5
          ? -1.
          : 1.;
  const double dir[3] = {sign * props->normal[0], sign * props->normal[1],
                         sign * props->normal[2]};
  wind_launch_kick(p, xp, props, cosmo, time, dir, dv, u_w);
}

/**
 * @brief Light mode: launch a newly spawned light particle with the specific
 * launch energy e_w of its parent's reservoir, along dir (unit vector).
 *
 * The dye is set to its initial value (Bennett et al. 2025, eq. 12) with the
 * Bernoulli speed (Delta v^2 + 2 gamma u_w)^1/2 of this launch, and stays
 * frozen until recoupling.
 */
INLINE static void wind_light_launch(struct part *p, struct xpart *xp,
                                     const struct wind_props *props,
                                     const struct cosmology *cosmo,
                                     const double time, const double dir[3],
                                     const double e_w) {
  double dv, u_w;
  wind_launch_kinematics(props->u_star, 1., e_w / props->u_star, props->f_kin,
                         props->energy_convention, hydro_gamma, &dv, &u_w);
  wind_launch_kick(p, xp, props, cosmo, time, dir, dv, u_w);
  p->wind_data.light = 1;
  p->wind_data.light_merges = 0;
  p->wind_data.m_pending = 0.;
  p->wind_data.E_pending = 0.;
  p->wind_data.dye = light_dye_init(
      props->light_fw_thresh, props->light_R200, props->light_k,
      wind_bernoulli_speed(dv, u_w, hydro_gamma), props->light_t0);
  p->wind_data.dye_time = -1.f;
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
  p->wind_data.recouple_e_th =
      hydro_get_physical_internal_energy(p, xp, cosmo);
  hydro_reset_acceleration(p);

  /* The refinement clock of a light particle starts now */
  if (p->wind_data.light) p->wind_data.dye_time = time;

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

#endif /* SWIFT_WIND_TIGRESS_H */
