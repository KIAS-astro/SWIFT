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
#include "gravity.h"
#include "hydro.h"
#include "part.h"
#include "potential.h"
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
 * @brief Gravitational potential per unit mass at a gas particle (physical):
 * the self-gravity potential of the last force calculation plus the external
 * potential.
 *
 * With self-gravity the particle's potential already holds the external
 * term, which external_gravity_acceleration() adds to it; without, the
 * external potential is evaluated at the particle.
 *
 * Valid for an active particle after the gravity tasks of the step, i.e. at
 * launch (star formation task) and recoupling (time-step task).
 *
 * @param p The #part.
 * @param potential The external potential.
 * @param phys_const The physical constants in internal units.
 * @param cosmo The cosmological model.
 * @param with_self_gravity Is self-gravity on?
 * @param with_external_gravity Is the external potential on?
 * @param time The current time.
 */
INLINE static float wind_specific_potential(
    const struct part *p, const struct external_potential *potential,
    const struct phys_const *phys_const, const struct cosmology *cosmo,
    const int with_self_gravity, const int with_external_gravity,
    const double time) {
  const struct gpart *gp = p->gpart;
  if (gp == NULL) return 0.f;
  if (with_self_gravity) return gravity_get_physical_potential(gp, cosmo);
  if (with_external_gravity)
    return external_gravity_get_potential_energy(time, potential, phys_const,
                                                 gp);
  return 0.f;
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
 * internal energy: this is how the tests decouple particles by hand. Such
 * a launch is counted in the global totals. Its launch potential is set at
 * its first time-step (wind_set_pending_launch_potential()), since no
 * gravity has been computed yet.
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
  p->wind_data.launch_phi = 0.f;
  p->wind_data.launch_phi_pending = decoupled ? 1 : 0;
  p->wind_data.recouple_time = -1.f;
  p->wind_data.recouple_e_kin = -1.f;
  p->wind_data.recouple_e_th = -1.f;
  p->wind_data.recouple_phi = 0.f;
  p->wind_data.recouple_distance = -1.f;
  p->wind_data.d_rec = wind_recoupling_distance(p, props);
  p->wind_data.launch_count = 0;
  p->wind_data.decoupled = decoupled ? 1 : 0;
  p->wind_data.tracer_mass = 0.f;
  p->wind_data.tracer_energy = 0.f;
  p->wind_data.tracer_potential_energy = 0.f;
  p->wind_data.tracer_launch_energy = 0.f;

  if (decoupled) {
    p->wind_data.launch_time = time;
    p->wind_data.launch_e_kin = wind_specific_kinetic(xp);
    p->wind_data.launch_e_th = xp->u_full;
    p->wind_data.launch_count = 1;
    const double m = hydro_get_mass(p);
    atomic_add_d(&wind_global_stats.n_launch, 1.);
    atomic_add_d(&wind_global_stats.m_launch, m);
    atomic_add_d(&wind_global_stats.E_kin_launch,
                 m * p->wind_data.launch_e_kin);
    atomic_add_d(&wind_global_stats.E_th_launch, m * p->wind_data.launch_e_th);
    hydro_reset_acceleration(p);
    /* Never computed for a decoupled particle; any positive value keeps the
     * pressure and sound speed set in the kick finite */
    if (p->rho <= 0.f) p->rho = hydro_get_mass(p) / (p->h * p->h * p->h);
  }
}

/**
 * @brief Set the launch potential of a particle decoupled in the initial
 * conditions, at its first time-step (the first time its potential exists).
 *
 * @param phi The gravitational potential per unit mass at the particle
 * (wind_specific_potential()).
 */
INLINE static void wind_set_pending_launch_potential(struct part *p,
                                                     const float phi) {
  if (!p->wind_data.launch_phi_pending) return;
  p->wind_data.launch_phi = phi;
  p->wind_data.launch_phi_pending = 0;
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
  if (props->model == wind_loading_none) return 0;
  if (xp->sf_data.SFR <= 0.f) return 0;
  const double p_star = xp->sf_data.SFR * dt_star / hydro_get_mass(p);
  const double u =
      random_unit_interval(p->id, ti_current, random_number_star_formation);
  return wind_star_formation_draw(props, p, xp, p_star, u) == wind_draw_wind;
}

/**
 * @brief Launch a particle (WD3): kick along +-n, set u = u_w, decouple.
 *
 * @param phi The gravitational potential per unit mass at the particle
 * (wind_specific_potential()).
 */
INLINE static void wind_launch(struct part *p, struct xpart *xp,
                               const struct wind_props *props,
                               const struct cosmology *cosmo,
                               const double time,
                               const integertime_t ti_current,
                               const float phi) {

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
  p->wind_data.launch_phi = phi;
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
 * energy; record them and add them to the particle's energy tracer.
 *
 * @param phi The gravitational potential per unit mass at the particle
 * (wind_specific_potential()).
 */
INLINE static void wind_recouple(struct part *p, struct xpart *xp,
                                 const struct wind_props *props,
                                 const struct cosmology *cosmo,
                                 const double time, const float phi) {

  p->wind_data.decoupled = 0;
  p->wind_data.recouple_time = time;
  p->wind_data.recouple_distance = wind_distance(p);
  p->wind_data.recouple_e_kin = wind_specific_kinetic(xp);
  p->wind_data.recouple_e_th = xp->u_full;
  p->wind_data.recouple_phi = phi;
  hydro_reset_acceleration(p);

  const double m = hydro_get_mass(p);
  const double E_rec =
      m * ((double)p->wind_data.recouple_e_kin + p->wind_data.recouple_e_th);
  const double E_pot = m * ((double)phi - p->wind_data.launch_phi);
  const double E_launch =
      m * ((double)p->wind_data.launch_e_kin + p->wind_data.launch_e_th);
  p->wind_data.tracer_mass += m;
  p->wind_data.tracer_energy += E_rec;
  p->wind_data.tracer_potential_energy += E_pot;
  p->wind_data.tracer_launch_energy += E_launch;

  /* The signal velocity of the first CFL time-step must know that the
   * particle arrives at its speed in gas that is, to first order, at rest */
  hydro_set_v_sig_based_on_velocity_kick(
      p, cosmo, sqrtf(2.f * p->wind_data.recouple_e_kin));

  atomic_add_d(&wind_global_stats.n_recouple, 1.);
  atomic_add_d(&wind_global_stats.m_recouple, m);
  atomic_add_d(&wind_global_stats.E_kin_recouple,
               m * p->wind_data.recouple_e_kin);
  atomic_add_d(&wind_global_stats.E_th_recouple,
               m * p->wind_data.recouple_e_th);
  atomic_add_d(&wind_global_stats.E_pot_recouple, E_pot);
  atomic_add_d(&wind_global_stats.E_launch_recouple, E_launch);
}

/**
 * @brief A gas particle becomes a star: move its energy tracer to the global
 * total held by stars, so that gas, flight and stars together account for
 * every launch.
 */
INLINE static void wind_part_becomes_star(const struct part *p) {
  if (p->wind_data.tracer_mass <= 0.f) return;
  atomic_add_d(&wind_global_stats.m_tracer_star, p->wind_data.tracer_mass);
  atomic_add_d(&wind_global_stats.E_tracer_star, p->wind_data.tracer_energy);
  atomic_add_d(&wind_global_stats.E_pot_tracer_star,
               p->wind_data.tracer_potential_energy);
  atomic_add_d(&wind_global_stats.E_launch_tracer_star,
               p->wind_data.tracer_launch_energy);
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
