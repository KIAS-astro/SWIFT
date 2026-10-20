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
 * @brief Mass loading of the warm channel (wind-design.md Sec. 10.3): the
 * constant Wind:warm_eta_M if positive, else 10^norm Sigma_SFR^slope; 0 if
 * the warm channel is off.
 */
INLINE static double wind_warm_eta_M(const struct wind_props *props,
                                     const struct xpart *xp) {
  if (!props->warm_phase || props->model == wind_loading_none) return 0.;
  if (props->warm_eta_M > 0.) return props->warm_eta_M;
  return wind_powerlaw_loading(props->warm_eta_M_log10_norm,
                               props->warm_eta_M_slope, xp->sf_data.Sigma_SFR);
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
  p->wind_data.launch_Sigma_SFR = -1.f;
  p->wind_data.launch_eta_M = -1.f;
  p->wind_data.launch_kick = -1.f;

  /* WindPhases and WindCREnergies are read from the ICs only for particles
   * decoupled there; a hand-decoupled particle without a phase is hot */
  if (!decoupled) {
    p->wind_data.phase = 0;
    p->wind_data.cr_energy = 0.;
  } else if (p->wind_data.phase != 2) {
    p->wind_data.phase = 1;
  }
  if (decoupled) {
    /* The launch direction of a hand-decoupled particle is the sign of its
     * initial velocity along n */
    const double vn = xp->v_full[0] * props->normal[0] +
                      xp->v_full[1] * props->normal[1] +
                      xp->v_full[2] * props->normal[2];
    p->wind_data.launch_sign = vn < 0. ? -1 : 1;
    if (p->wind_data.phase == 2)
      p->wind_data.d_rec =
          fmax(props->warm_d_rec,
               props->d_rec_kernel_factor * kernel_gamma * p->h);
    atomic_add_d(&wind_global_stats.E_cr_launch, p->wind_data.cr_energy);
  } else {
    p->wind_data.launch_sign = 1;
  }

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
 * @return 1 for a star, 2 for a hot launch, 3 for a warm launch, 0
 * otherwise (enum wind_draw4_outcome).
 */
__attribute__((always_inline)) INLINE static int wind_star_formation_draw(
    const struct wind_props *props, const struct part *p,
    const struct xpart *xp, const double p_star, const double u) {
  double eta_M, eta_E;
  wind_loadings(props, xp, &eta_M, &eta_E);
  return wind_draw4(u, p_star, eta_M * p_star,
                    wind_warm_eta_M(props, xp) * p_star);
}

/**
 * @brief Should a star-forming particle that did not form a star launch,
 * and in which phase?
 *
 * Repeats the draw of the star formation module with the same uniform
 * number, so that star and the two wind phases exclude each other.
 *
 * @return 0 for no launch, 1 for hot, 2 for warm.
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
  const int o = wind_star_formation_draw(props, p, xp, p_star, u);
  if (o == wind_draw4_hot) return 1;
  if (o == wind_draw4_warm) return 2;
  return 0;
}

/**
 * @brief Launch a particle: kick along +-n, set its internal energy,
 * give it its CR energy, decouple (WD3; warm phase and CR energy:
 * wind-design.md Sec. 10).
 *
 * @param phase 1 hot (kick and u_w from eta_M,hot and eta_E,hot), 2 warm
 * (kick warm_v, u = warm_u).
 */
INLINE static void wind_launch(struct part *p, struct xpart *xp,
                               const struct wind_props *props,
                               const struct cosmology *cosmo,
                               const double time,
                               const integertime_t ti_current,
                               const int phase) {

  double eta_M, eta_E, dv, u_w;
  wind_loadings(props, xp, &eta_M, &eta_E);
  const double eta_M_warm = wind_warm_eta_M(props, xp);
  if (phase == 2) {
    if (props->warm_speed_pdf) {
      const double u1 = random_unit_interval(p->id, ti_current,
                                             random_number_wind_warm_speed_1);
      const double u2 = random_unit_interval(p->id, ti_current,
                                             random_number_wind_warm_speed_2);
      dv = wind_warm_pdf_speed(u1, u2, props->warm_pdf_v0, props->warm_climb_v);
    } else {
      dv = props->warm_v;
    }
    u_w = props->warm_u;
  } else {
    wind_launch_kinematics(props->u_star, eta_M, eta_E, props->f_kin,
                           props->energy_convention, hydro_gamma, &dv, &u_w);
  }

  /* CR energy: eta_E,CR u_* per stellar mass in expectation, shared by the
   * carrying phases */
  double E_cr = 0.;
  if (props->cr_eta_E > 0. && (phase == 1 || props->cr_carriers_both)) {
    const double eta_carry = eta_M + (props->cr_carriers_both ? eta_M_warm : 0.);
    E_cr = hydro_get_mass(p) *
           wind_cr_specific_energy(props->u_star, props->cr_eta_E, eta_carry);
  }

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
  p->wind_data.d_rec =
      phase == 2 ? fmax(props->warm_d_rec,
                        props->d_rec_kernel_factor * kernel_gamma * p->h)
                 : wind_recoupling_distance(p, props);
  p->wind_data.phase = phase;
  p->wind_data.launch_sign = sign < 0. ? -1 : 1;
  p->wind_data.cr_energy = E_cr;
  p->wind_data.launch_Sigma_SFR = xp->sf_data.Sigma_SFR;
  p->wind_data.launch_eta_M = phase == 2 ? eta_M_warm : eta_M;
  p->wind_data.launch_kick = dv;

  const double m = hydro_get_mass(p);
  const double E_kick = m * (0.5 * dv * dv + hydro_gamma * u_w);
  atomic_add_d(&wind_global_stats.n_launch, 1.);
  atomic_add_d(&wind_global_stats.m_launch, m);
  atomic_add_d(&wind_global_stats.E_kin_launch, m * p->wind_data.launch_e_kin);
  atomic_add_d(&wind_global_stats.E_th_launch, m * p->wind_data.launch_e_th);
  /* A launch at loading eta stands for m / eta of stars in expectation, so
   * the sum of m / eta over the launches of a phase estimates the stellar
   * mass formed whatever the spread of Sigma_SFR (a test of the loadings) */
  if (phase == 2) {
    atomic_add_d(&wind_global_stats.n_launch_warm, 1.);
    atomic_add_d(&wind_global_stats.m_launch_warm, m);
    atomic_add_d(&wind_global_stats.m_over_eta_warm, m / eta_M_warm);
    atomic_add_d(&wind_global_stats.E_launch_warm, E_kick);
    atomic_add_d(&wind_global_stats.p_launch_warm, m * dv);
  } else {
    atomic_add_d(&wind_global_stats.n_launch_hot, 1.);
    atomic_add_d(&wind_global_stats.m_launch_hot, m);
    atomic_add_d(&wind_global_stats.m_over_eta_hot, m / eta_M);
    atomic_add_d(&wind_global_stats.E_launch_hot, E_kick);
    atomic_add_d(&wind_global_stats.p_launch_hot, m * dv);
  }
  atomic_add_d(&wind_global_stats.E_cr_launch, E_cr);
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
 * @brief Distance that decides recoupling: the 3D distance for the hot
 * phase; for the warm phase the height gained along the launch direction,
 * since a slow particle's 3D path is dominated by the disc rotation
 * (wind-design.md Sec. 10.3).
 */
__attribute__((always_inline)) INLINE static double wind_recoupling_measure(
    const struct part *p, const struct wind_props *props) {
  if (p->wind_data.phase != 2) return wind_distance(p);
  const double *dx = p->wind_data.dx;
  return p->wind_data.launch_sign *
         (dx[0] * props->normal[0] + dx[1] * props->normal[1] +
          dx[2] * props->normal[2]);
}

/**
 * @brief Should a decoupled particle recouple (WD5)? A warm particle also
 * recouples at its apex (wind-design.md Sec. 10.3).
 */
INLINE static int wind_should_recouple(const struct part *p,
                                       const struct xpart *xp,
                                       const struct wind_props *props,
                                       const double time) {
  if (!p->wind_data.decoupled) return 0;
  if (p->wind_data.phase == 2 &&
      wind_past_apex(xp->v_full, props->normal, p->wind_data.launch_sign))
    return 1;
  return wind_recoupling_measure(p, props) >= p->wind_data.d_rec ||
         time - p->wind_data.launch_time >= props->t_rec;
}

/**
 * @brief Recouple a particle (WD6): it keeps its velocity and internal
 * energy, to which its CR energy is added as heat if Wind:cr_deposit is
 * thermal; record them. The recorded internal energy is the one before the
 * deposit.
 */
INLINE static void wind_recouple(struct part *p, struct xpart *xp,
                                 const struct wind_props *props,
                                 const struct cosmology *cosmo,
                                 const double time) {

  if (p->wind_data.phase == 2 &&
      wind_past_apex(xp->v_full, props->normal, p->wind_data.launch_sign))
    atomic_add_d(&wind_global_stats.n_recouple_apex, 1.);

  p->wind_data.decoupled = 0;
  p->wind_data.recouple_time = time;
  p->wind_data.recouple_distance = wind_recoupling_measure(p, props);
  p->wind_data.recouple_e_kin = wind_specific_kinetic(xp);
  p->wind_data.recouple_e_th = xp->u_full;
  hydro_reset_acceleration(p);

  const double m = hydro_get_mass(p);
  const double E_cr = p->wind_data.cr_energy;
  float cs_new = 0.f;
  if (E_cr > 0.) {
    atomic_add_d(&wind_global_stats.E_cr_release, E_cr);
    if (props->cr_deposit_thermal) {
      const double u_new = xp->u_full + E_cr / m;
      hydro_set_physical_internal_energy(p, xp, cosmo, u_new);
      hydro_set_drifted_physical_internal_energy(p, cosmo, NULL, u_new);
      atomic_add_d(&wind_global_stats.E_cr_deposit, E_cr);
      cs_new = sqrtf(hydro_gamma * hydro_gamma_minus_one * u_new);
    }
    p->wind_data.cr_energy = 0.;
  }

  /* The signal velocity of the first CFL time-step must know that the
   * particle arrives at its speed in gas that is, to first order, at rest,
   * and with the sound speed of any CR energy just deposited */
  hydro_set_v_sig_based_on_velocity_kick(
      p, cosmo, sqrtf(2.f * p->wind_data.recouple_e_kin) + cs_new);

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
