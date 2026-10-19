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
#ifndef SWIFT_WIND_TIGRESS_PROPERTIES_H
#define SWIFT_WIND_TIGRESS_PROPERTIES_H

/**
 * @file src/wind/TIGRESS/wind_properties.h
 * @brief Decoupled wind with the TIGRESS loadings: parameters and global
 * counters (src/wind.c).
 */

#include "inline.h"
#include "wind_physics.h"

#include <stdio.h>

struct phys_const;
struct unit_system;
struct swift_params;
struct engine;

/**
 * @brief How eta_M and eta_E are chosen.
 */
enum wind_loading_model {
  wind_loading_none,     /*<! no launches */
  wind_loading_constant, /*<! constant (eta_M, eta_E) */
  wind_loading_TIGRESS   /*<! Kim et al. (2020b) hot phase at Sigma_SFR */
};

/**
 * @brief Properties of the wind model (internal units unless stated).
 */
struct wind_props {

  /*! Loading model */
  enum wind_loading_model model;

  /*! Constant loadings */
  double eta_M, eta_E;

  /*! Power-law loadings 10^norm Sigma_SFR^slope (TIGRESS model); defaults
   * are the K20b hot-phase fits at |z| = H */
  double eta_M_log10_norm, eta_M_slope, eta_E_log10_norm, eta_E_slope;

  /*! SN energy per stellar mass, 1e51 erg / 95.5 Msun */
  double u_star;

  /*! What eta_E counts: kinetic + enthalpy or kinetic + internal energy */
  enum wind_energy_convention energy_convention;

  /*! Kinetic fraction (Delta v^2 / 2) / (eta_E u_* / eta_M); Mach 1 is 1/4
   * (enthalpy) or 5/14 (internal) */
  double f_kin;

  /*! Unit vector of the launch direction (the disc normal), +- */
  double normal[3];

  /*! Recoupling distance (minimum) and time */
  double d_rec, t_rec;

  /*! The recoupling distance of a particle is at least this many kernel
   * supports (kernel_gamma h) at launch */
  double d_rec_kernel_factor;

  /*! Time-step cap while decoupled */
  float dt_max;

  /*! 1: the wind is carried by light gas particles spawned at tree rebuilds
   * from per-particle reservoirs (Wind:mode light); 0: whole particles */
  int light;

  /*! Light mode: base gas mass m_g and light mass m_l = m_g / N */
  double light_m_g, light_m_l;

  /*! Light mode: N = m_g / m_l and its upper bound */
  double light_N, light_max_mass_ratio;

  /*! Light mode: refinement clock (Bennett et al. 2025, eqs. 6-13): R_200,
   * t_0, k, f_w,thresh and f_w,ref */
  double light_R200, light_t0, light_k, light_fw_thresh, light_fw_ref;

  /*! Light mode: forced tree-rebuild interval and the time of the last
   * light-particle rebuild */
  double light_rebuild_interval, light_last_rebuild;

  /*! Light mode: optional central source (CC85 test): centre, radius, mass
   * and energy injection rates (rates 0: no source) and the mass injected
   * but not yet spawned */
  double source_centre[3], source_radius, source_Mdot, source_Edot;
  double source_m_carry;
};

/**
 * @brief Running totals of the launches and recouplings on this rank.
 * Energies in internal units, kinetic and thermal separately.
 */
struct wind_global_stats {
  double n_launch, m_launch, E_kin_launch, E_th_launch;
  double n_recouple, m_recouple, E_kin_recouple, E_th_recouple;

  /*! Light mode: mass and energy added to reservoirs, mass that left with a
   * star, light particles spawned (number, mass), mass injected by the
   * central source, merges (number, mass of the removed partners), relative
   * kinetic energy turned into heat at merging. Reservoir mass: added =
   * spawned (minus source) + rounding + to stars + held */
  double m_pending_added, E_pending_added, m_pending_to_stars;

  /*! Light mode: reservoir mass minus the mass spawned from it, summed over
   * the draws at the rebuilds (zero in expectation; it stays in the
   * parents) */
  double m_rounding;
  double n_spawn, m_spawn, m_source;
  double n_merge, m_merge, E_merge_heat;

  /*! Momentum given by all launch kicks, sum of m Delta v (both modes) */
  double P_launch[3];
};

extern struct wind_global_stats wind_global_stats;

void wind_props_init(struct wind_props *props,
                     const struct phys_const *phys_const,
                     const struct unit_system *us,
                     struct swift_params *params);
void wind_props_print(const struct wind_props *props);
void wind_struct_dump(const struct wind_props *props, FILE *stream);
void wind_struct_restore(struct wind_props *props, FILE *stream);
void wind_print_stats(const struct engine *e);

#endif /* SWIFT_WIND_TIGRESS_PROPERTIES_H */
