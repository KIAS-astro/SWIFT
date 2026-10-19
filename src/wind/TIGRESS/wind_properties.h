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

  /*! Kinetic fraction eta_E,kin / eta_E (5/14 for Mach 1) */
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
};

/**
 * @brief Running totals of the launches and recouplings on this rank.
 * Energies in internal units, kinetic and thermal separately.
 */
struct wind_global_stats {
  double n_launch, m_launch, E_kin_launch, E_th_launch;
  double n_recouple, m_recouple, E_kin_recouple, E_th_recouple;
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
