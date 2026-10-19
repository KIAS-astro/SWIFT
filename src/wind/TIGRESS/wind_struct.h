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
#ifndef SWIFT_WIND_TIGRESS_STRUCT_H
#define SWIFT_WIND_TIGRESS_STRUCT_H

/**
 * @file src/wind/TIGRESS/wind_struct.h
 * @brief Decoupled wind: the data carried by a gas particle.
 */

/**
 * @brief Wind data carried by a gas particle.
 *
 * Energies are per unit mass, physical, in internal units. The recoupling
 * fields are negative until the particle first recouples.
 */
struct wind_part_data {

  /*! Displacement since the launch, accumulated in the drift (so that it is
   * not wrapped by periodic boundaries) */
  double dx[3];

  /*! Position at the launch */
  double launch_x[3];

  /*! Time of the last launch */
  float launch_time;

  /*! Kinetic energy per mass just after the last launch */
  float launch_e_kin;

  /*! Internal energy per mass just after the last launch */
  float launch_e_th;

  /*! Time of the last recoupling */
  float recouple_time;

  /*! Kinetic energy per mass at the last recoupling */
  float recouple_e_kin;

  /*! Internal energy per mass at the last recoupling */
  float recouple_e_th;

  /*! Distance from the launch point at the last recoupling */
  float recouple_distance;

  /*! Recoupling distance of this launch: max(d_rec, factor kernel_gamma h) */
  float d_rec;

  /*! Gas density at the particle from the non-decoupled neighbours whose
   * kernels contain it (scatter estimate), set at the start of every step
   * while decoupled when recoupling by density or depositing into a host;
   * -1 when not computed */
  float rho_ambient;

  /*! Number of launches of this particle */
  int launch_count;

  /*! 1 while the particle is decoupled from the hydrodynamics */
  char decoupled;

  /*! Criterion that ended the last flight (enum wind_recoupling_reason) */
  char recouple_reason;

  /*! Criterion met, waiting to be merged into its host at the next step
   * (deposit into a host); 0 otherwise */
  char recouple_pending;
};

#endif /* SWIFT_WIND_TIGRESS_STRUCT_H */
