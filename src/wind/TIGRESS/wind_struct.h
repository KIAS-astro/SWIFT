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

  /*! Minimum distance for pressure recoupling of this launch */
  float d_min;

  /*! Subgrid CC85 wind of this launch: source radius, mass injection rate,
   * asymptotic speed */
  float cc85_R, cc85_Mdot, cc85_v_inf;

  /*! CC85 and ambient pressures at the last recoupling */
  float recouple_P_w, recouple_P_amb;

  /*! Sigma_SFR [Msun kpc^-2 yr^-1] and the loadings used at the last
   * launch */
  float launch_Sigma_SFR, launch_eta_M, launch_eta_E;

  /*! Number of launches of this particle */
  int launch_count;

  /*! 1 while the particle is decoupled from the hydrodynamics */
  char decoupled;

  /*! Why the particle last recoupled (enum wind_recoupling_reason) */
  char recouple_reason;
};

#endif /* SWIFT_WIND_TIGRESS_STRUCT_H */
