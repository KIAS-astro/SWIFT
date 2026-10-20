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

  /*! CR energy carried while decoupled and released at recoupling; a total,
   * not per mass, so that launched = deposited to round-off */
  double cr_energy;

  /*! Sigma_SFR [Msun kpc^-2 yr^-1] and the mass loading of the phase at
   * the last launch */
  float launch_Sigma_SFR;
  float launch_eta_M;

  /*! Speed of the kick of the last launch */
  float launch_kick;

  /*! Gas density at the particle from the non-decoupled neighbours whose
   * kernels contain it (scatter estimate), set at the start of every step
   * while decoupled when recoupling by density or depositing into a host;
   * -1 when not computed */
  float rho_ambient;

  /*! Time and position of the last merge of a wind particle into this
   * particle as its host (deposit into a host); -1 if never a host */
  double merge_x[3];
  float merge_time;

  /*! Number of launches of this particle */
  int launch_count;

  /*! 1 while the particle is decoupled from the hydrodynamics */
  char decoupled;

  /*! Phase of the last launch: 0 none, 1 hot, 2 warm */
  char phase;

  /*! Sign of the last launch along the launch direction, +-1 */
  char launch_sign;

  /*! Criterion that ended the last flight (enum wind_recoupling_reason) */
  char recouple_reason;

  /*! Criterion met, waiting to be merged into its host at the next step
   * (deposit into a host); 0 otherwise */
  char recouple_pending;
};

#endif /* SWIFT_WIND_TIGRESS_STRUCT_H */
