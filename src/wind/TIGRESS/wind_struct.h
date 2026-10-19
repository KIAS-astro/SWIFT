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

  /*! Number of launches of this particle */
  int launch_count;

  /*! 1 while the particle is decoupled from the hydrodynamics */
  char decoupled;
};

/**
 * @brief Why a wind carrier deposits.
 */
enum wind_carrier_criterion {
  wind_carrier_flying = 0,   /*<! not yet */
  wind_carrier_distance = 1, /*<! travelled d_rec */
  wind_carrier_time = 2      /*<! age reached t_rec */
};

/**
 * @brief Wind-carrier data carried by a star particle (wind-carrier-design.md).
 *
 * A carrier is a star particle that moves wind material from star-forming
 * gas to the place where it recouples. Energies per unit mass, physical.
 */
struct wind_spart_data {

  /*! Displacement since the launch, accumulated in the drift */
  double dx[3];

  /*! Position at the launch */
  double launch_x[3];

  /*! Time of the launch */
  double launch_time;

  /*! Internal energy per unit mass carried */
  float u;

  /*! Recoupling distance: max(d_rec, factor x kernel support at launch) */
  float d_rec;

  /*! Gravitational acceleration at the last time-step, for the potential
   * energy of the deposited mass */
  float a_grav[3];

  /*! Sum of the kernel weights of the gas neighbours at the deposit */
  float weight_sum;

  /*! Number of gas neighbours at the deposit */
  int ngb_count;

  /*! 1 for a carrier, 0 for an ordinary star particle */
  char carrier;

  /*! Why the carrier deposits (#wind_carrier_criterion), 0 while flying */
  char deposit;
};

#endif /* SWIFT_WIND_TIGRESS_STRUCT_H */
