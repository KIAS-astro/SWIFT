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
 * Energies are per unit mass, physical, in internal units, except the
 * tracer fields, which are masses and energies. The recoupling times,
 * energies and distance are negative until the particle first recouples.
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

  /*! Gravitational potential per mass at the last launch */
  float launch_phi;

  /*! 1 until launch_phi is set at the first time-step of a particle
   * decoupled in the initial conditions, when its potential first exists */
  char launch_phi_pending;

  /*! Time of the last recoupling */
  float recouple_time;

  /*! Kinetic energy per mass at the last recoupling */
  float recouple_e_kin;

  /*! Internal energy per mass at the last recoupling */
  float recouple_e_th;

  /*! Gravitational potential per mass at the last recoupling */
  float recouple_phi;

  /*! Energy tracer: wind mass received at recoupling, summed over all the
   * recouplings of this particle. A gas particle keeps it until it forms a
   * star; SPH does not mix it into neighbours. */
  float tracer_mass;

  /*! Energy tracer: kinetic plus internal energy received at recoupling,
   * m (e_kin + u), summed over recouplings */
  float tracer_energy;

  /*! Energy tracer: potential energy gained in flight, m (phi_rec -
   * phi_launch), summed over recouplings */
  float tracer_potential_energy;

  /*! Energy tracer: launch energy m (e_kin + u) of the flights that ended in
   * the recouplings counted above */
  float tracer_launch_energy;

  /*! Distance from the launch point at the last recoupling */
  float recouple_distance;

  /*! Recoupling distance of this launch: max(d_rec, factor kernel_gamma h) */
  float d_rec;

  /*! Number of launches of this particle */
  int launch_count;

  /*! 1 while the particle is decoupled from the hydrodynamics */
  char decoupled;
};

#endif /* SWIFT_WIND_TIGRESS_STRUCT_H */
