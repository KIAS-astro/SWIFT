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
#ifndef SWIFT_WIND_TIGRESS_IO_H
#define SWIFT_WIND_TIGRESS_IO_H

/**
 * @file src/wind/TIGRESS/wind_io.h
 * @brief Decoupled wind: fields read from the initial conditions and written
 * to snapshots.
 */

#include "io_properties.h"

/**
 * @brief Read the optional flag WindDecoupled (int8; 1 decouples the
 * particle from the start).
 *
 * @return The number of fields to read.
 */
INLINE static int wind_read_particles(struct part *parts,
                                      struct io_props *list) {
  list[0] = io_make_input_field("WindDecoupled", CHAR, 1, OPTIONAL,
                                UNIT_CONV_NO_UNITS, parts, wind_data.decoupled);
  return 1;
}

/**
 * @brief The wind fields written to snapshots.
 *
 * @return The number of fields to write.
 */
INLINE static int wind_write_particles(const struct part *parts,
                                       const struct xpart *xparts,
                                       struct io_props *list) {

  list[0] = io_make_output_field("WindDecoupled", CHAR, 1, UNIT_CONV_NO_UNITS,
                                 0.f, parts, wind_data.decoupled,
                                 "1 while the particle is a decoupled wind "
                                 "particle, 0 otherwise");
  list[1] = io_make_output_field("WindLaunchCounts", INT, 1,
                                 UNIT_CONV_NO_UNITS, 0.f, parts,
                                 wind_data.launch_count,
                                 "Number of times the particle was launched");
  list[2] = io_make_output_field(
      "WindLaunchTimes", FLOAT, 1, UNIT_CONV_TIME, 0.f, parts,
      wind_data.launch_time, "Time of the last launch (-1 if never launched)");
  list[3] = io_make_output_field("WindLaunchPositions", DOUBLE, 3,
                                 UNIT_CONV_LENGTH, 1.f, parts,
                                 wind_data.launch_x,
                                 "Position at the last launch");
  list[4] = io_make_output_field(
      "WindDisplacements", DOUBLE, 3, UNIT_CONV_LENGTH, 1.f, parts,
      wind_data.dx,
      "Displacement since the last launch, accumulated while decoupled "
      "(not wrapped by periodic boundaries)");
  list[5] = io_make_output_field(
      "WindLaunchKineticEnergies", FLOAT, 1, UNIT_CONV_ENERGY_PER_UNIT_MASS,
      0.f, parts, wind_data.launch_e_kin,
      "Kinetic energy per unit mass just after the last launch (physical)");
  list[6] = io_make_output_field(
      "WindLaunchInternalEnergies", FLOAT, 1, UNIT_CONV_ENERGY_PER_UNIT_MASS,
      0.f, parts, wind_data.launch_e_th,
      "Internal energy per unit mass just after the last launch (physical)");
  list[7] = io_make_output_field(
      "WindRecouplingTimes", FLOAT, 1, UNIT_CONV_TIME, 0.f, parts,
      wind_data.recouple_time,
      "Time of the last recoupling (-1 if never recoupled)");
  list[8] = io_make_output_field(
      "WindRecouplingKineticEnergies", FLOAT, 1,
      UNIT_CONV_ENERGY_PER_UNIT_MASS, 0.f, parts, wind_data.recouple_e_kin,
      "Kinetic energy per unit mass at the last recoupling (physical)");
  list[9] = io_make_output_field(
      "WindRecouplingInternalEnergies", FLOAT, 1,
      UNIT_CONV_ENERGY_PER_UNIT_MASS, 0.f, parts, wind_data.recouple_e_th,
      "Internal energy per unit mass at the last recoupling (physical)");
  list[10] = io_make_output_field(
      "WindRecouplingDistances", FLOAT, 1, UNIT_CONV_LENGTH, 1.f, parts,
      wind_data.recouple_distance,
      "Distance from the launch point at the last recoupling");
  list[11] = io_make_output_field(
      "WindRecouplingDistanceThresholds", FLOAT, 1, UNIT_CONV_LENGTH, 1.f,
      parts, wind_data.d_rec,
      "Recoupling distance of the last launch, max(d_rec, factor x kernel "
      "support at launch)");
  list[12] = io_make_output_field(
      "WindLaunchPotentials", FLOAT, 1, UNIT_CONV_ENERGY_PER_UNIT_MASS, 0.f,
      parts, wind_data.launch_phi,
      "Gravitational potential per unit mass (self-gravity plus external) at "
      "the last launch (physical)");
  list[13] = io_make_output_field(
      "WindRecouplingPotentials", FLOAT, 1, UNIT_CONV_ENERGY_PER_UNIT_MASS,
      0.f, parts, wind_data.recouple_phi,
      "Gravitational potential per unit mass (self-gravity plus external) at "
      "the last recoupling (physical)");
  list[14] = io_make_output_field(
      "WindTracerMasses", FLOAT, 1, UNIT_CONV_MASS, 0.f, parts,
      wind_data.tracer_mass,
      "Wind mass this particle received at its recouplings, summed over "
      "recouplings");
  list[15] = io_make_output_field(
      "WindTracerEnergies", FLOAT, 1, UNIT_CONV_ENERGY, 0.f, parts,
      wind_data.tracer_energy,
      "Kinetic plus internal energy this particle carried at its "
      "recouplings, m (v^2/2 + u), summed over recouplings (physical)");
  list[16] = io_make_output_field(
      "WindTracerPotentialEnergies", FLOAT, 1, UNIT_CONV_ENERGY, 0.f, parts,
      wind_data.tracer_potential_energy,
      "Potential energy gained in flight, m (phi_recouple - phi_launch), "
      "summed over recouplings (physical)");
  list[17] = io_make_output_field(
      "WindTracerLaunchEnergies", FLOAT, 1, UNIT_CONV_ENERGY, 0.f, parts,
      wind_data.tracer_launch_energy,
      "Kinetic plus internal energy at launch, m (v^2/2 + u), of the flights "
      "that ended in the recouplings counted by the tracer (physical)");
  return 18;
}

#endif /* SWIFT_WIND_TIGRESS_IO_H */
