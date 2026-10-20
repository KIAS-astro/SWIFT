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
  list[1] = io_make_input_field("WindPhases", CHAR, 1, OPTIONAL,
                                UNIT_CONV_NO_UNITS, parts, wind_data.phase);
  list[2] = io_make_input_field("WindCREnergies", DOUBLE, 1, OPTIONAL,
                                UNIT_CONV_ENERGY, parts, wind_data.cr_energy);
  return 3;
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
      "Distance from the launch point at the last recoupling (warm phase: "
      "height gained along the launch direction)");
  list[11] = io_make_output_field(
      "WindRecouplingDistanceThresholds", FLOAT, 1, UNIT_CONV_LENGTH, 1.f,
      parts, wind_data.d_rec,
      "Recoupling distance of the last launch, max(d_rec, factor x kernel "
      "support at launch)");
  list[12] = io_make_output_field(
      "WindPhases", CHAR, 1, UNIT_CONV_NO_UNITS, 0.f, parts, wind_data.phase,
      "Phase of the last launch: 0 never launched, 1 hot, 2 warm");
  list[13] = io_make_output_field(
      "WindCREnergies", DOUBLE, 1, UNIT_CONV_ENERGY, 0.f, parts,
      wind_data.cr_energy,
      "CR energy carried while decoupled (0 after recoupling)");
  list[14] = io_make_output_field(
      "WindLaunchSigmaSFR", FLOAT, 1, UNIT_CONV_NO_UNITS, 0.f, parts,
      wind_data.launch_Sigma_SFR,
      "Sigma_SFR [Msun kpc^-2 yr^-1] of the particle at the last launch (-1 "
      "if never launched)");
  list[15] = io_make_output_field(
      "WindLaunchMassLoadings", FLOAT, 1, UNIT_CONV_NO_UNITS, 0.f, parts,
      wind_data.launch_eta_M,
      "Mass loading of the launched phase at the last launch (-1 if never "
      "launched)");
  list[16] = io_make_output_field(
      "WindLaunchKicks", FLOAT, 1, UNIT_CONV_VELOCITY, 0.f, parts,
      wind_data.launch_kick,
      "Speed of the kick along the launch direction at the last launch "
      "(physical; -1 if never launched)");
  list[17] = io_make_output_field(
      "WindAmbientDensities", FLOAT, 1, UNIT_CONV_DENSITY, -3.f, parts,
      wind_data.rho_ambient,
      "Gas density at the particle from the non-decoupled neighbours whose "
      "kernels contain it, at the start of the last step while decoupled "
      "(density recoupling or host deposit; -1 if never computed)");
  list[18] = io_make_output_field(
      "WindRecouplingReasons", CHAR, 1, UNIT_CONV_NO_UNITS, 0.f, parts,
      wind_data.recouple_reason,
      "Criterion that ended the last flight: 0 none, 1 distance or height, 2 "
      "density, 3 time, 4 apex");
  list[19] = io_make_output_field(
      "WindMergeTimes", FLOAT, 1, UNIT_CONV_TIME, 0.f, parts,
      wind_data.merge_time,
      "Time at which a wind particle was last merged into this particle as "
      "its host (-1 if never)");
  list[20] = io_make_output_field(
      "WindMergePositions", DOUBLE, 3, UNIT_CONV_LENGTH, 1.f, parts,
      wind_data.merge_x,
      "Position of this particle at the last merge of a wind particle into "
      "it (not shifted to the box centre; 0 if never a host)");
  return 21;
}

#endif /* SWIFT_WIND_TIGRESS_IO_H */
