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
#ifndef SWIFT_ENTROPY_FLOOR_PRFM_H
#define SWIFT_ENTROPY_FLOOR_PRFM_H

/* Code config */
#include <config.h>

/* System include */
#include <stdio.h>

/* Local includes */
#include "prfm_physics.h"

/* Pre-declarations */
struct cosmology;
struct part;
struct phys_const;
struct unit_system;
struct hydro_props;
struct swift_params;

/**
 * @file src/entropy_floor/PRFM/entropy_floor.h
 * @brief The TIGRESS effective equation of state used as a pressure floor.
 *
 * Above n_H = rho / (1.4 m_H) >= n_th the pressure is held at or above
 * P_eff(n_H): OK22 (J26) P/k_B = 4.7e4 n_H^1.8 K cm^-3, or K24
 * P/k_B = 2.85e4 n_H^1.29 Z'^-0.10. Below n_th there is no floor.
 * It is applied wherever the EAGLE floor is (cooling and hydro kick), through
 * the same interface.
 */

/**
 * @brief Properties of the PRFM entropy floor.
 */
struct entropy_floor_properties {

  /*! Calibration of the effective EoS */
  struct prfm_calibration cal;

  /*! Which calibration (for printing) */
  enum prfm_calibration_choice cal_choice;

  /*! Density threshold n_th in H atoms per cm^3 */
  double density_threshold_H_p_cm3;

  /*! Density threshold as a physical mass density (internal units) */
  double density_threshold;

  /*! Mass density to n_H in cm^-3: n_H = rho * rho_to_n_H_cgs */
  double rho_to_n_H_cgs;

  /*! Pressure from cgs to internal units */
  double pressure_from_cgs;

  /*! P = rho k_B T / (mu m_p): T = P * pressure_to_T / rho */
  double pressure_to_T;

  /*! Inverse of the solar metal mass fraction, for Z' */
  double solar_metallicity_inv;
};

float entropy_floor_gas_pressure(const float rho_phys, const float rho_com,
                                 const struct cosmology *cosmo,
                                 const struct entropy_floor_properties *props);

float entropy_floor(const struct part *p, const struct cosmology *cosmo,
                    const struct entropy_floor_properties *props);

float entropy_floor_gas_temperature(
    const float rho_phys, const float rho_com, const struct cosmology *cosmo,
    const struct entropy_floor_properties *props);

float entropy_floor_temperature(const struct part *p,
                                const struct cosmology *cosmo,
                                const struct entropy_floor_properties *props);

void entropy_floor_init(struct entropy_floor_properties *props,
                        const struct phys_const *phys_const,
                        const struct unit_system *us,
                        const struct hydro_props *hydro_props,
                        struct swift_params *params);

void entropy_floor_print(const struct entropy_floor_properties *props);

#ifdef HAVE_HDF5
void entropy_floor_write_flavour(hid_t h_grp);
#endif

void entropy_floor_struct_dump(const struct entropy_floor_properties *props,
                               FILE *stream);

void entropy_floor_struct_restore(struct entropy_floor_properties *props,
                                  FILE *stream);

#endif /* SWIFT_ENTROPY_FLOOR_PRFM_H */
