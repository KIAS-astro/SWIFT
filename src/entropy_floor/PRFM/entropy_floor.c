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
#include <config.h>

/* System includes */
#include <string.h>

#ifdef HAVE_HDF5
#include <hdf5.h>
#endif

/* Local includes */
#include "chemistry.h"
#include "cosmology.h"
#include "entropy_floor.h"
#include "hydro.h"
#include "parser.h"

/*! Mean mass per hydrogen atom in units of m_H (J26: n_H = rho / 1.4 m_H) */
#define PRFM_FLOOR_MASS_PER_H 1.4

/**
 * @brief Floor pressure (internal units) at physical density rho_phys and
 * metallicity Z' (solar units).
 */
static float entropy_floor_prfm_pressure(
    const float rho_phys, const double Zprime,
    const struct entropy_floor_properties *props) {

  if (rho_phys < props->density_threshold) return 0.f;
  const double n_H = rho_phys * props->rho_to_n_H_cgs;
  return prfm_eos_pressure(n_H, &props->cal, Zprime) * props->pressure_from_cgs;
}

/**
 * @brief Metallicity of a particle in solar units.
 */
static double entropy_floor_prfm_Zprime(
    const struct part *p, const struct entropy_floor_properties *props) {
  return chemistry_get_total_metal_mass_fraction_for_star_formation(p) *
         props->solar_metallicity_inv;
}

/**
 * @brief Compute the pressure from the entropy floor at a given density.
 *
 * No particle is given, so K24 is evaluated at Z' = 1.
 *
 * @param rho_phys The physical density (internal units).
 * @param rho_com The comoving density (internal units).
 * @param cosmo The cosmological model.
 * @param props The properties of the entropy floor.
 */
float entropy_floor_gas_pressure(const float rho_phys, const float rho_com,
                                 const struct cosmology *cosmo,
                                 const struct entropy_floor_properties *props) {
  return entropy_floor_prfm_pressure(rho_phys, 1., props);
}

/**
 * @brief Compute the entropy floor of a given #part.
 *
 * @param p The #part.
 * @param cosmo The cosmological model.
 * @param props The properties of the entropy floor.
 */
float entropy_floor(const struct part *p, const struct cosmology *cosmo,
                    const struct entropy_floor_properties *props) {

  const float rho_phys = hydro_get_physical_density(p, cosmo);
  const float pressure = entropy_floor_prfm_pressure(
      rho_phys, entropy_floor_prfm_Zprime(p, props), props);

  /* Entropy is the same in co-moving and physical frames */
  return gas_entropy_from_pressure(rho_phys, pressure);
}

/**
 * @brief Temperature of neutral gas on the floor at a given density.
 *
 * @param rho_phys The physical density (internal units).
 * @param rho_com The comoving density (internal units).
 * @param cosmo The cosmological model.
 * @param props The properties of the entropy floor.
 */
float entropy_floor_gas_temperature(
    const float rho_phys, const float rho_com, const struct cosmology *cosmo,
    const struct entropy_floor_properties *props) {

  if (rho_phys <= 0.f) return 0.f;
  return entropy_floor_gas_pressure(rho_phys, rho_com, cosmo, props) *
         props->pressure_to_T / rho_phys;
}

/**
 * @brief Temperature of neutral gas on the floor for a given #part.
 *
 * @param p The #part.
 * @param cosmo The cosmological model.
 * @param props The properties of the entropy floor.
 */
float entropy_floor_temperature(const struct part *p,
                                const struct cosmology *cosmo,
                                const struct entropy_floor_properties *props) {

  const float rho_phys = hydro_get_physical_density(p, cosmo);
  if (rho_phys <= 0.f) return 0.f;
  return entropy_floor_prfm_pressure(rho_phys,
                                     entropy_floor_prfm_Zprime(p, props),
                                     props) *
         props->pressure_to_T / rho_phys;
}

/**
 * @brief Initialise the entropy floor from the PRFMEntropyFloor section.
 *
 * @param props The entropy floor properties to fill.
 * @param phys_const The physical constants.
 * @param us The system of units used internally.
 * @param hydro_props The properties of the hydro scheme.
 * @param params The YAML parameter file.
 */
void entropy_floor_init(struct entropy_floor_properties *props,
                        const struct phys_const *phys_const,
                        const struct unit_system *us,
                        const struct hydro_props *hydro_props,
                        struct swift_params *params) {

  char cal[32] = "OK22";
  parser_get_opt_param_string(params, "PRFMEntropyFloor:calibration", cal,
                              "OK22");
  if (strcmp(cal, "OK22") == 0)
    props->cal_choice = prfm_calibration_OK22;
  else if (strcmp(cal, "K24") == 0)
    props->cal_choice = prfm_calibration_K24;
  else
    error("PRFMEntropyFloor:calibration must be OK22 or K24, got '%s'", cal);
  prfm_calibration_init(props->cal_choice, &props->cal);

  props->density_threshold_H_p_cm3 = parser_get_opt_param_double(
      params, "PRFMEntropyFloor:density_threshold_H_p_cm3", 0.13);
  props->solar_metallicity_inv =
      1. / parser_get_opt_param_double(
               params, "PRFMEntropyFloor:solar_metallicity", 0.0134);

  const double rho_to_cgs = units_cgs_conversion_factor(us, UNIT_CONV_DENSITY);
  const double m_H_cgs =
      phys_const->const_proton_mass * units_cgs_conversion_factor(us, UNIT_CONV_MASS);

  props->rho_to_n_H_cgs = rho_to_cgs / (PRFM_FLOOR_MASS_PER_H * m_H_cgs);
  props->density_threshold =
      props->density_threshold_H_p_cm3 / props->rho_to_n_H_cgs;
  props->pressure_from_cgs =
      1. / units_cgs_conversion_factor(us, UNIT_CONV_PRESSURE);
  props->pressure_to_T = hydro_props->mu_neutral *
                         phys_const->const_proton_mass /
                         phys_const->const_boltzmann_k;
}

/**
 * @brief Print the properties of the entropy floor to stdout.
 *
 * @param props The entropy floor properties.
 */
void entropy_floor_print(const struct entropy_floor_properties *props) {

  message("Entropy floor is 'PRFM' (TIGRESS effective EoS) with:");
  message(
      "P/k_B = %.3g n_H^%.3g Z'^%.3g K cm^-3 above n_H = %.3g cm^-3 "
      "(n_H = rho / 1.4 m_H), calibration %s",
      props->cal.eos_norm, props->cal.eos_slope, props->cal.eos_Z,
      props->density_threshold_H_p_cm3,
      props->cal_choice == prfm_calibration_K24 ? "K24" : "OK22");
}

#ifdef HAVE_HDF5

/**
 * @brief Writes the current model of entropy floor to the file
 * @param h_grp The HDF5 group in which to write
 */
void entropy_floor_write_flavour(hid_t h_grp) {

  io_write_attribute_s(h_grp, "Entropy floor", "PRFM");
}
#endif

/**
 * @brief Write an entropy floor struct to the given FILE as a stream of bytes.
 *
 * @param props the struct
 * @param stream the file stream
 */
void entropy_floor_struct_dump(const struct entropy_floor_properties *props,
                               FILE *stream) {

  restart_write_blocks((void *)props, sizeof(struct entropy_floor_properties),
                       1, stream, "entropy floor", "entropy floor properties");
}

/**
 * @brief Restore a entropy floor struct from the given FILE as a stream of
 * bytes.
 *
 * @param props the struct
 * @param stream the file stream
 */
void entropy_floor_struct_restore(struct entropy_floor_properties *props,
                                  FILE *stream) {

  restart_read_blocks((void *)props, sizeof(struct entropy_floor_properties), 1,
                      stream, NULL, "entropy floor properties");
}
