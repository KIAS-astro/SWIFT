/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2018 Folkert Nobels (nobels@strw.leidenuniv.nl)
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
#ifndef SWIFT_STAR_FORMATION_PRFM_IO_H
#define SWIFT_STAR_FORMATION_PRFM_IO_H

/* Config parameters. */
#include <config.h>

/* Local includes */
#include "io_properties.h"

/**
 * @brief Specifies which s-particle fields to read from a dataset
 *
 * @param sparts The s-particle array.
 * @param list The list of i/o properties to read.
 *
 * @return num_fields The number of i/o fields to read.
 */
INLINE static int star_formation_read_particles(struct spart *sparts,
                                                struct io_props *list) {
  return 0;
}

/**
 * @brief Specifies which particle fields to write to a dataset
 *
 * @param parts The particle array.
 * @param xparts The extended data particle array.
 * @param list The list of i/o properties to write.
 *
 * @return Returns the number of fields to write.
 */
__attribute__((always_inline)) INLINE static int star_formation_write_particles(
    const struct part *parts, const struct xpart *xparts,
    struct io_props *list) {

  list[0] = io_make_output_field(
      "StarFormationRates", FLOAT, 1, UNIT_CONV_SFR, 0.f, xparts, sf_data.SFR,
      "If positive, star formation rates of the particles. If negative, stores "
      "the last time/scale-factor at which the gas particle was star-forming. "
      "If zero, the particle was never star-forming.");

  list[1] = io_make_output_field(
      "PRFMGasSurfaceDensities", FLOAT, 1, UNIT_CONV_MASS_PER_UNIT_AREA, 0.f,
      xparts, sf_data.Sigma_g,
      "PRFM gas surface density Sigma_g used at the particle's last active "
      "step: of the face-on map, or of the kernel column with its "
      "correction. Zero if the particle saw no gas on the map (map "
      "estimator).");
  list[9] = io_make_output_field(
      "PRFMMapGasSurfaceDensities", FLOAT, 1, UNIT_CONV_MASS_PER_UNIT_AREA,
      0.f, xparts, sf_data.Sigma_g_map,
      "PRFM gas surface density Sigma_g of the face-on map at the particle, "
      "at its last active step, whichever estimator is used.");
  list[2] = io_make_output_field(
      "PRFMStellarSurfaceDensities", FLOAT, 1, UNIT_CONV_MASS_PER_UNIT_AREA,
      0.f, xparts, sf_data.Sigma_s,
      "PRFM stellar surface density Sigma_* of the face-on map at the "
      "particle, at its last active step.");
  list[3] = io_make_output_field(
      "PRFMStellarVerticalDispersions", FLOAT, 1, UNIT_CONV_VELOCITY, 0.f,
      xparts, sf_data.sigma_sz,
      "PRFM stellar vertical velocity dispersion sigma_*z of the face-on map "
      "at the particle, at its last active step.");
  list[4] = io_make_output_field(
      "PRFMDarkMatterDensities", FLOAT, 1, UNIT_CONV_DENSITY, 0.f, xparts,
      sf_data.rho_d,
      "PRFM dark matter density rho_d at the particle used in the weight, at "
      "its last active step.");
  list[5] = io_make_output_field(
      "PRFMWeights", FLOAT, 1, UNIT_CONV_PRESSURE, 0.f, xparts, sf_data.W,
      "PRFM weight W of the gas column (J26 eq. 31), the equilibrium midplane "
      "pressure, at the particle's last active step.");
  list[6] = io_make_output_field(
      "PRFMEffectiveDispersions", FLOAT, 1, UNIT_CONV_VELOCITY, 0.f, xparts,
      sf_data.sigma_eff,
      "PRFM effective velocity dispersion sigma_eff(W) of the calibration, at "
      "the particle's last active step.");
  list[7] = io_make_output_field(
      "PRFMEquilibriumDensities", FLOAT, 1, UNIT_CONV_DENSITY, 0.f, xparts,
      sf_data.rho_eq,
      "PRFM equilibrium midplane density W / sigma_eff^2 = Sigma_g / (2 H_g) "
      "(n_H,eq = rho_eq / 1.4 m_H), at the particle's last active step.");
  list[8] = io_make_output_field(
      "PRFMEquilibriumScaleHeights", FLOAT, 1, UNIT_CONV_LENGTH, 0.f, xparts,
      sf_data.H_g,
      "PRFM equilibrium gas half-thickness H_g = Sigma_g sigma_eff^2 / (2 W) "
      "(t_dyn = 2 H_g / sigma_eff), at the particle's last active step.");

  return 10;
}

/**
 * @brief Specifies which sparticle fields to write to a dataset
 *
 * @param sparts The star particle array.
 * @param list The list of i/o properties to write.
 *
 * @return Returns the number of fields to write.
 */
__attribute__((always_inline)) INLINE static int
star_formation_write_sparticles(const struct spart *sparts,
                                struct io_props *list) {
  return 0;
}
#endif /* SWIFT_STAR_FORMATION_PRFM_IO_H */
