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
#ifndef SWIFT_WIND_PHYSICS_H
#define SWIFT_WIND_PHYSICS_H

/**
 * @file src/wind_physics.h
 * @brief Decoupled galactic wind: the pure functions of the launch, shared by
 * the TIGRESS wind module (src/wind/TIGRESS/) and the unit test
 * tests/testWind.c.
 *
 * Loadings: hot-phase fits of Kim et al. (2020b, Table 5) at |z| = H,
 *    eta_M = 10^-0.86 Sigma_SFR^-0.07,  eta_E = 10^-0.70 Sigma_SFR^0.14,
 * Sigma_SFR in Msun kpc^-2 yr^-1, clipped to [1e-4, 1].
 * Launch (Smith et al. 2024, eqs. 6-8): with u_* the SN energy per stellar
 * mass, a kick Delta v = (2 eta_E,kin u_* / eta_M)^1/2 and an internal energy
 * u_w = eta_E,th u_* / eta_M, with eta_E,kin / eta_E,th = 5/9 (Mach 1 for
 * gamma = 5/3).
 * Draw (Springel & Hernquist 2003): one uniform number decides star (u < p_*),
 * wind (p_* <= u < p_* + p_w) or nothing, p_w = eta_M p_*; if p_* + p_w > 1
 * both are divided by their sum.
 */

/* Some standard headers. */
#include <math.h>

/*! Lower and upper end of the Sigma_SFR range of the TIGRESS fits */
#define WIND_SIGMA_SFR_MIN 1e-4
#define WIND_SIGMA_SFR_MAX 1.

/*! Fraction of the wind energy launched as kinetic energy at Mach 1:
 * eta_E,kin / eta_E = 5 / (5 + 9) */
#define WIND_KINETIC_FRACTION_MACH1 (5. / 14.)

/*! SN energy per stellar mass formed, 1e51 erg / 95.5 Msun [erg g^-1] */
#define WIND_U_STAR_CGS (1e51 / (95.5 * 1.98841e33))

/**
 * @brief Outcome of the star-formation draw.
 */
enum wind_draw_outcome {
  wind_draw_nothing = 0,
  wind_draw_star = 1,
  wind_draw_wind = 2
};

/**
 * @brief Sigma_SFR clipped to the fitted range [Msun kpc^-2 yr^-1].
 */
static inline double wind_clip_Sigma_SFR(const double Sigma_SFR) {
  return fmin(fmax(Sigma_SFR, WIND_SIGMA_SFR_MIN), WIND_SIGMA_SFR_MAX);
}

/*! Hot-phase fits of K20b Table 5 at |z| = H: log10 of the normalisation
 * and slope against Sigma_SFR [Msun kpc^-2 yr^-1] */
#define WIND_TIGRESS_ETA_M_LOG10_NORM -0.86
#define WIND_TIGRESS_ETA_M_SLOPE -0.07
#define WIND_TIGRESS_ETA_E_LOG10_NORM -0.70
#define WIND_TIGRESS_ETA_E_SLOPE 0.14

/**
 * @brief A loading 10^log10_norm Sigma_SFR^slope, Sigma_SFR clipped to
 * [1e-4, 1] [Msun kpc^-2 yr^-1].
 */
static inline double wind_powerlaw_loading(const double log10_norm,
                                           const double slope,
                                           const double Sigma_SFR) {
  return pow(10., log10_norm) * pow(wind_clip_Sigma_SFR(Sigma_SFR), slope);
}

/**
 * @brief TIGRESS hot-phase mass loading eta_M (K20b Table 5).
 *
 * @param Sigma_SFR [Msun kpc^-2 yr^-1], clipped to [1e-4, 1].
 */
static inline double wind_TIGRESS_eta_M(const double Sigma_SFR) {
  return wind_powerlaw_loading(WIND_TIGRESS_ETA_M_LOG10_NORM,
                               WIND_TIGRESS_ETA_M_SLOPE, Sigma_SFR);
}

/**
 * @brief TIGRESS hot-phase energy loading eta_E (K20b Table 5).
 *
 * @param Sigma_SFR [Msun kpc^-2 yr^-1], clipped to [1e-4, 1].
 */
static inline double wind_TIGRESS_eta_E(const double Sigma_SFR) {
  return wind_powerlaw_loading(WIND_TIGRESS_ETA_E_LOG10_NORM,
                               WIND_TIGRESS_ETA_E_SLOPE, Sigma_SFR);
}

/**
 * @brief Launch kick and internal energy of a wind particle.
 *
 * m (Delta v^2 / 2 + u_w) = m u_* eta_E / eta_M.
 *
 * @param u_star SN energy per stellar mass (any units of energy per mass).
 * @param eta_M Mass loading (> 0).
 * @param eta_E Energy loading.
 * @param f_kin Kinetic fraction eta_E,kin / eta_E.
 * @param dv (return) The kick, in the velocity units of u_star^1/2.
 * @param u_w (return) The internal energy per mass.
 */
static inline void wind_launch_kinematics(const double u_star,
                                          const double eta_M,
                                          const double eta_E,
                                          const double f_kin, double *dv,
                                          double *u_w) {
  const double e_w = eta_E * u_star / eta_M;
  *dv = sqrt(2. * f_kin * e_w);
  *u_w = (1. - f_kin) * e_w;
}

/**
 * @brief Star, wind or nothing from one uniform number.
 *
 * @param u Uniform number in [0, 1).
 * @param p_star Probability of forming a star, mdot_* dt / m_g.
 * @param p_wind Probability of launching, eta_M p_star.
 */
static inline enum wind_draw_outcome wind_draw(const double u,
                                               const double p_star,
                                               const double p_wind) {
  const double total = p_star + p_wind;
  const double norm = total > 1. ? 1. / total : 1.;
  if (u < p_star * norm) return wind_draw_star;
  if (u < total * norm) return wind_draw_wind;
  return wind_draw_nothing;
}

/**
 * @brief Number of wind carriers launched in a step: floor(p_c), plus one
 * with probability p_c - floor(p_c). The expectation is p_c.
 *
 * @param p_c Expected number, eta_M mdot_* dt / m_c.
 * @param u Uniform number in [0, 1).
 */
static inline int wind_carrier_count(const double p_c, const double u) {
  const double n = floor(p_c);
  return (int)n + (u < p_c - n ? 1 : 0);
}

/**
 * @brief Kernel share of one gas neighbour in a carrier's deposit.
 *
 * @param w The neighbour's kernel value W(r_j, h_c).
 * @param w_sum The sum of W over all neighbours.
 */
static inline double wind_carrier_share(const double w, const double w_sum) {
  return w / w_sum;
}

/**
 * @brief Mass fraction (of a metal) after adding dm at fraction f_c to m at
 * fraction f; conserves the metal mass m f + dm f_c.
 */
static inline double wind_carrier_mix(const double m, const double f,
                                      const double dm, const double f_c) {
  return (m * f + dm * f_c) / (m + dm);
}

/**
 * @brief Deposit a mass dm of a carrier into one gas particle.
 *
 * Mass and momentum add; the thermal energy is what makes the total
 * (kinetic + thermal + potential) energy of the pair the same before and
 * after:
 *   E_th' = m u + dm (v_c^2 / 2 + u_c) + m v^2 / 2 - (m + dm) v'^2 / 2
 *           - dm a_c . (x_c - x),
 * where the last term is the potential energy gained by moving dm from the
 * carrier to the gas particle (exact in a uniform field a_c).
 *
 * @param m Gas mass.
 * @param v Gas velocity.
 * @param u Gas internal energy per mass.
 * @param dm Deposited mass.
 * @param v_c Carrier velocity.
 * @param u_c Carrier internal energy per mass.
 * @param a_dx a_c . (x_c - x).
 * @param v_new (return) The gas velocity after the deposit.
 * @return The gas thermal energy (not per mass) after the deposit.
 */
static inline double wind_carrier_deposit(const double m, const double v[3],
                                          const double u, const double dm,
                                          const double v_c[3],
                                          const double u_c, const double a_dx,
                                          double v_new[3]) {
  const double m_new = m + dm;
  double v2 = 0., v2_new = 0., v2_c = 0.;
  for (int k = 0; k < 3; k++) {
    v_new[k] = (m * v[k] + dm * v_c[k]) / m_new;
    v2 += v[k] * v[k];
    v2_new += v_new[k] * v_new[k];
    v2_c += v_c[k] * v_c[k];
  }
  return m * u + dm * (0.5 * v2_c + u_c) + 0.5 * m * v2 -
         0.5 * m_new * v2_new - dm * a_dx;
}

#endif /* SWIFT_WIND_PHYSICS_H */
