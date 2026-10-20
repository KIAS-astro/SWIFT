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
 * Launch: with u_* the SN energy per stellar mass, the energy per launched
 * mass e_w = eta_E u_* / eta_M is split into a kick Delta v and an internal
 * energy u_w, the kinetic part being a fraction f of e_w. Two conventions
 * (Wind:energy_convention):
 *  - enthalpy (K20b, whose eta_E counts the kinetic plus enthalpy flux):
 *      Delta v^2 / 2 + gamma u_w = e_w,
 *    so that the Bernoulli speed (Delta v^2 + 2 gamma u_w)^1/2 = (2 e_w)^1/2;
 *  - internal (Smith et al. 2024, eqs. 6-8; Arkenstone):
 *      Delta v^2 / 2 + u_w = e_w.
 * At Mach 1 the sound speed squared gamma (gamma - 1) u_w equals Delta v^2.
 * With enthalpy, (Delta v^2 / 2) : gamma u_w = (gamma - 1) / 2, so
 *   f = (gamma - 1) / (gamma + 1) = 1/4 for gamma = 5/3;
 * with internal energy, (Delta v^2 / 2) : u_w = gamma (gamma - 1) / 2, so
 *   f = gamma (gamma - 1) / (2 + gamma (gamma - 1)) = 5/14 for gamma = 5/3.
 * Draw (Springel & Hernquist 2003): one uniform number decides star (u < p_*),
 * wind (p_* <= u < p_* + p_w) or nothing, p_w = eta_M p_*; if p_* + p_w > 1
 * both are divided by their sum.
 */

/* Some standard headers. */
#include <math.h>

/*! Lower and upper end of the Sigma_SFR range of the TIGRESS fits */
#define WIND_SIGMA_SFR_MIN 1e-4
#define WIND_SIGMA_SFR_MAX 1.

/**
 * @brief What the energy loading eta_E counts at launch.
 */
enum wind_energy_convention {
  wind_energy_internal = 0, /*<! kinetic + internal energy */
  wind_energy_enthalpy = 1  /*<! kinetic energy + enthalpy (K20b) */
};

/**
 * @brief Kinetic fraction f = (Delta v^2 / 2) / e_w of a Mach-1 launch.
 *
 * Mach 1 means Delta v^2 = gamma (gamma - 1) u_w. Enthalpy:
 * (Delta v^2 / 2) / (gamma u_w) = (gamma - 1) / 2 = f / (1 - f), so
 * f = (gamma - 1) / (gamma + 1). Internal energy:
 * (Delta v^2 / 2) / u_w = gamma (gamma - 1) / 2 = f / (1 - f), so
 * f = gamma (gamma - 1) / (2 + gamma (gamma - 1)).
 *
 * @param conv The energy convention.
 * @param gamma The adiabatic index.
 */
static inline double wind_kinetic_fraction_mach1(
    const enum wind_energy_convention conv, const double gamma) {
  if (conv == wind_energy_enthalpy) return (gamma - 1.) / (gamma + 1.);
  const double r = gamma * (gamma - 1.);
  return r / (2. + r);
}

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
 * With e_w = u_* eta_E / eta_M and h = gamma (enthalpy) or 1 (internal):
 * Delta v^2 / 2 = f e_w and h u_w = (1 - f) e_w.
 *
 * @param u_star SN energy per stellar mass (any units of energy per mass).
 * @param eta_M Mass loading (> 0).
 * @param eta_E Energy loading.
 * @param f_kin Kinetic fraction (Delta v^2 / 2) / e_w.
 * @param conv The energy convention.
 * @param gamma The adiabatic index.
 * @param dv (return) The kick, in the velocity units of u_star^1/2.
 * @param u_w (return) The internal energy per mass.
 */
static inline void wind_launch_kinematics(
    const double u_star, const double eta_M, const double eta_E,
    const double f_kin, const enum wind_energy_convention conv,
    const double gamma, double *dv, double *u_w) {
  const double e_w = eta_E * u_star / eta_M;
  const double h = conv == wind_energy_enthalpy ? gamma : 1.;
  *dv = sqrt(2. * f_kin * e_w);
  *u_w = (1. - f_kin) * e_w / h;
}

/**
 * @brief Bernoulli speed (Delta v^2 + 2 gamma u)^1/2.
 */
static inline double wind_bernoulli_speed(const double dv, const double u,
                                          const double gamma) {
  return sqrt(dv * dv + 2. * gamma * u);
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

/*
 * Two-phase wind (wind-design.md Sec. 10): the hot channel launched above,
 * and a warm channel launched at a fixed speed and temperature that
 * recouples near the disc-halo interface or at its apex.
 */

/**
 * @brief Outcome of the four-way draw.
 */
enum wind_draw4_outcome {
  wind_draw4_nothing = 0,
  wind_draw4_star = 1,
  wind_draw4_hot = 2,
  wind_draw4_warm = 3
};

/**
 * @brief Star, hot wind, warm wind or nothing from one uniform number.
 *
 * The outcomes are the disjoint intervals [0, p_* N), [p_* N, (p_* + p_h) N),
 * [(p_* + p_h) N, (p_* + p_h + p_w) N) and the rest of [0, 1), with N = 1 if
 * p_* + p_h + p_w <= 1 and 1 / (p_* + p_h + p_w) otherwise. With p_w = 0 the
 * outcome is that of wind_draw() for every u.
 *
 * @param u Uniform number in [0, 1).
 * @param p_star Probability of forming a star, mdot_* dt / m_g.
 * @param p_hot Probability of a hot launch, eta_M,hot p_star.
 * @param p_warm Probability of a warm launch, eta_M,warm p_star.
 */
static inline enum wind_draw4_outcome wind_draw4(const double u,
                                                 const double p_star,
                                                 const double p_hot,
                                                 const double p_warm) {
  const double total = p_star + p_hot + p_warm;
  const double norm = total > 1. ? 1. / total : 1.;
  if (u < p_star * norm) return wind_draw4_star;
  if (u < (p_star + p_hot) * norm) return wind_draw4_hot;
  if (u < total * norm) return wind_draw4_warm;
  return wind_draw4_nothing;
}

/*! Defaults of the loading sets of wind-design.md Sec. 10.3. Warm mass
 * loading 10^norm Sigma_SFR^slope through K26 at |z| = 2 kpc and
 * Sigma_SFR = 3.6e-3 (crmhd 2.8, mhd 1.3), with the K20b cool slope */
#define WIND_WARM_ETA_M_SLOPE -0.44
#define WIND_WARM_ETA_M_LOG10_NORM_K20 -0.961
#define WIND_WARM_ETA_M_LOG10_NORM_TIGRESSPP -0.628
/*! TIGRESSpp hot loadings: the K20b fits times K26 crmhd / mhd at 2 kpc,
 * eta_M 0.17 / 0.33 and eta_E 0.016 / 0.040 */
#define WIND_TIGRESSPP_HOT_ETA_M_FACTOR 0.515
#define WIND_TIGRESSPP_HOT_ETA_E_FACTOR 0.40
/*! TIGRESSpp CR energy loading beyond 1 kpc (K26 Fig. 11) */
#define WIND_TIGRESSPP_CR_ETA_E 0.3

/**
 * @brief Energy per mass of a warm launch, kinetic plus enthalpy,
 * v^2 / 2 + gamma u; its energy loading is eta_M,warm times this over u_*.
 */
static inline double wind_warm_specific_energy(const double v_warm,
                                               const double u_warm,
                                               const double gamma) {
  return 0.5 * v_warm * v_warm + gamma * u_warm;
}

/**
 * @brief CR energy per launched mass, e_CR = eta_E,CR u_* / eta_M,carry.
 *
 * With eta_M,carry the summed mass loading of the phases that carry CR
 * energy, the expected CR energy launched per stellar mass formed is
 * e_CR eta_M,carry = eta_E,CR u_*, whichever phases carry it.
 *
 * @param u_star SN energy per stellar mass.
 * @param eta_E_cr CR energy loading.
 * @param eta_M_carry Summed mass loading of the carrying phases (> 0).
 */
static inline double wind_cr_specific_energy(const double u_star,
                                             const double eta_E_cr,
                                             const double eta_M_carry) {
  return eta_E_cr * u_star / eta_M_carry;
}

/*
 * Warm launch speeds drawn from the K20a cool-phase outflow velocity PDF
 * (wind-design.md Sec. 11). K20a fit the mass-flux PDF per dex of v_out as
 * (v/v0)^2 exp(-v/v0); per unit v this is the Gamma distribution of shape 2,
 * p(v) = v / v0^2 exp(-v/v0), mean 2 v0. The speed at the reference height
 * v_ref is drawn from it; the launch speed adds the potential climb from the
 * midplane to that height, v_L^2 = v_ref^2 + v_climb^2.
 */

/**
 * @brief Launch speed from two uniform numbers.
 *
 * v_ref = -v0 ln((1 - u1)(1 - u2)) is the sum of two exponential numbers of
 * mean v0, which is Gamma(2, v0) distributed; 1 - u lies in (0, 1].
 *
 * @param u1 Uniform number in [0, 1).
 * @param u2 Uniform number in [0, 1), independent of u1.
 * @param v0 Scale speed of the PDF.
 * @param v_climb Speed lost climbing to the reference height.
 */
static inline double wind_warm_pdf_speed(const double u1, const double u2,
                                         const double v0,
                                         const double v_climb) {
  const double v_ref = -v0 * (log1p(-u1) + log1p(-u2));
  return sqrt(v_ref * v_ref + v_climb * v_climb);
}

/**
 * @brief Cumulative distribution of the launch speed, the fraction of the
 * launched mass with v_L < v: 1 - (1 + x) exp(-x), x = (v^2 -
 * v_climb^2)^1/2 / v0; 0 for v <= v_climb.
 */
static inline double wind_warm_pdf_cdf(const double v, const double v0,
                                       const double v_climb) {
  if (v <= v_climb) return 0.;
  const double x = sqrt(v * v - v_climb * v_climb) / v0;
  return -expm1(-x) - x * exp(-x);
}

/**
 * @brief Mean of v_L^2 / 2 over the PDF. E[v_ref^2] = 6 v0^2, so this is
 * 3 v0^2 + v_climb^2 / 2.
 */
static inline double wind_warm_pdf_mean_kinetic(const double v0,
                                                const double v_climb) {
  return 3. * v0 * v0 + 0.5 * v_climb * v_climb;
}

/**
 * @brief Fraction of the launched mass with v_ref > v_extra, i.e. that a
 * ballistic flight carries to the height where it has lost a further speed
 * v_extra beyond the reference height: (1 + v_extra/v0) exp(-v_extra/v0).
 */
static inline double wind_warm_pdf_survival(const double v_extra,
                                            const double v0) {
  if (v_extra <= 0.) return 1.;
  return (1. + v_extra / v0) * exp(-v_extra / v0);
}

/**
 * @brief Has a particle launched along sign * n passed its apex, i.e. has
 * its velocity no positive component along sign * n?
 *
 * @param v The velocity.
 * @param n The launch direction (unit vector).
 * @param sign The sign of the launch, +-1.
 */
static inline int wind_past_apex(const float v[3], const double n[3],
                                 const double sign) {
  return sign * (v[0] * n[0] + v[1] * n[1] + v[2] * n[2]) <= 0.;
}

#endif /* SWIFT_WIND_PHYSICS_H */
