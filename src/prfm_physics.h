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
#ifndef SWIFT_PRFM_PHYSICS_H
#define SWIFT_PRFM_PHYSICS_H

/**
 * @file src/prfm_physics.h
 * @brief Pressure-regulated, feedback-modulated (PRFM) star formation:
 * the pure functions of the model, shared by the PRFM star formation module,
 * the PRFM entropy floor and the unit test tests/testPRFM.c.
 *
 * PRFM-int (Jeffreson et al. 2026, J26, Sec. 2.6, eqs. 28-32) from surface
 * densities, with the TIGRESS calibrations of Ostriker & Kim (2022, OK22,
 * default) and Kim et al. (2024, K24).
 *
 * Units: everything is cgs. Surface densities in g cm^-2, volume densities in
 * g cm^-3, velocities in cm s^-1, pressures and weights in erg cm^-3 (the
 * calibrations are written in P/k_B [K cm^-3], converted here with
 * PRFM_BOLTZMANN_K_CGS), lengths in cm, times in s. Metallicity Z' is in
 * solar units and is clipped below at 0.1, the lowest calibrated value.
 */

/* Some standard headers. */
#include <math.h>

/*! Newton's constant [cm^3 g^-1 s^-2], as src/physical_constants_cgs.h */
#define PRFM_NEWTON_G_CGS 6.67430e-8

/*! Boltzmann's constant [erg K^-1], as src/physical_constants_cgs.h */
#define PRFM_BOLTZMANN_K_CGS 1.380649e-16

/*! Reference pressure P_0 / k_B of the calibrations [K cm^-3] */
#define PRFM_P0_OVER_KB 1e4

/*! Lowest metallicity Z' of the K24 calibration */
#define PRFM_ZPRIME_MIN 0.1

/*! Initial sigma_eff of the fixed-point iteration [cm s^-1] (H24 p. 16) */
#define PRFM_SIGMA_START_CGS 1e6

/**
 * @brief The calibration of yield, dispersion and effective EoS.
 *
 * Upsilon_tot = upsilon0 (W/P_0)^-alpha Z'^upsilon_Z,
 * sigma_eff = sigma0 max(W/P_0, 1)^beta Z'^sigma_Z,
 * P_eff/k_B = eos_norm n_H^eos_slope Z'^eos_Z, with n_H in cm^-3.
 */
struct prfm_calibration {

  /*! Yield at P_0 [cm s^-1] */
  double upsilon0;

  /*! Slope of the yield with weight */
  double alpha;

  /*! Metallicity exponent of the yield */
  double upsilon_Z;

  /*! Velocity dispersion at and below P_0 [cm s^-1] */
  double sigma0;

  /*! Slope of the dispersion with weight */
  double beta;

  /*! Metallicity exponent of the dispersion */
  double sigma_Z;

  /*! Effective EoS P/k_B at n_H = 1 cm^-3 [K cm^-3] */
  double eos_norm;

  /*! Slope of the effective EoS with n_H */
  double eos_slope;

  /*! Metallicity exponent of the effective EoS */
  double eos_Z;
};

/**
 * @brief The choice of calibration.
 */
enum prfm_calibration_choice {
  prfm_calibration_OK22, /*!< Ostriker & Kim (2022), solar metallicity */
  prfm_calibration_K24   /*!< Kim et al. (2024), metallicity dependent */
};

/**
 * @brief Results of the PRFM-int closure.
 */
struct prfm_state {

  /*! Weight W [erg cm^-3] */
  double W;

  /*! Effective velocity dispersion [cm s^-1] */
  double sigma_eff;

  /*! Gas scale height H_g = Sigma_g sigma_eff^2 / (2 W) [cm] */
  double H_g;

  /*! Dynamical time t_dyn = 2 H_g / sigma_eff [s] */
  double t_dyn;

  /*! Depletion time t_dep = Upsilon_tot Sigma_g / W [s] */
  double t_dep;
};

/**
 * @brief Fill a calibration.
 *
 * OK22 eqs. 25c (yield) and 22 (dispersion) with the J26 EoS
 * P/k_B = 4.7e4 n_H^1.8; K24 eqs. for Upsilon, sigma_eff and P_eff with Z'.
 *
 * @param choice Which calibration.
 * @param cal (return) The calibration.
 */
static inline void prfm_calibration_init(enum prfm_calibration_choice choice,
                                         struct prfm_calibration *cal) {
  if (choice == prfm_calibration_K24) {
    cal->upsilon0 = 1650e5;
    cal->alpha = 0.29;
    cal->upsilon_Z = -0.27;
    cal->sigma0 = 11.7e5;
    cal->beta = 0.12;
    cal->sigma_Z = 0.03;
    cal->eos_norm = 2.85e4;
    cal->eos_slope = 1.29;
    cal->eos_Z = -0.10;
  } else {
    cal->upsilon0 = 1028e5;
    cal->alpha = 0.212;
    cal->upsilon_Z = 0.;
    cal->sigma0 = 12e5;
    cal->beta = 0.22;
    cal->sigma_Z = 0.;
    cal->eos_norm = 4.7e4;
    cal->eos_slope = 1.8;
    cal->eos_Z = 0.;
  }
}

/**
 * @brief Metallicity Z' clipped at the lowest calibrated value.
 */
static inline double prfm_clip_Zprime(const double Zprime) {
  return Zprime > PRFM_ZPRIME_MIN ? Zprime : PRFM_ZPRIME_MIN;
}

/**
 * @brief Feedback yield Upsilon_tot [cm s^-1] at weight W [erg cm^-3].
 */
static inline double prfm_upsilon(const double W,
                                  const struct prfm_calibration *cal,
                                  const double Zprime) {
  const double x = W / (PRFM_P0_OVER_KB * PRFM_BOLTZMANN_K_CGS);
  return cal->upsilon0 * pow(x, -cal->alpha) *
         pow(prfm_clip_Zprime(Zprime), cal->upsilon_Z);
}

/**
 * @brief Effective velocity dispersion [cm s^-1] at weight W [erg cm^-3],
 * held at sigma0 (12 km/s for OK22) below P_0.
 */
static inline double prfm_sigma_eff(const double W,
                                    const struct prfm_calibration *cal,
                                    const double Zprime) {
  double x = W / (PRFM_P0_OVER_KB * PRFM_BOLTZMANN_K_CGS);
  if (x < 1.) x = 1.;
  return cal->sigma0 * pow(x, cal->beta) *
         pow(prfm_clip_Zprime(Zprime), cal->sigma_Z);
}

/**
 * @brief Effective EoS pressure [erg cm^-3] at hydrogen number density n_H
 * [cm^-3].
 */
static inline double prfm_eos_pressure(const double n_H,
                                       const struct prfm_calibration *cal,
                                       const double Zprime) {
  return cal->eos_norm * pow(n_H, cal->eos_slope) *
         pow(prfm_clip_Zprime(Zprime), cal->eos_Z) * PRFM_BOLTZMANN_K_CGS;
}

/**
 * @brief Weight of the gas layer, J26 eq. 31 [erg cm^-3].
 *
 * Sigma_eff = Sigma_g + 2 Sigma_* / (1 + sigma_*z / sigma_eff),
 * W = pi G Sigma_g Sigma_eff / 4
 *     {1 + [1 + c_d G rho_d sigma_eff^2 / (pi G Sigma_eff)^2]^(1/2)}.
 *
 * @param Sigma_g Gas surface density [g cm^-2].
 * @param Sigma_s Stellar surface density [g cm^-2].
 * @param rho_d Dark matter density [g cm^-3].
 * @param sigma_eff Gas effective velocity dispersion [cm s^-1].
 * @param sigma_sz Stellar vertical velocity dispersion [cm s^-1].
 * @param c_d Dark matter coefficient (16 pi / 3 for an NFW core, J26).
 */
static inline double prfm_weight(const double Sigma_g, const double Sigma_s,
                                 const double rho_d, const double sigma_eff,
                                 const double sigma_sz, const double c_d) {
  const double G = PRFM_NEWTON_G_CGS;
  const double Sigma_eff = Sigma_g + 2. * Sigma_s / (1. + sigma_sz / sigma_eff);
  const double piGS = M_PI * G * Sigma_eff;
  const double x = c_d * G * rho_d * sigma_eff * sigma_eff / (piGS * piGS);
  return 0.25 * M_PI * G * Sigma_g * Sigma_eff * (1. + sqrt(1. + x));
}

/**
 * @brief The PRFM-int closure: weight, dispersion, scale height, dynamical
 * and depletion times of a gas layer (design note Sec. 2, steps 2-4).
 *
 * sigma_eff starts at 10 km/s and is iterated n_iter times through
 * W(sigma_eff) and sigma_eff(W); W is then evaluated once more at the final
 * sigma_eff. t_dep = Upsilon_tot Sigma_g / W, i.e. R_f = 1.
 *
 * @param Sigma_g Gas surface density [g cm^-2], > 0.
 * @param Sigma_s Stellar surface density [g cm^-2].
 * @param rho_d Dark matter density [g cm^-3].
 * @param sigma_sz Stellar vertical dispersion [cm s^-1]; a negative value
 * means sigma_*z = sigma_eff at each iteration.
 * @param c_d Dark matter coefficient.
 * @param cal The calibration.
 * @param Zprime Metallicity in solar units.
 * @param n_iter Number of fixed-point iterations (5 in J26 and H24).
 * @param out (return) The state.
 */
static inline void prfm_int(const double Sigma_g, const double Sigma_s,
                            const double rho_d, const double sigma_sz,
                            const double c_d,
                            const struct prfm_calibration *cal,
                            const double Zprime, const int n_iter,
                            struct prfm_state *out) {
  double sigma = PRFM_SIGMA_START_CGS;
  for (int i = 0; i < n_iter; i++) {
    const double W = prfm_weight(Sigma_g, Sigma_s, rho_d, sigma,
                                 sigma_sz < 0. ? sigma : sigma_sz, c_d);
    sigma = prfm_sigma_eff(W, cal, Zprime);
  }
  const double W = prfm_weight(Sigma_g, Sigma_s, rho_d, sigma,
                               sigma_sz < 0. ? sigma : sigma_sz, c_d);
  out->W = W;
  out->sigma_eff = sigma;
  out->H_g = Sigma_g * sigma * sigma / (2. * W);
  out->t_dyn = 2. * out->H_g / sigma;
  out->t_dep = prfm_upsilon(W, cal, Zprime) * Sigma_g / W;
}

#endif /* SWIFT_PRFM_PHYSICS_H */
