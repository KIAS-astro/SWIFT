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
 * solar units (Z_sun = 0.014, K24 Sec. 2.3) and is clipped below at 0.1, the
 * lowest calibrated value.
 */

/* Some standard headers. */
#include <math.h>

/*! Newton's constant [cm^3 g^-1 s^-2], as src/physical_constants_cgs.h */
#define PRFM_NEWTON_G_CGS 6.67430e-8

/*! Boltzmann's constant [erg K^-1], as src/physical_constants_cgs.h */
#define PRFM_BOLTZMANN_K_CGS 1.380649e-16

/*! Proton mass [g], as src/physical_constants_cgs.h */
#define PRFM_PROTON_MASS_CGS 1.67262192369e-24

/*! Mass per hydrogen atom in units of m_H (J26: n_H = rho / 1.4 m_H) */
#define PRFM_MASS_PER_H 1.4

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
 * sigma_eff = sigma0 x^beta Z'^sigma_Z, x = W/P_0, or max(W/P_0, 1) with
 * the floor,
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

  /*! Hold sigma_eff at its value at P_0 below P_0? */
  int sigma_floor;

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
 * OK22: eqs. 25c (yield) and Sec. 4.6 (dispersion), with the J26 EoS
 * P/k_B = 4.7e4 n_H^1.8 and the J26 floor sigma_eff = 12 km/s below P_0.
 *
 * K24: eqs. 20 (yield) and 23 (dispersion), with no dispersion floor (eq. 23
 * is fitted down to W/k_B = 10^3.1 K cm^-3), and the EoS implied by eq. 23:
 * P = sigma_eff(P)^2 rho gives
 * P = P_0 [rho sigma0^2 Z'^(2 sigma_Z) / P_0]^(1/(1 - 2 beta)),
 * i.e. P/k_B = 3.03e4 n_H^1.316 Z'^0.079, so that the EoS pressure at the
 * equilibrium density W / sigma_eff^2 equals W.
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
    cal->sigma_floor = 0;
    const double P0 = PRFM_P0_OVER_KB * PRFM_BOLTZMANN_K_CGS;
    const double n0 = P0 / (cal->sigma0 * cal->sigma0 * PRFM_MASS_PER_H *
                            PRFM_PROTON_MASS_CGS);
    cal->eos_slope = 1. / (1. - 2. * cal->beta);
    cal->eos_norm = PRFM_P0_OVER_KB * pow(n0, -cal->eos_slope);
    cal->eos_Z = 2. * cal->sigma_Z * cal->eos_slope;
  } else {
    cal->upsilon0 = 1028e5;
    cal->alpha = 0.212;
    cal->upsilon_Z = 0.;
    cal->sigma0 = 12e5;
    cal->beta = 0.22;
    cal->sigma_Z = 0.;
    cal->sigma_floor = 1;
    cal->eos_norm = 4.7e4;
    cal->eos_slope = 1.8;
    cal->eos_Z = 0.;
  }
}

/**
 * @brief The functional form of the effective EoS.
 */
enum prfm_eos_form {
  prfm_eos_J26,      /*!< J26 eq. 15: P/k_B = 4.7e4 n_H^1.8 */
  prfm_eos_OK22_fit, /*!< OK22 eq. 27 (B25 eq. 19): 10^4.30 n_H^1.43 */
  prfm_eos_K24       /*!< implied by K24 eq. 23: 3.03e4 n_H^1.316 Z'^0.079 */
};

/**
 * @brief Name of an effective EoS form, as written in the parameter file.
 */
static inline const char *prfm_eos_form_name(const enum prfm_eos_form form) {
  switch (form) {
    case prfm_eos_J26:
      return "J26";
    case prfm_eos_OK22_fit:
      return "OK22_fit";
    case prfm_eos_K24:
      return "K24";
  }
  return "unknown";
}

/**
 * @brief Set the effective EoS of a calibration to a given form.
 *
 * J26: eq. 15, the OK22 sigma_eff-P fit written as an EoS (H24 eq. 35).
 * OK22_fit: the direct fit to the TIGRESS midplane two-phase gas,
 * log P/k_B = 1.43 log n_H + 4.30 (OK22 eq. 27, B25 eq. 19). B25 defines
 * n_H = X_H rho / m_p; here n_H = rho / (1.4 m_H) for every form, 6 per cent
 * lower at X_H = 0.76.
 * K24: the form prfm_calibration_init gives the K24 calibration.
 *
 * @param form Which form.
 * @param cal (return) The calibration whose eos_* members are set.
 */
static inline void prfm_eos_form_init(const enum prfm_eos_form form,
                                      struct prfm_calibration *cal) {
  if (form == prfm_eos_K24) {
    struct prfm_calibration k24;
    prfm_calibration_init(prfm_calibration_K24, &k24);
    cal->eos_norm = k24.eos_norm;
    cal->eos_slope = k24.eos_slope;
    cal->eos_Z = k24.eos_Z;
  } else if (form == prfm_eos_OK22_fit) {
    cal->eos_norm = pow(10., 4.30);
    cal->eos_slope = 1.43;
    cal->eos_Z = 0.;
  } else {
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
 * held at its P_0 value below P_0 if the calibration has the floor (OK22).
 */
static inline double prfm_sigma_eff(const double W,
                                    const struct prfm_calibration *cal,
                                    const double Zprime) {
  double x = W / (PRFM_P0_OVER_KB * PRFM_BOLTZMANN_K_CGS);
  if (cal->sigma_floor && x < 1.) x = 1.;
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

/**
 * @brief Column of the cubic spline (M4) kernel of compact support H through
 * a line at height z above its centre, per unit density:
 * w(z; H) = 2 pi int_|z|^H W(r) r dr, W(r) = 8 / (pi H^3) k(r / H), with
 * k(q) = 1 - 6 q^2 + 6 q^3 (q < 1/2), 2 (1 - q)^3 (1/2 <= q < 1).
 * w(0; H) = 1.4 / H.
 *
 * @param z Height above the kernel centre (any length unit).
 * @param H Kernel support, same unit.
 * @return w(z; H) in inverse length units.
 */
static inline double prfm_kernel_projected(const double z, const double H) {
  const double q = fabs(z) / H;
  if (q >= 1.) return 0.;
  if (q < 0.5) {
    const double q2 = q * q;
    return 16. / H * (0.0875 - 0.5 * q2 + 1.5 * q2 * q2 - 1.2 * q2 * q2 * q);
  }
  const double y = 1. - q;
  const double y4 = y * y * y * y;
  return 16. / H * (0.5 * y4 - 0.4 * y4 * y);
}

/**
 * @brief Kernel-column bias f(x) of a sech^2 layer: the estimate
 * rho_SPH(0) / w(0; H_k) of a layer rho_0 sech^2(z / H), Sigma = 2 rho_0 H,
 * is f(H_k / H) Sigma, with
 * f(x) = int_0^inf sech^2(u) w(u; x) du / w(0; x) (lengths in units of H).
 * Simpson's rule with 128 intervals over u in [0, min(x, 20)]
 * (sech^2(20) = 2e-17).
 *
 * @param x Kernel support over layer half-thickness, H_k / H > 0.
 */
static inline double prfm_kernel_column_bias(const double x) {
  const int n = 128;
  const double umax = x < 20. ? x : 20.;
  const double du = umax / n;
  double sum = 0.;
  for (int i = 0; i <= n; i++) {
    const double u = i * du;
    const double c = cosh(u);
    const double wgt = (i == 0 || i == n) ? 1. : (i % 2 ? 4. : 2.);
    sum += wgt * prfm_kernel_projected(u, x) / (c * c);
  }
  return sum * du / 3. / prfm_kernel_projected(0., x);
}

/**
 * @brief Residual ln Sigma_g - ln Sigma_est + ln f(H_k / H_g(Sigma_g)) of the
 * equilibrium correction, with H_k / H_g held at >= 1. Fills @p out.
 */
static inline double prfm_kernel_column_residual(
    const double lnSigma, const double lnSigma_est, const double H_k,
    const double Sigma_s, const double rho_d, const double sigma_sz,
    const double c_d, const struct prfm_calibration *cal, const double Zprime,
    const int n_iter, struct prfm_state *out) {
  prfm_int(exp(lnSigma), Sigma_s, rho_d, sigma_sz, c_d, cal, Zprime, n_iter,
           out);
  const double x = H_k / out->H_g;
  return lnSigma - lnSigma_est + log(prfm_kernel_column_bias(x > 1. ? x : 1.));
}

/**
 * @brief Gas surface density from the kernel-column estimate, corrected for
 * the thickness of the PRFM-int equilibrium layer.
 *
 * Solves Sigma_g = Sigma_est / f(H_k / H_g(Sigma_g)), with H_g from
 * prfm_int() and f from prfm_kernel_column_bias(). H_k / H_g is held at >= 1
 * (f >= f(1) = 0.33): below that the kernel lies inside the layer and the
 * estimate is a volume density, which the correction does not describe. The
 * residual of prfm_kernel_column_residual() increases with Sigma_g, is < 0 at
 * Sigma_est and >= 0 at Sigma_est / f(1), so the root is bracketed; it is
 * found in ln Sigma_g by regula falsi (Illinois), to 1e-7 in the residual.
 *
 * @param Sigma_est Uncorrected estimate rho_SPH / w(0; H_k) [g cm^-2].
 * @param H_k Kernel support [cm].
 * @param n_corr Largest number of regula falsi steps.
 * Other parameters and @p out as prfm_int(); @p out is the state at the
 * returned Sigma_g.
 * @return The corrected Sigma_g [g cm^-2].
 */
static inline double prfm_kernel_column_equilibrium(
    const double Sigma_est, const double H_k, const double Sigma_s,
    const double rho_d, const double sigma_sz, const double c_d,
    const struct prfm_calibration *cal, const double Zprime, const int n_iter,
    const int n_corr, struct prfm_state *out) {
  const double l0 = log(Sigma_est);
  double a = l0;
  double ga = prfm_kernel_column_residual(a, l0, H_k, Sigma_s, rho_d, sigma_sz,
                                          c_d, cal, Zprime, n_iter, out);
  double b = l0 - log(prfm_kernel_column_bias(1.));
  double gb = prfm_kernel_column_residual(b, l0, H_k, Sigma_s, rho_d, sigma_sz,
                                          c_d, cal, Zprime, n_iter, out);
  if (gb <= 0.) return exp(b);
  for (int i = 0; i < n_corr; i++) {
    const double c = (a * gb - b * ga) / (gb - ga);
    const double gc = prfm_kernel_column_residual(
        c, l0, H_k, Sigma_s, rho_d, sigma_sz, c_d, cal, Zprime, n_iter, out);
    if (gc * gb < 0.) {
      a = b;
      ga = gb;
    } else {
      ga *= 0.5;
    }
    b = c;
    gb = gc;
    if (fabs(gc) < 1e-7) break;
  }
  return exp(b);
}

#endif /* SWIFT_PRFM_PHYSICS_H */
