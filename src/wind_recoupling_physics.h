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
#ifndef SWIFT_WIND_RECOUPLING_PHYSICS_H
#define SWIFT_WIND_RECOUPLING_PHYSICS_H

/**
 * @file src/wind_recoupling_physics.h
 * @brief Decoupled wind, experiment A (ai-notes wind-design.md Sec. 9): the
 * pure functions shared by the TIGRESS wind module and the unit test
 * tests/testWindRecoupling.c.
 *
 * 1. Loadings at a height z: the K20b hot-phase fits at |z| = 500 pc and
 *    1 kpc (least-squares fits to the seven model means of the K20b Zenodo
 *    table), log10 eta interpolated linearly in log10 z between them and
 *    clamped outside; the factor eta(z) / eta(H) rescales any loading.
 * 2. The Chevalier & Clegg (1985) wind of a sphere of radius R with mass and
 *    energy injection rates Mdot and Edot = Mdot v_inf^2 / 2 (gamma = 5/3,
 *    adiabatic, no gravity): Mach number M(x), x = r / R, and pressure
 *    P(r) = Mdot v_inf / (4 pi R^2) Pi(x), Pi = g(x) w(M) / (gamma M^2),
 *    g = x (x < 1) or x^-2 (x >= 1), w = M / (M^2 + 2 / (gamma - 1))^1/2.
 * 3. The recoupling trigger: distance, time, or pressure balance.
 */

/* Some standard headers. */
#include <math.h>

/*! Adiabatic index of the CC85 solution */
#define WIND_CC85_GAMMA (5. / 3.)

/*! Heights of the fixed-height fits [kpc] */
#define WIND_K20_Z_LOW 0.5
#define WIND_K20_Z_HIGH 1.0

/*! Hot-phase fits at |z| = 500 pc: log10 eta = norm + slope log10 Sigma_SFR */
#define WIND_K20_Z500_ETA_M_LOG10_NORM -0.984
#define WIND_K20_Z500_ETA_M_SLOPE -0.145
#define WIND_K20_Z500_ETA_E_LOG10_NORM -0.953
#define WIND_K20_Z500_ETA_E_SLOPE -0.006

/*! Hot-phase fits at |z| = 1 kpc */
#define WIND_K20_Z1000_ETA_M_LOG10_NORM -1.185
#define WIND_K20_Z1000_ETA_M_SLOPE -0.100
#define WIND_K20_Z1000_ETA_E_LOG10_NORM -1.381
#define WIND_K20_Z1000_ETA_E_SLOPE 0.013

/*! Hot-phase fits at |z| = H (K20b Table 5) */
#define WIND_K20_H_ETA_M_LOG10_NORM -0.86
#define WIND_K20_H_ETA_M_SLOPE -0.07
#define WIND_K20_H_ETA_E_LOG10_NORM -0.70
#define WIND_K20_H_ETA_E_SLOPE 0.14

/**
 * @brief Hot-phase loadings of the fixed-height fits at height z.
 *
 * @param z_kpc Height [kpc]; the 500 pc fit below 0.5 kpc, the 1 kpc fit
 * above 1 kpc, log-log interpolation in between.
 * @param Sigma_SFR [Msun kpc^-2 yr^-1], clipped to [1e-4, 1].
 * @param eta_M (return) Mass loading.
 * @param eta_E (return) Energy loading.
 */
static inline void wind_K20_loadings_at_height(const double z_kpc,
                                               const double Sigma_SFR,
                                               double *eta_M, double *eta_E) {
  const double ls = log10(fmin(fmax(Sigma_SFR, 1e-4), 1.));
  const double lm0 =
      WIND_K20_Z500_ETA_M_LOG10_NORM + WIND_K20_Z500_ETA_M_SLOPE * ls;
  const double le0 =
      WIND_K20_Z500_ETA_E_LOG10_NORM + WIND_K20_Z500_ETA_E_SLOPE * ls;
  const double lm1 =
      WIND_K20_Z1000_ETA_M_LOG10_NORM + WIND_K20_Z1000_ETA_M_SLOPE * ls;
  const double le1 =
      WIND_K20_Z1000_ETA_E_LOG10_NORM + WIND_K20_Z1000_ETA_E_SLOPE * ls;
  double s = (log10(fmax(z_kpc, 1e-30)) - log10(WIND_K20_Z_LOW)) /
             (log10(WIND_K20_Z_HIGH) - log10(WIND_K20_Z_LOW));
  s = fmin(fmax(s, 0.), 1.);
  *eta_M = pow(10., lm0 + s * (lm1 - lm0));
  *eta_E = pow(10., le0 + s * (le1 - le0));
}

/**
 * @brief Factors f_M = eta_M(z) / eta_M(H) and f_E = eta_E(z) / eta_E(H).
 *
 * @param z_kpc Height [kpc].
 * @param Sigma_SFR [Msun kpc^-2 yr^-1], clipped to [1e-4, 1].
 * @param f_M (return) Mass loading factor.
 * @param f_E (return) Energy loading factor.
 */
static inline void wind_K20_height_factors(const double z_kpc,
                                           const double Sigma_SFR,
                                           double *f_M, double *f_E) {
  const double ls = log10(fmin(fmax(Sigma_SFR, 1e-4), 1.));
  double eta_M, eta_E;
  wind_K20_loadings_at_height(z_kpc, Sigma_SFR, &eta_M, &eta_E);
  *f_M = eta_M /
         pow(10., WIND_K20_H_ETA_M_LOG10_NORM + WIND_K20_H_ETA_M_SLOPE * ls);
  *f_E = eta_E /
         pow(10., WIND_K20_H_ETA_E_LOG10_NORM + WIND_K20_H_ETA_E_SLOPE * ls);
}

/**
 * @brief Left side of the CC85 Mach-number relation minus its right side,
 * as a function of ln M, on the branch of x.
 *
 * x < 1: ln{[(3g + M^-2)/(1 + 3g)]^(-(3g+1)/(5g+1))
 *            [(g - 1 + 2 M^-2)/(1 + g)]^((g+1)/(2(5g+1)))} - ln x;
 * x >= 1: (2/(g-1)) ln M + ((g+1)/(2(g-1))) ln[(g - 1 + 2 M^-2)/(1 + g)]
 *          - 2 ln x.
 */
static inline double wind_cc85_residual(const double lnM, const double x) {
  const double g = WIND_CC85_GAMMA;
  const double im2 = exp(-2. * lnM);
  if (x < 1.)
    return -(3. * g + 1.) / (5. * g + 1.) *
               log((3. * g + im2) / (1. + 3. * g)) +
           (g + 1.) / (2. * (5. * g + 1.)) * log((g - 1. + 2. * im2) / (1. + g)) -
           log(x);
  return 2. / (g - 1.) * lnM +
         (g + 1.) / (2. * (g - 1.)) * log((g - 1. + 2. * im2) / (1. + g)) -
         2. * log(x);
}

/**
 * @brief CC85 Mach number at x = r / R (subsonic inside, supersonic
 * outside), by bisection in ln M.
 *
 * On each branch the residual increases with M (inside, M is proportional
 * to x for small x; outside, M grows as x^(gamma - 1)).
 */
static inline double wind_cc85_mach(const double x) {
  if (x == 1.) return 1.;
  double lo, hi;
  if (x < 1.) {
    lo = -60.;
    hi = 0.;
  } else {
    lo = 0.;
    hi = 60.;
  }
  for (int i = 0; i < 200; i++) {
    const double mid = 0.5 * (lo + hi);
    const double f = wind_cc85_residual(mid, x);
    if (f < 0.)
      lo = mid;
    else
      hi = mid;
    if (hi - lo < 1e-14) break;
  }
  return exp(0.5 * (lo + hi));
}

/**
 * @brief Dimensionless CC85 pressure Pi(x) = P 4 pi R^2 / (Mdot v_inf).
 */
static inline double wind_cc85_pressure_dimensionless(const double x) {
  const double g = WIND_CC85_GAMMA;
  const double M = wind_cc85_mach(x);
  const double w = M / sqrt(M * M + 2. / (g - 1.));
  const double gx = x < 1. ? x : 1. / (x * x);
  return gx * w / (g * M * M);
}

/**
 * @brief CC85 pressure at a distance r from the centre of the source.
 *
 * @param r Distance (same length unit as R).
 * @param R Source radius.
 * @param Mdot Mass injection rate.
 * @param v_inf Asymptotic speed (2 Edot / Mdot)^1/2.
 * @return P in units of Mdot v_inf / R^2.
 */
static inline double wind_cc85_pressure(const double r, const double R,
                                        const double Mdot,
                                        const double v_inf) {
  if (R <= 0. || Mdot <= 0.) return 0.;
  return Mdot * v_inf / (4. * M_PI * R * R) *
         wind_cc85_pressure_dimensionless(r / R);
}

/**
 * @brief Recoupling criterion of a decoupled particle.
 */
enum wind_recoupling_mode {
  wind_recoupling_by_distance = 0, /*<! at d_rec or t_rec (WD5) */
  wind_recoupling_by_pressure = 1  /*<! at CC85 pressure balance (Sec. 9.2) */
};

/**
 * @brief Why a particle recoupled (0: it did not). The values follow the
 * density recoupling of branch arkenstone-tests, where 2 is density.
 */
enum wind_recoupling_reason {
  wind_reason_none = 0,
  wind_reason_distance = 1,
  wind_reason_time = 3,
  wind_reason_pressure = 4
};

/**
 * @brief The recoupling trigger.
 *
 * distance mode: d >= d_max or t >= t_rec.
 * pressure mode: d >= d_min and P_amb > 0 and P_w <= P_amb, or the two
 * backstops; pressure is tested first.
 *
 * @param mode The recoupling mode.
 * @param d Distance from the launch point.
 * @param t Time since the launch.
 * @param d_min Minimum distance for pressure recoupling.
 * @param d_max Recoupling distance (backstop in pressure mode).
 * @param t_rec Recoupling time.
 * @param P_w Wind pressure at d.
 * @param P_amb Ambient pressure (0: unknown).
 */
static inline enum wind_recoupling_reason wind_recoupling_trigger(
    const enum wind_recoupling_mode mode, const double d, const double t,
    const double d_min, const double d_max, const double t_rec,
    const double P_w, const double P_amb) {
  if (mode == wind_recoupling_by_pressure && d >= d_min && P_amb > 0. &&
      P_w <= P_amb)
    return wind_reason_pressure;
  if (d >= d_max) return wind_reason_distance;
  if (t >= t_rec) return wind_reason_time;
  return wind_reason_none;
}

#endif /* SWIFT_WIND_RECOUPLING_PHYSICS_H */
