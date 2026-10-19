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

/*
 * Checks src/wind_recoupling_physics.h (wind-design.md Sec. 9.4):
 *  - CC85: M(x) satisfies its implicit relation to 1e-10 at 41 points
 *    1e-3 <= x <= 1e3; M(1) = 1; M(1 -+ 1e-9) = 1 to 1e-3;
 *    subsonic inside, supersonic outside; Pi continuous across x = 1
 *    (1e-5 at x = 1 -+ 1e-12, since M - 1 grows as |x - 1|^1/2); Pi(x) against the asymptote
 *    0.0945 x^-10/3 (1e-3 at x = 1e3) and against the leading form
 *    x^-2 w(M_a) / (gamma M_a^2) with the asymptotic Mach number M_a
 *    (2e-3 at x = 100); Pi decreasing for x > 1; the inner limit Pi(x -> 0)
 *    finite; the dimensional pressure at one hand-computed point (1e-12);
 *  - loadings at a height: the 500 pc and 1 kpc fits reproduced (1e-12),
 *    the log-midpoint z = 0.5 sqrt(2) kpc gives the geometric mean (1e-12),
 *    clamping below 0.5 and above 1 kpc, and the factors at
 *    Sigma_SFR = 1e-2, z = 2 kpc against the fits by hand (1e-12) and the
 *    numbers of wind-design.md Sec. 9.1 (0.104 / 0.19 and 0.039 / 0.105,
 *    to the two digits given, 2e-2);
 *  - the recoupling trigger: a truth table.
 */

#include <config.h>

/* Local includes. */
#include "wind_recoupling_physics.h"

/* System includes. */
#include <math.h>
#include <stdio.h>

static int n_fail = 0;

static void check(const char *name, const double measured,
                  const double expected, const double tol) {
  const double rel = expected != 0. ? fabs(measured / expected - 1.)
                                    : fabs(measured - expected);
  const int ok = rel <= tol;
  printf("%s %-60s measured %.12g expected %.12g rel %.2e (tol %.0e)\n",
         ok ? "ok  " : "FAIL", name, measured, expected, rel, tol);
  if (!ok) n_fail++;
}

static void check_int(const char *name, const int measured,
                      const int expected) {
  const int ok = measured == expected;
  printf("%s %-60s measured %d expected %d\n", ok ? "ok  " : "FAIL", name,
         measured, expected);
  if (!ok) n_fail++;
}

/* The implicit CC85 relations evaluated directly (not through the residual
 * function of the header) */
static double cc85_lhs(const double M, const double x) {
  const double g = WIND_CC85_GAMMA;
  if (x < 1.)
    return pow((3. * g + 1. / (M * M)) / (1. + 3. * g),
               -(3. * g + 1.) / (5. * g + 1.)) *
           pow((g - 1. + 2. / (M * M)) / (1. + g),
               (g + 1.) / (2. * (5. * g + 1.)));
  return pow(M, 2. / (g - 1.)) *
         pow((g - 1. + 2. / (M * M)) / (1. + g), (g + 1.) / (2. * (g - 1.)));
}

int main(void) {

  const double g = WIND_CC85_GAMMA;

  /* --- CC85 Mach number --- */
  double worst = 0.;
  int sub_ok = 1, sup_ok = 1;
  for (int i = 0; i <= 40; i++) {
    const double x = pow(10., -3. + 6. * i / 40.);
    const double M = wind_cc85_mach(x);
    const double target = x < 1. ? x : x * x;
    const double rel = fabs(cc85_lhs(M, x) / target - 1.);
    if (rel > worst) worst = rel;
    if (x < 1. && M >= 1.) sub_ok = 0;
    if (x > 1. && M <= 1.) sup_ok = 0;
  }
  check("CC85: max |relation(M(x)) / target - 1|, 1e-3 <= x <= 1e3", worst,
        0., 1e-10);
  check_int("CC85: subsonic for x < 1", sub_ok, 1);
  check_int("CC85: supersonic for x > 1", sup_ok, 1);
  check("CC85: M(1)", wind_cc85_mach(1.), 1., 0.);
  check("CC85: M(1 - 1e-9)", wind_cc85_mach(1. - 1e-9), 1., 1e-3);
  check("CC85: M(1 + 1e-9)", wind_cc85_mach(1. + 1e-9), 1., 1e-3);

  /* --- CC85 pressure --- */
  const double coef = 1. / (g * pow((g + 1.) / (g - 1.), (g + 1.) / 2.));
  check("CC85: asymptotic coefficient 1/(gamma 4^(4/3))", coef, 0.0945, 1e-3);
  check("CC85: Pi(1e3) against 0.0945 x^-10/3",
        wind_cc85_pressure_dimensionless(1e3), coef * pow(1e3, -10. / 3.),
        1e-3);
  {
    const double x = 100.;
    const double Ma = pow((g + 1.) / (g - 1.), (g + 1.) / 4.) * pow(x, g - 1.);
    const double w = Ma / sqrt(Ma * Ma + 2. / (g - 1.));
    check("CC85: Pi(100) against x^-2 w(M_a) / (gamma M_a^2)",
          wind_cc85_pressure_dimensionless(x), w / (g * Ma * Ma * x * x),
          2e-3);
  }
  int mono = 1;
  double prev = wind_cc85_pressure_dimensionless(1.);
  for (int i = 1; i <= 60; i++) {
    const double x = pow(10., 3. * i / 60.);
    const double P = wind_cc85_pressure_dimensionless(x);
    if (P >= prev) mono = 0;
    prev = P;
  }
  check_int("CC85: Pi decreasing for 1 < x <= 1e3", mono, 1);
  {
    const double p1 = wind_cc85_pressure_dimensionless(1e-4);
    const double p2 = wind_cc85_pressure_dimensionless(1e-3);
    check("CC85: Pi(1e-4) / Pi(1e-3) (finite inner limit)", p1 / p2, 1.,
          1e-5);
    /* M - 1 grows as |x - 1|^1/2 at the sonic point */
    check("CC85: Pi continuous across x = 1 (x = 1 -+ 1e-12)",
          wind_cc85_pressure_dimensionless(1. - 1e-12),
          wind_cc85_pressure_dimensionless(1. + 1e-12), 1e-5);
  }
  {
    const double R = 3., Mdot = 2., v_inf = 5., r = 7.5;
    check("CC85: P(r) = Mdot v_inf / (4 pi R^2) Pi(r / R)",
          wind_cc85_pressure(r, R, Mdot, v_inf),
          Mdot * v_inf / (4. * M_PI * R * R) *
              wind_cc85_pressure_dimensionless(r / R),
          1e-12);
    check("CC85: P = 0 for Mdot = 0", wind_cc85_pressure(r, R, 0., v_inf), 0.,
          0.);
  }

  /* --- loadings at a height --- */
  const double S = 1e-2, ls = log10(S);
  double eM, eE, eM2, eE2;
  wind_K20_loadings_at_height(0.5, S, &eM, &eE);
  check("height: eta_M(500 pc)", eM, pow(10., -0.984 - 0.145 * ls), 1e-12);
  check("height: eta_E(500 pc)", eE, pow(10., -0.953 - 0.006 * ls), 1e-12);
  wind_K20_loadings_at_height(1.0, S, &eM, &eE);
  check("height: eta_M(1 kpc)", eM, pow(10., -1.185 - 0.100 * ls), 1e-12);
  check("height: eta_E(1 kpc)", eE, pow(10., -1.381 + 0.013 * ls), 1e-12);
  wind_K20_loadings_at_height(0.5 * sqrt(2.), S, &eM, &eE);
  check("height: eta_M(log-midpoint) = geometric mean", eM,
        sqrt(pow(10., -0.984 - 0.145 * ls) * pow(10., -1.185 - 0.100 * ls)),
        1e-12);
  check("height: eta_E(log-midpoint) = geometric mean", eE,
        sqrt(pow(10., -0.953 - 0.006 * ls) * pow(10., -1.381 + 0.013 * ls)),
        1e-12);
  wind_K20_loadings_at_height(0.1, S, &eM, &eE);
  wind_K20_loadings_at_height(0.5, S, &eM2, &eE2);
  check("height: clamped below 500 pc (eta_M)", eM, eM2, 1e-15);
  wind_K20_loadings_at_height(20., S, &eM, &eE);
  wind_K20_loadings_at_height(1., S, &eM2, &eE2);
  check("height: clamped above 1 kpc (eta_E)", eE, eE2, 1e-15);
  wind_K20_loadings_at_height(1., 1e-6, &eM, &eE);
  wind_K20_loadings_at_height(1., 1e-4, &eM2, &eE2);
  check("height: Sigma_SFR clipped at 1e-4", eM, eM2, 1e-15);
  double fM, fE;
  wind_K20_height_factors(2., S, &fM, &fE);
  check("height factor f_M(2 kpc, 1e-2)", fM,
        pow(10., -1.185 - 0.100 * ls) / pow(10., -0.86 - 0.07 * ls), 1e-12);
  check("height factor f_E(2 kpc, 1e-2)", fE,
        pow(10., -1.381 + 0.013 * ls) / pow(10., -0.70 + 0.14 * ls), 1e-12);
  check("height factor f_M against Sec. 9.1 (0.104 / 0.19)", fM, 0.104 / 0.19,
        2e-2);
  check("height factor f_E against Sec. 9.1 (0.039 / 0.105)", fE,
        0.039 / 0.105, 2e-2);

  /* --- trigger --- */
  const enum wind_recoupling_mode D = wind_recoupling_by_distance;
  const enum wind_recoupling_mode P = wind_recoupling_by_pressure;
  /* arguments: mode, d, t, d_min, d_max, t_rec, P_w, P_amb */
  check_int("trigger: distance mode, short of d_max and t_rec",
            wind_recoupling_trigger(D, 1., 1., 0.5, 2., 10., 1., 5.),
            wind_reason_none);
  check_int("trigger: distance mode ignores pressure",
            wind_recoupling_trigger(D, 1.5, 1., 0.5, 2., 10., 1., 5.),
            wind_reason_none);
  check_int("trigger: distance mode at d_max",
            wind_recoupling_trigger(D, 2., 1., 0.5, 2., 10., 1., 5.),
            wind_reason_distance);
  check_int("trigger: distance mode at t_rec",
            wind_recoupling_trigger(D, 1., 10., 0.5, 2., 10., 1., 5.),
            wind_reason_time);
  check_int("trigger: pressure, P_w <= P_amb but d < d_min",
            wind_recoupling_trigger(P, 0.4, 1., 0.5, 2., 10., 1., 5.),
            wind_reason_none);
  check_int("trigger: pressure, d >= d_min, P_w <= P_amb",
            wind_recoupling_trigger(P, 0.5, 1., 0.5, 2., 10., 5., 5.),
            wind_reason_pressure);
  check_int("trigger: pressure, d >= d_min, P_w > P_amb",
            wind_recoupling_trigger(P, 1., 1., 0.5, 2., 10., 6., 5.),
            wind_reason_none);
  check_int("trigger: pressure, P_amb = 0 never by pressure",
            wind_recoupling_trigger(P, 1., 1., 0.5, 2., 10., 0., 0.),
            wind_reason_none);
  check_int("trigger: pressure mode, backstop d_max",
            wind_recoupling_trigger(P, 2.5, 1., 0.5, 2., 10., 6., 5.),
            wind_reason_distance);
  check_int("trigger: pressure mode, backstop t_rec",
            wind_recoupling_trigger(P, 1., 11., 0.5, 2., 10., 6., 5.),
            wind_reason_time);
  check_int("trigger: pressure reported before backstops",
            wind_recoupling_trigger(P, 3., 11., 0.5, 2., 10., 1., 5.),
            wind_reason_pressure);

  printf("%d failure(s)\n", n_fail);
  return n_fail == 0 ? 0 : 1;
}
