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
 * Checks src/prfm_physics.h against the Python reference implementation
 * swift_ism_model/prfm.py (repository swift-ism-model) and its unit tests:
 *  - t_dep of PRFM-int with OK22, rho_d = 0.01 Msun/pc^3, sigma_*z = sigma_eff,
 *    against the published values (3 per cent) and the Python numbers
 *    (1e-5, the difference of the physical constants of the two codes);
 *  - the gas-only limit W = pi G Sigma_g^2 / 2 (1e-6);
 *  - OK22 Upsilon(1e4 K cm^-3) = 1028 km/s, K24 Upsilon(1e4, Z' = 0.1) =
 *    3072 km/s (2e-3), and the clipping of Z' at 0.1.
 */

#include <config.h>

/* Local includes. */
#include "prfm_physics.h"

/* System includes. */
#include <math.h>
#include <stdio.h>

/* Units of the Python reference */
static const double msun = 1.98841e33;
static const double pc = 3.0856776e18;
static const double gyr = 3.15576e16;
static const double kms = 1e5;

static int n_fail = 0;

static void check(const char *name, const double measured,
                  const double expected, const double tol) {
  const double rel = fabs(measured / expected - 1.);
  const int ok = rel <= tol;
  printf("%s %-50s measured %.10g expected %.10g rel %.2e (tol %.0e)\n",
         ok ? "ok  " : "FAIL", name, measured, expected, rel, tol);
  if (!ok) n_fail++;
}

int main(int argc, char *argv[]) {

  struct prfm_calibration ok22, k24;
  prfm_calibration_init(prfm_calibration_OK22, &ok22);
  prfm_calibration_init(prfm_calibration_K24, &k24);

  const double Sigma_unit = msun / (pc * pc);
  const double rho_unit = msun / (pc * pc * pc);
  const double P_unit = PRFM_BOLTZMANN_K_CGS; /* K cm^-3 to erg cm^-3 */
  const double c_d = 16. * M_PI / 3.;

  /* Depletion times: (Sigma_g, Sigma_*) [Msun/pc^2], published t_dep [Gyr],
   * Python t_dep [Gyr], W [K cm^-3], sigma_eff [km/s], H_g [pc] */
  const double cases[4][7] = {
      {5., 10., 8.6, 8.573689851, 3573.929594, 12., 493.7732069},
      {10., 40., 2.5, 2.467976302, 17690.7928, 13.60459601, 256.4277493},
      {28., 110., 0.61, 0.6112282641, 130858.8386, 21.1289502, 234.1271754},
      {100., 200., 0.18, 0.1844291357, 1005297.726, 33.08918574, 266.942963}};

  for (int i = 0; i < 4; i++) {
    struct prfm_state s;
    prfm_int(cases[i][0] * Sigma_unit, cases[i][1] * Sigma_unit,
             0.01 * rho_unit, -1., c_d, &ok22, 1., 5, &s);
    char name[64];
    sprintf(name, "(%g,%g): t_dep vs published [Gyr]", cases[i][0],
            cases[i][1]);
    check(name, s.t_dep / gyr, cases[i][2], 3e-2);
    sprintf(name, "(%g,%g): t_dep vs Python [Gyr]", cases[i][0], cases[i][1]);
    check(name, s.t_dep / gyr, cases[i][3], 1e-5);
    sprintf(name, "(%g,%g): W vs Python [K cm^-3]", cases[i][0], cases[i][1]);
    check(name, s.W / P_unit, cases[i][4], 1e-5);
    sprintf(name, "(%g,%g): sigma_eff vs Python [km/s]", cases[i][0],
            cases[i][1]);
    check(name, s.sigma_eff / kms, cases[i][5], 1e-5);
    sprintf(name, "(%g,%g): H_g vs Python [pc]", cases[i][0], cases[i][1]);
    check(name, s.H_g / pc, cases[i][6], 1e-5);
    sprintf(name, "(%g,%g): t_dyn = 2 H_g / sigma_eff", cases[i][0],
            cases[i][1]);
    check(name, s.t_dyn, 2. * s.H_g / s.sigma_eff, 1e-12);
  }

  /* Gas-only limit (H24 eq. 5) */
  {
    struct prfm_state s;
    const double Sg = 20. * Sigma_unit;
    prfm_int(Sg, 0., 0., -1., c_d, &ok22, 1., 5, &s);
    check("gas only: W / (pi G Sigma_g^2 / 2)", s.W,
          0.5 * M_PI * PRFM_NEWTON_G_CGS * Sg * Sg, 1e-6);
    check("gas only: W vs Python [K cm^-3]", s.W / P_unit, 13246.8078115,
          1e-5);
    check("gas only: t_dep = 2 Upsilon / (pi G Sigma_g)", s.t_dep,
          2. * prfm_upsilon(s.W, &ok22, 1.) / (M_PI * PRFM_NEWTON_G_CGS * Sg),
          1e-12);
  }

  /* Yields */
  check("OK22 Upsilon(1e4) [km/s]",
        prfm_upsilon(1e4 * P_unit, &ok22, 1.) / kms, 1028., 2e-3);
  check("K24 Upsilon(1e4, Z'=0.1) [km/s]",
        prfm_upsilon(1e4 * P_unit, &k24, 0.1) / kms, 3072., 2e-3);
  check("K24 Upsilon(1e4, Z'=1) [km/s]",
        prfm_upsilon(1e4 * P_unit, &k24, 1.) / kms, 1650., 2e-3);
  check("K24 Upsilon(1e6, Z'=0.1) [km/s]",
        prfm_upsilon(1e6 * P_unit, &k24, 0.1) / kms, 808., 2e-3);
  check("K24 Upsilon clipped at Z'=0.1",
        prfm_upsilon(1e4 * P_unit, &k24, 0.01),
        prfm_upsilon(1e4 * P_unit, &k24, 0.1), 1e-15);

  /* Dispersion floor below P_0 */
  check("OK22 sigma_eff(1e3) [km/s]", prfm_sigma_eff(1e3 * P_unit, &ok22, 1.) / kms,
        12., 1e-15);

  /* Effective EoS at n_H = 1 cm^-3 */
  check("OK22 P_eff(1)/k_B [K cm^-3]", prfm_eos_pressure(1., &ok22, 1.) / P_unit,
        4.7e4, 1e-15);
  check("K24 P_eff(10, Z'=1)/k_B [K cm^-3]",
        prfm_eos_pressure(10., &k24, 1.) / P_unit, 2.85e4 * pow(10., 1.29),
        1e-15);

  if (n_fail) {
    printf("%d checks failed\n", n_fail);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
