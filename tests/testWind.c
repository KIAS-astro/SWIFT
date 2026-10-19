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
 * Checks src/wind_physics.h:
 *  - launch kinematics: Delta v^2 / 2 + u_w = u_* eta_E / eta_M (1e-12) and
 *    (Delta v^2 / 2) / u_w = 5/9 (1e-12), at the Milky Way TIGRESS loadings
 *    and at a constant (eta_M, eta_E) = (0.5, 0.1);
 *  - the draw: on a grid of 1e6 uniform numbers the star and wind fractions
 *    are p_* and p_w (1e-5, the grid spacing), the outcomes exclude each
 *    other by construction, and for p_* + p_w > 1 both are renormalised;
 *  - the loadings against the K20b formulas evaluated by hand at
 *    Sigma_SFR = 1e-3, 1e-2, 1e-1 (1e-12), and the clipping outside [1e-4, 1].
 */

#include <config.h>

/* Local includes. */
#include "wind_physics.h"

/* System includes. */
#include <math.h>
#include <stdio.h>

static int n_fail = 0;

static void check(const char *name, const double measured,
                  const double expected, const double tol) {
  const double rel = expected != 0. ? fabs(measured / expected - 1.)
                                    : fabs(measured - expected);
  const int ok = rel <= tol;
  printf("%s %-58s measured %.12g expected %.12g rel %.2e (tol %.0e)\n",
         ok ? "ok  " : "FAIL", name, measured, expected, rel, tol);
  if (!ok) n_fail++;
}

static void check_draw(const double p_star, const double p_wind) {
  const int n = 1000000;
  int n_star = 0, n_wind = 0;
  for (int i = 0; i < n; i++) {
    const double u = (i + 0.5) / n;
    const enum wind_draw_outcome o = wind_draw(u, p_star, p_wind);
    if (o == wind_draw_star) n_star++;
    if (o == wind_draw_wind) n_wind++;
  }
  const double total = p_star + p_wind;
  const double norm = total > 1. ? 1. / total : 1.;
  char name[96];
  sprintf(name, "draw (p_*=%g, p_w=%g): star fraction", p_star, p_wind);
  check(name, (double)n_star / n, p_star * norm, 1e-5);
  sprintf(name, "draw (p_*=%g, p_w=%g): wind fraction", p_star, p_wind);
  check(name, (double)n_wind / n, p_wind * norm, 1e-5);
  sprintf(name, "draw (p_*=%g, p_w=%g): star + wind fraction", p_star,
          p_wind);
  check(name, (double)(n_star + n_wind) / n, total * norm, 1e-5);
}

int main(int argc, char *argv[]) {

  const double kms = 1e5;
  const double u_star = WIND_U_STAR_CGS;
  check("u_* [(km/s)^2] = 1e51 erg / 95.5 Msun", u_star / (kms * kms),
        5.2659e5, 1e-4);

  /* Loadings: (Sigma_SFR, eta_M, eta_E) by hand from K20b Table 5 */
  const double sig[3] = {1e-3, 1e-2, 1e-1};
  for (int i = 0; i < 3; i++) {
    const double eM = pow(10., -0.86 - 0.07 * log10(sig[i]));
    const double eE = pow(10., -0.70 + 0.14 * log10(sig[i]));
    char name[96];
    sprintf(name, "eta_M,hot(Sigma_SFR = %g)", sig[i]);
    check(name, wind_TIGRESS_eta_M(sig[i]), eM, 1e-12);
    sprintf(name, "eta_E,hot(Sigma_SFR = %g)", sig[i]);
    check(name, wind_TIGRESS_eta_E(sig[i]), eE, 1e-12);
  }
  check("eta_M clipped below at Sigma_SFR = 1e-4",
        wind_TIGRESS_eta_M(1e-7), wind_TIGRESS_eta_M(1e-4), 0.);
  check("eta_E clipped above at Sigma_SFR = 1", wind_TIGRESS_eta_E(30.),
        wind_TIGRESS_eta_E(1.), 0.);
  check("eta_M(1e-2) = 0.19 (WD3)", wind_TIGRESS_eta_M(1e-2), 0.19, 0.03);
  check("eta_E(1e-2) = 0.11 (WD3, rounded)", wind_TIGRESS_eta_E(1e-2), 0.11,
        0.05);

  /* Launch kinematics */
  const double loads[2][2] = {
      {wind_TIGRESS_eta_M(1e-2), wind_TIGRESS_eta_E(1e-2)}, {0.5, 0.1}};
  for (int i = 0; i < 2; i++) {
    double dv, u_w;
    const double eM = loads[i][0], eE = loads[i][1];
    wind_launch_kinematics(u_star, eM, eE, WIND_KINETIC_FRACTION_MACH1, &dv,
                           &u_w);
    char name[96];
    sprintf(name, "(eta_M, eta_E) = (%.3g, %.3g): dv^2/2 + u_w", eM, eE);
    check(name, 0.5 * dv * dv + u_w, u_star * eE / eM, 1e-12);
    sprintf(name, "(eta_M, eta_E) = (%.3g, %.3g): (dv^2/2) / u_w", eM, eE);
    check(name, 0.5 * dv * dv / u_w, 5. / 9., 1e-12);
    sprintf(name, "(eta_M, eta_E) = (%.3g, %.3g): Mach (gamma = 5/3)", eM, eE);
    check(name, dv / sqrt(5. / 3. * 2. / 3. * u_w), 1., 1e-12);
    const double T_w = u_w * (2. / 3.) * 0.6 * 1.67262192e-24 / 1.380649e-16;
    printf("     Delta v = %.1f km/s, u_w = %.4g (km/s)^2, T(mu=0.6) = %.3g K\n",
           dv / kms, u_w / (kms * kms), T_w);
    if (i == 0) {
      check("Sigma_SFR = 1e-2: Delta v = 454 km/s (WD3)", dv / kms, 454., 0.01);
      check("Sigma_SFR = 1e-2: T_w = 9.0e6 K (WD3)", T_w, 9.0e6, 0.01);
    }
  }

  /* Smith et al. (2024a) Table 2 LowSpec and HighSpec, as run by the
   * isolated_galaxy/arkenstone_va regression test: eta_E = eta_E,kin +
   * eta_E,th and kinetic_fraction = eta_E,kin / eta_E. The table gives three
   * digits, so the tolerance is 3e-3. */
  {
    const double eM[2] = {6.41, 0.32}, eK[2] = {0.869, 0.321},
                 eT[2] = {0.097, 0.579}, dv_tab[2] = {378., 1028.},
                 cs_tab[2] = {94., 1028.};
    const char *names[2] = {"LowSpec", "HighSpec"};
    for (int i = 0; i < 2; i++) {
      double dv, u_w;
      wind_launch_kinematics(u_star, eM[i], eK[i] + eT[i],
                             eK[i] / (eK[i] + eT[i]), &dv, &u_w);
      char name[96];
      sprintf(name, "S24a %s: Delta v [km/s]", names[i]);
      check(name, dv / kms, dv_tab[i], 3e-3);
      sprintf(name, "S24a %s: c_s [km/s]", names[i]);
      check(name, sqrt(5. / 3. * 2. / 3. * u_w) / kms, cs_tab[i], 3e-3);
    }
  }

  /* The draw */
  check_draw(0.01, 0.005);
  check_draw(0.3, 0.06);
  check_draw(0.8, 0.4);
  check_draw(2., 1.);
  check_draw(0.2, 0.);

  if (n_fail) {
    printf("%d checks failed\n", n_fail);
    return 1;
  }
  printf("All checks passed\n");
  return 0;
}
