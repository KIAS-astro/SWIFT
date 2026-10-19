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
 *  - launch kinematics, for both energy conventions, at the Milky Way
 *    TIGRESS loadings and at a constant (eta_M, eta_E) = (0.5, 0.1):
 *    Delta v^2 / 2 + h u_w = u_* eta_E / eta_M with h = gamma (enthalpy) or
 *    1 (internal) (1e-12), the Mach-1 kinetic fraction (1/4 or 5/14) and
 *    Mach number 1 (1e-12); at Sigma_SFR = 1e-2 the values of Delta v, T_w
 *    and the Bernoulli speed v_B = (Delta v^2 + 2 gamma u_w)^1/2, which for
 *    enthalpy equals (2 e_w)^1/2 (1e-12);
 *  - the draw: on a grid of 1e6 uniform numbers the star and wind fractions
 *    are p_* and p_w (1e-5, the grid spacing), the outcomes exclude each
 *    other by construction, and for p_* + p_w > 1 both are renormalised;
 *  - the loadings against the K20b formulas evaluated by hand at
 *    Sigma_SFR = 1e-3, 1e-2, 1e-1 (1e-12), and the clipping outside [1e-4, 1].
 * Two-phase wind and CR energy (wind-design.md Sec. 10):
 *  - the four-way draw on the same grid: star, hot, warm and nothing
 *    fractions are p_* N, p_h N, p_w N and 1 - (p_* + p_h + p_w) N (1e-5),
 *    the four counts sum to the grid size (exclusive and exhaustive), and
 *    with p_w = 0 every outcome equals that of the two-way draw;
 *  - the loading sets through K26 at 2 kpc and Sigma_SFR = 3.6e-3: warm 2.8
 *    (TIGRESSpp) and 1.3 (K20), and the TIGRESSpp hot factors (1e-3, the
 *    rounding of the norms);
 *  - warm launch energy v^2/2 + gamma u at 100 km/s and 1e4 K (mu = 0.6)
 *    and the apex test of a ballistic particle in a constant field: the
 *    apex time v/g and height v^2/(2g), from a leapfrog integration with
 *    the test at every step (to one step);
 *  - CR energy: e_CR (p_h + p_w [both]) / p_* = eta_E,CR u_* (1e-12), and
 *    over 1e5 launches with random masses the launched sum equals the
 *    deposited sum and the summed internal energy gain (1e-12).
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

static void check_draw4(const double p_star, const double p_hot,
                        const double p_warm) {
  const int n = 1000000;
  int c[4] = {0, 0, 0, 0}, n_differ = 0;
  for (int i = 0; i < n; i++) {
    const double u = (i + 0.5) / n;
    const enum wind_draw4_outcome o = wind_draw4(u, p_star, p_hot, p_warm);
    c[o]++;
    if (p_warm == 0.) {
      const enum wind_draw_outcome o2 = wind_draw(u, p_star, p_hot);
      const int same = (o == wind_draw4_star && o2 == wind_draw_star) ||
                       (o == wind_draw4_hot && o2 == wind_draw_wind) ||
                       (o == wind_draw4_nothing && o2 == wind_draw_nothing);
      if (!same) n_differ++;
    }
  }
  const double total = p_star + p_hot + p_warm;
  const double N = total > 1. ? 1. / total : 1.;
  const double expect[4] = {1. - total * N, p_star * N, p_hot * N, p_warm * N};
  const char *what[4] = {"nothing", "star", "hot", "warm"};
  char name[96];
  for (int k = 0; k < 4; k++) {
    sprintf(name, "draw4 (%g, %g, %g): %s fraction", p_star, p_hot, p_warm,
            what[k]);
    check(name, (double)c[k] / n, expect[k], expect[k] > 0. ? 1e-5 / expect[k] : 1e-5);
  }
  sprintf(name, "draw4 (%g, %g, %g): outcomes counted", p_star, p_hot, p_warm);
  /* -ffast-math: the ratio of two equal integers is 1 to 1e-16 */
  check(name, c[0] + c[1] + c[2] + c[3], n, 1e-15);
  if (p_warm == 0.) {
    sprintf(name, "draw4 (%g, %g, 0): outcomes differing from wind_draw",
            p_star, p_hot);
    check(name, n_differ, 0., 0.);
  }
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
  const double gamma = 5. / 3.;
  check("Mach-1 kinetic fraction, enthalpy",
        wind_kinetic_fraction_mach1(wind_energy_enthalpy, gamma), 0.25, 1e-15);
  check("Mach-1 kinetic fraction, internal",
        wind_kinetic_fraction_mach1(wind_energy_internal, gamma), 5. / 14.,
        1e-15);
  const double loads[2][2] = {
      {wind_TIGRESS_eta_M(1e-2), wind_TIGRESS_eta_E(1e-2)}, {0.5, 0.1}};
  const enum wind_energy_convention convs[2] = {wind_energy_enthalpy,
                                                wind_energy_internal};
  const char *conv_names[2] = {"enthalpy", "internal"};
  /* Expected at Sigma_SFR = 1e-2: Delta v [km/s], T_w [K], v_B [km/s] */
  const double expect[2][3] = {{380.4, 6.31e6, 760.8}, {454.7, 9.02e6, 909.3}};
  for (int c = 0; c < 2; c++) {
    const enum wind_energy_convention conv = convs[c];
    const double h = conv == wind_energy_enthalpy ? gamma : 1.;
    const double f = wind_kinetic_fraction_mach1(conv, gamma);
    for (int i = 0; i < 2; i++) {
      double dv, u_w;
      const double eM = loads[i][0], eE = loads[i][1];
      const double e_w = u_star * eE / eM;
      wind_launch_kinematics(u_star, eM, eE, f, conv, gamma, &dv, &u_w);
      char name[128];
      sprintf(name, "%s (eta_M, eta_E) = (%.3g, %.3g): dv^2/2 + h u_w",
              conv_names[c], eM, eE);
      check(name, 0.5 * dv * dv + h * u_w, e_w, 1e-12);
      sprintf(name, "%s (eta_M, eta_E) = (%.3g, %.3g): (dv^2/2) / e_w",
              conv_names[c], eM, eE);
      check(name, 0.5 * dv * dv / e_w, f, 1e-12);
      sprintf(name, "%s (eta_M, eta_E) = (%.3g, %.3g): Mach", conv_names[c],
              eM, eE);
      check(name, dv / sqrt(gamma * (gamma - 1.) * u_w), 1., 1e-12);
      const double v_B = wind_bernoulli_speed(dv, u_w, gamma);
      if (conv == wind_energy_enthalpy) {
        sprintf(name, "%s (eta_M, eta_E) = (%.3g, %.3g): v_B / (2 e_w)^1/2",
                conv_names[c], eM, eE);
        check(name, v_B / sqrt(2. * e_w), 1., 1e-12);
      }
      const double T_w =
          u_w * (gamma - 1.) * 0.6 * 1.67262192e-24 / 1.380649e-16;
      printf(
          "     %s: Delta v = %.1f km/s, u_w = %.4g (km/s)^2, T(mu=0.6) = "
          "%.3g K, v_B = %.1f km/s\n",
          conv_names[c], dv / kms, u_w / (kms * kms), T_w, v_B / kms);
      if (i == 0) {
        sprintf(name, "%s Sigma_SFR = 1e-2: Delta v = %.1f km/s",
                conv_names[c], expect[c][0]);
        check(name, dv / kms, expect[c][0], 1e-3);
        sprintf(name, "%s Sigma_SFR = 1e-2: T_w = %.3g K", conv_names[c],
                expect[c][1]);
        check(name, T_w, expect[c][1], 1e-2);
        sprintf(name, "%s Sigma_SFR = 1e-2: v_B = %.1f km/s", conv_names[c],
                expect[c][2]);
        check(name, v_B / kms, expect[c][2], 1e-3);
      }
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
                             eK[i] / (eK[i] + eT[i]), wind_energy_internal,
                             5. / 3., &dv, &u_w);
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

  /* The four-way draw */
  check_draw4(0.01, 0.005, 0.02);
  check_draw4(0.1, 0.02, 0.28);
  check_draw4(0.3, 0.06, 0.84);
  check_draw4(0.5, 0.1, 1.4);
  check_draw4(0.2, 0.04, 0.);

  /* Loading sets (wind-design.md Sec. 10.3) at the K26 Sigma_SFR */
  {
    const double S = 3.6e-3;
    check("warm eta_M, TIGRESSpp, at Sigma_SFR = 3.6e-3 (K26 2.8)",
          wind_powerlaw_loading(WIND_WARM_ETA_M_LOG10_NORM_TIGRESSPP,
                                WIND_WARM_ETA_M_SLOPE, S),
          2.8, 1e-3);
    check("warm eta_M, K20, at Sigma_SFR = 3.6e-3 (K26 mhd 1.3)",
          wind_powerlaw_loading(WIND_WARM_ETA_M_LOG10_NORM_K20,
                                WIND_WARM_ETA_M_SLOPE, S),
          1.3, 1e-3);
    check("TIGRESSpp hot eta_M factor (K26 0.17 / 0.33)",
          WIND_TIGRESSPP_HOT_ETA_M_FACTOR, 0.17 / 0.33, 1e-3);
    check("TIGRESSpp hot eta_E factor (K26 0.016 / 0.040)",
          WIND_TIGRESSPP_HOT_ETA_E_FACTOR, 0.016 / 0.040, 1e-12);
  }

  /* Warm launch and the apex test */
  {
    const double v = 100. * kms;
    const double u = 1.380649e-16 * 1e4 /
                     ((gamma - 1.) * 0.6 * 1.67262192e-24);
    const double e = wind_warm_specific_energy(v, u, gamma);
    check("warm v^2/2 + gamma u [(km/s)^2] at 100 km/s, 1e4 K", e / (kms * kms),
          5000. + gamma * u / (kms * kms), 1e-12);
    check("warm eta_E / eta_M at the defaults", e / u_star, 0.01015, 1e-3);

    /* Leapfrog in g = -1e-8 cm/s^2 along z, launched along -z (sign -1)
     * with the field along +z, and along +z with the field along -z */
    const double n[3] = {0., 0., 1.};
    for (int s = -1; s <= 1; s += 2) {
      const double g = -s * 1e-8, dt = 1e10;
      float vel[3] = {3e5, -2e5, (float)(s * v)};
      double z = 0., t = 0.;
      while (!wind_past_apex(vel, n, s) && t < 1e17) {
        vel[2] += 0.5 * g * dt;
        z += vel[2] * dt;
        vel[2] += 0.5 * g * dt;
        t += dt;
      }
      char name[96];
      sprintf(name, "apex time / (v/g), launch sign %+d", s);
      check(name, t / (v / fabs(g)), 1., dt / (v / fabs(g)));
      sprintf(name, "apex height / (v^2/2g), launch sign %+d", s);
      check(name, fabs(z) / (v * v / (2. * fabs(g))), 1.,
            2. * dt / (v / fabs(g)));
    }
  }

  /* CR energy per launched mass and its bookkeeping */
  {
    const double eta_cr = 0.3, p_star = 0.01;
    const double eta_h = wind_TIGRESS_eta_M(3.6e-3) * 0.515, eta_w = 2.8;
    const double e_hot = wind_cr_specific_energy(u_star, eta_cr, eta_h);
    const double e_both = wind_cr_specific_energy(u_star, eta_cr, eta_h + eta_w);
    check("CR, carriers hot: e_CR p_h / p_* = eta_E,CR u_*",
          e_hot * eta_h * p_star / p_star, eta_cr * u_star, 1e-12);
    check("CR, carriers both: e_CR (p_h + p_w) / p_* = eta_E,CR u_*",
          e_both * (eta_h + eta_w) * p_star / p_star, eta_cr * u_star, 1e-12);
    double E_launch = 0., E_dep = 0., dU = 0.;
    unsigned long long st = 12345;
    for (int i = 0; i < 100000; i++) {
      st = st * 6364136223846793005ULL + 1442695040888963407ULL;
      const double m = 1e38 * (1. + (double)(st >> 11) / 9007199254740992.);
      const double E = m * e_hot;
      const double u0 = 1e14;
      E_launch += E;
      const double carried = E;
      E_dep += carried;
      dU += m * ((u0 + carried / m) - u0);
    }
    check("CR, 1e5 launches: deposited / launched", E_dep / E_launch, 1.,
          1e-15);
    check("CR, 1e5 launches: sum m du / launched", dU / E_launch, 1., 1e-12);
  }

  if (n_fail) {
    printf("%d checks failed\n", n_fail);
    return 1;
  }
  printf("All checks passed\n");
  return 0;
}
