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
 * Checks src/prfm_physics.h.
 *
 * Against our own numbers:
 *  - t_dep of PRFM-int with OK22, rho_d = 0.01 Msun/pc^3, sigma_*z = sigma_eff,
 *    against the values computed in ai-notes papers-part8-prfm.md Sec. 11F
 *    (rounded to two digits, 3 per cent; not from any paper) and against the
 *    Python reference swift_ism_model/prfm.py (repository swift-ism-model;
 *    1e-5, the difference of the physical constants of the two codes);
 *  - the gas-only limit W = pi G Sigma_g^2 / 2 (1e-6);
 *  - OK22 Upsilon(1e4 K cm^-3) = 1028 km/s, K24 Upsilon(1e4, Z' = 0.1) =
 *    3072 km/s (2e-3), and the clipping of Z' at 0.1;
 *  - the dispersion floor (OK22 only) and the K24 EoS implied by eq. 23.
 *
 * Against the papers (Hassan et al. 2024, H24; Jeffreson et al. 2026, J26;
 * Kim et al. 2024, K24):
 *  - H24 eq. 26, H = 110 pc (sigma_eff / 15 km/s)^2 [(Sigma_* + Sigma_g) /
 *    150 Msun/pc^2]^-1, the equal-thickness, no-dark-matter scale height,
 *    from prfm_weight at a fixed sigma_eff; tolerance 4.5 per cent, half a
 *    unit in the second printed digit of 110;
 *  - J26 eq. 17 (OK22), eps_dyn = t_dyn / t_dep = 0.012 P_4^0.43, and K24
 *    eq. 28 (J26 eq. A6), eps_dyn = 0.0071 W_4^0.41 Z'^0.30 at Z' = 1, 0.3,
 *    0.1, from prfm_int for gas-only layers over the fitted ranges
 *    (P_4 = 1-100 above the OK22 floor; W_4 = 0.2-80 for K24, whose fit
 *    covers log W/k_B = 3.1-5.9). Tolerance: half a unit in the last digit of
 *    the printed coefficient (4.2 and 0.7 per cent), plus for OK22 half a
 *    unit in the last digit of the exponent times ln(10) log10(P_4)
 *    (the K24 exponents are exact sums of the eq. 20 and 23 exponents);
 *  - the TIGRESS-NCR models of K24 (Tables 1-3, the 18 models with
 *    Z'_g = Z'_d and beta_0 = 1): Table 1 Sigma_star, z_*, rho_dm and Table 2
 *    Sigma_gas, Sigma_sp through the weight with the K24 calibration, and
 *    W / Upsilon_tot against Table 2 Sigma_SFR. See the comment there.
 *
 * The kernel-column estimate of the gas surface density:
 *  - w(z; H) of prfm_kernel_projected against 2 pi int_z^H W(r) r dr of the
 *    build's kernel_eval_double (1e-6; the kernel constants are floats);
 *  - the bias f(x) of a sech^2 layer against a 10^5-point trapezoid sum
 *    (1e-6);
 *  - a uniform slab of half-thickness Z: the SPH density at its midplane,
 *    rho_0 int 4 pi r^2 W(r) min(1, Z / r) dr from the build's kernel,
 *    divided by w(0; H), against the exact Sigma_est / Sigma =
 *    int_-Z^Z w(z; H) dz / (2 Z w(0; H)), H / (2.8 Z) for Z >= H (1e-6);
 *  - the equilibrium correction returns Sigma_g of a sech^2 layer from its
 *    estimate f(H_k / H_g) Sigma_g (1e-6).
 */

#include <config.h>

/* Local includes. */
#include "prfm_physics.h"

/* System includes. */
#include <math.h>
#include <stdio.h>

#include "kernel_hydro.h"

/* Units of the Python reference */
static const double msun = 1.98841e33;
static const double pc = 3.0856776e18;
static const double gyr = 3.15576e16;
static const double kms = 1e5;

static int n_fail = 0;

/* The build's kernel of support H at radius r [1/length^3] */
static double kernel_W(const double r, const double H) {
  const double h = H / (double)kernel_gamma;
  double w;
  kernel_eval_double(r / h, &w);
  return w / (h * h * h);
}

/* Midpoint sum of f over [a, b] with n points */
static double midpoint(double (*f)(double, const double *), const double *par,
                       const double a, const double b, const int n) {
  const double dx = (b - a) / n;
  double s = 0.;
  for (int i = 0; i < n; i++) s += f(a + (i + 0.5) * dx, par);
  return s * dx;
}

/* 2 pi r W(r; H), par = {H} */
static double ring(double r, const double *par) {
  return 2. * M_PI * r * kernel_W(r, par[0]);
}

/* 4 pi r^2 W(r; H) min(1, Z / r), par = {H, Z} */
static double slab_shell(double r, const double *par) {
  const double frac = r > par[1] ? par[1] / r : 1.;
  return 4. * M_PI * r * r * kernel_W(r, par[0]) * frac;
}

/* sech^2(u) w(u; x) / w(0; x), par = {x} */
static double sech2_w(double u, const double *par) {
  const double c = cosh(u);
  return prfm_kernel_projected(u, par[0]) / prfm_kernel_projected(0., par[0]) /
         (c * c);
}

/* int_0^zeta w(z; 1) dz, exact */
static double kernel_projected_cumulative(const double zeta) {
  if (zeta >= 1.) return 0.5;
  if (zeta < 0.5) {
    const double z2 = zeta * zeta;
    return 16. * (0.0875 * zeta - z2 * zeta / 6. + 0.3 * z2 * z2 * zeta -
                  0.2 * z2 * z2 * z2);
  }
  const double y = 1. - zeta;
  const double y5 = y * y * y * y * y;
  const double a5 = 1. / 32.;
  return 16. * (0.0875 * 0.5 - 0.125 / 6. + 0.3 * a5 - 0.2 * a5 * 0.5) +
         16. * ((a5 / 10. - a5 * 0.5 / 15.) - (y5 / 10. - y5 * y / 15.));
}

static void check(const char *name, const double measured,
                  const double expected, const double tol) {
  const double rel = fabs(measured / expected - 1.);
  const int ok = rel <= tol;
  printf("%s %-50s measured %.10g expected %.10g rel %.2e (tol %.0e)\n",
         ok ? "ok  " : "FAIL", name, measured, expected, rel, tol);
  if (!ok) n_fail++;
}

static void check_range(const char *name, const double measured,
                        const double lo, const double hi) {
  const int ok = measured >= lo && measured <= hi;
  printf("%s %-50s measured %.10g range [%g, %g]\n", ok ? "ok  " : "FAIL",
         name, measured, lo, hi);
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

  /* Depletion times: (Sigma_g, Sigma_*) [Msun/pc^2], t_dep of P8 Sec. 11F
   * [Gyr], Python t_dep [Gyr], W [K cm^-3], sigma_eff [km/s], H_g [pc] */
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
    sprintf(name, "(%g,%g): t_dep vs notes P8 11F [Gyr]", cases[i][0],
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
        prfm_upsilon(1e4 * P_unit, &k24, 0.1), 1e-12);

  /* Dispersion floor below P_0: OK22 (J26 Sec. 2.5) only; K24 eq. 23 is
   * fitted down to 10^3.1 K cm^-3 and gives 8.9 km/s at 1e3 */
  check("OK22 sigma_eff(1e3) [km/s]", prfm_sigma_eff(1e3 * P_unit, &ok22, 1.) / kms,
        12., 1e-12);
  check("K24 sigma_eff(1e3, Z'=1) [km/s], no floor",
        prfm_sigma_eff(1e3 * P_unit, &k24, 1.) / kms, 11.7 * pow(10., -0.12),
        1e-12);

  /* Effective EoS at n_H = 1 cm^-3 */
  check("OK22 P_eff(1)/k_B [K cm^-3]", prfm_eos_pressure(1., &ok22, 1.) / P_unit,
        4.7e4, 1e-12);

  /* K24 EoS implied by eq. 23, against its rounded form 3.03e4 n_H^1.316
   * Z'^0.079 (prfm-method-spec.md Sec. 1.3); tolerance half a unit in the
   * last printed digit */
  check("K24 EoS P/k_B at n_H = 1, Z' = 1 [K cm^-3]",
        prfm_eos_pressure(1., &k24, 1.) / P_unit, 3.03e4, 0.005 / 3.03);
  check("K24 EoS slope", k24.eos_slope, 1.316, 0.0005 / 1.316);
  check("K24 EoS metallicity exponent", k24.eos_Z, 0.079, 0.0005 / 0.079);

  /* K24 EoS consistency: P_eff at the equilibrium density W / sigma_eff(W)^2
   * returns W (exact up to rounding) */
  {
    const double Ws[4] = {1e3, 1e4, 1e5, 1e6};
    const double Zs[3] = {1., 0.3, 0.1};
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 3; j++) {
        const double W = Ws[i] * P_unit;
        const double s = prfm_sigma_eff(W, &k24, Zs[j]);
        const double n_eq =
            W / (s * s) / (PRFM_MASS_PER_H * PRFM_PROTON_MASS_CGS);
        char name[80];
        sprintf(name, "K24 P_eff(n_eq)/W, W/k_B = %g, Z' = %g", Ws[i], Zs[j]);
        check(name, prfm_eos_pressure(n_eq, &k24, Zs[j]) / W, 1., 1e-10);
      }
    }
  }

  /* H24 eq. 26: H_g = sigma_eff^2 / (pi G (Sigma_g + Sigma_*)) with
   * H_* = H_g and no dark matter, i.e. sigma_*z = sigma_eff, rho_d = 0 */
  {
    const double totals[2] = {150., 300.};
    for (int i = 0; i < 2; i++) {
      const double Sg = 28. * Sigma_unit;
      const double Ss = (totals[i] - 28.) * Sigma_unit;
      const double sigma = 15. * kms;
      const double W = prfm_weight(Sg, Ss, 0., sigma, sigma, c_d);
      const double H = Sg * sigma * sigma / (2. * W);
      char name[80];
      sprintf(name, "H24 eq. 26 H_g [pc], Sigma_g + Sigma_* = %g", totals[i]);
      check(name, H / pc, 110. * 150. / totals[i], 0.045);
    }
  }

  /* J26 eq. 17 (OK22) and K24 eq. 28: eps_dyn = t_dyn / t_dep for gas-only
   * layers, Sigma_g = (2 W / pi G)^(1/2) */
  {
    for (int i = 0; i <= 8; i++) {
      const double P4 = pow(10., 0.25 * i);
      const double Sg =
          sqrt(2. * P4 * 1e4 * P_unit / (M_PI * PRFM_NEWTON_G_CGS));
      struct prfm_state s;
      prfm_int(Sg, 0., 0., -1., c_d, &ok22, 1., 5, &s);
      const double P4_s = s.W / (1e4 * P_unit);
      char name[80];
      sprintf(name, "J26 eq. 17 eps_dyn, OK22, P_4 = %.3g", P4_s);
      check(name, s.t_dyn / s.t_dep, 0.012 * pow(P4_s, 0.43),
            0.0005 / 0.012 + 0.005 * log(10.) * log10(P4_s));
    }
    const double Zs[3] = {1., 0.3, 0.1};
    for (int j = 0; j < 3; j++) {
      for (int i = 0; i <= 6; i++) {
        const double W4 = 0.2 * pow(400., i / 6.);
        const double Sg =
            sqrt(2. * W4 * 1e4 * P_unit / (M_PI * PRFM_NEWTON_G_CGS));
        struct prfm_state s;
        prfm_int(Sg, 0., 0., -1., c_d, &k24, Zs[j], 5, &s);
        const double W4_s = s.W / (1e4 * P_unit);
        char name[80];
        sprintf(name, "K24 eq. 28 eps_dyn, W_4 = %.3g, Z' = %g", W4_s, Zs[j]);
        check(name, s.t_dyn / s.t_dep,
              0.0071 * pow(W4_s, 0.41) * pow(Zs[j], 0.30), 0.00005 / 0.0071);
      }
    }
  }

  /* TIGRESS-NCR models, K24 Tables 1-3.
   *
   * The weight of each model is computed with the K24 calibration from
   * Table 2 Sigma_gas, Table 1 Sigma_star and rho_dm, and H_* = z_* (Table 1;
   * the old stellar disc has midplane density Sigma_star / (2 z_*), K24 eq.
   * 12). The young star particles, Table 2 Sigma_sp, are counted as a disc
   * as thick as the gas, as K24 eq. 10 counts them. TIGRESS has a vertical
   * dark matter gravity 4 pi G rho_dm z, i.e. Omega_d^2 = 4 pi G rho_dm, so
   * c_d = 8 zeta Omega_d^2 / (G rho_dm) = 32 pi / 3 (zeta = 1/3). The H24
   * cubic for H_g at the measured H_* is solved by fixed-point iteration of
   * J26 eq. 31 with sigma_*z / sigma_eff = H_* / H_g, which is the quadratic
   * J26 eq. 31 solves at a given H_* / H_g.
   *
   * 1. W / W_measured (Table 3 column 2) must lie in [0.6, 1.25]: the range
   *    of K24's own analytic estimator P_DE / W over these models (Table 3
   *    column 10 over column 2: 0.62 for S150-Om100q0-Z1.0 to 1.21 for
   *    S05-Z1.0).
   * 2. W_measured / Upsilon_tot(W_measured, Z') against Table 2 Sigma_SFR,
   *    which tests the eq. 20 fit alone: K24 Fig. 10 shows relative errors
   *    of that fit "mostly" within +-0.5; at least 80 per cent of the models
   *    must lie within it. */
  {
    /* Z', Sigma_gas, Sigma_SFR [Msun/kpc^2/yr], Sigma_sp, Sigma_star [Msun/pc^2],
     * z_* [pc], rho_dm [Msun/pc^3], W_measured [1e4 K cm^-3] */
    const double tig[18][8] = {
        {1., 4.07, 1.13e-4, 0.242, 1., 500., 0.002, 0.125},     /* S05-Z1.0 */
        {0.1, 4.76, 4.74e-5, 0.0616, 1., 500., 0.002, 0.146},   /* S05-Z0.1 */
        {3., 8.74, 3.74e-3, 3.48, 42., 245., 0.0064, 1.61},     /* R8-Z3.0 */
        {1., 9.22, 2.97e-3, 2.55, 42., 245., 0.0064, 1.80},     /* R8-Z1.0 */
        {0.3, 9.96, 1.92e-3, 1.67, 42., 245., 0.0064, 1.84},    /* R8-Z0.3 */
        {0.1, 10.6, 1.29e-3, 0.970, 42., 245., 0.0064, 1.85},   /* R8-Z0.1 */
        {1., 16.2, 0.0114, 13.1, 42., 245., 0.0064, 4.30},      /* S30-Z1.0 */
        {0.1, 22.5, 7.92e-3, 6.67, 42., 245., 0.0064, 5.67},    /* S30-Z0.1 */
        {3., 35.7, 0.0503, 16.4, 50., 500., 0.005, 10.9},       /* LGR4-Z3.0 */
        {1., 36.1, 0.0343, 14.6, 50., 500., 0.005, 10.4},       /* LGR4-Z1.0 */
        {0.3, 42.0, 0.0252, 8.19, 50., 500., 0.005, 11.8},      /* LGR4-Z0.3 */
        {0.1, 44.6, 0.0186, 5.43, 50., 500., 0.005, 12.6},      /* LGR4-Z0.1 */
        {1., 42.0, 0.0860, 50.6, 50., 500., 0.005, 22.2},       /* S100-Z1.0 */
        {0.1, 62.4, 0.0885, 34.7, 50., 500., 0.005, 28.2},      /* S100-Z0.1 */
        {1., 88.7, 0.133, 66.0, 50., 500., 0.005, 58.7},  /* S150-Om200-Z1.0 */
        {0.1, 115., 0.108, 37.5, 50., 500., 0.005, 72.7}, /* S150-Om200-Z0.1 */
        {1., 69.9, 0.278, 74.3, 50., 500., 0.005, 53.4},  /* S150-Om100q0-Z1.0 */
        {0.1, 68.7, 0.261, 74.5, 50., 500., 0.005, 42.0} /* S150-Om100q0-Z0.1 */
    };
    const double c_d_tigress = 32. * M_PI / 3.;
    const double sfr_unit = msun / (1e6 * pc * pc) / 3.15576e7;
    int n_within = 0;
    for (int m = 0; m < 18; m++) {
      const double Zp = tig[m][0];
      const double Sg = tig[m][1] * Sigma_unit;
      const double Ssp = tig[m][3] * Sigma_unit;
      const double Ss = tig[m][4] * Sigma_unit;
      const double Hs = tig[m][5] * pc;
      const double rho_d = tig[m][6] * rho_unit;
      const double W_meas = tig[m][7] * 1e4 * P_unit;

      double sigma = PRFM_SIGMA_START_CGS;
      double W = prfm_weight(Sg, Ss + Ssp, rho_d, sigma, sigma, c_d_tigress);
      for (int it = 0; it < 200; it++) {
        const double ratio = Hs / (Sg * sigma * sigma / (2. * W));
        W = prfm_weight(Sg, Ss + 0.5 * Ssp * (1. + ratio), rho_d, sigma,
                        sigma * ratio, c_d_tigress);
        sigma = prfm_sigma_eff(W, &k24, Zp);
      }
      char name[80];
      sprintf(name, "K24 TIGRESS model %d: W / W_measured", m);
      check_range(name, W / W_meas, 0.6, 1.25);

      const double sfr = W_meas / prfm_upsilon(W_meas, &k24, Zp) / sfr_unit;
      const double rel = sfr / tig[m][2] - 1.;
      printf("     K24 TIGRESS model %2d: Sigma_SFR(W_measured) / Table 2 = %.3f\n",
             m, sfr / tig[m][2]);
      if (fabs(rel) <= 0.5) n_within++;
    }
    check_range("K24 TIGRESS: fraction with |Sigma_SFR error| <= 0.5",
                n_within / 18., 0.8, 1.);
  }

  /* Kernel column of the build's kernel */
  {
    const double H = 2.7;
    for (int i = 0; i < 6; i++) {
      const double q = 0.15 * i + 0.05 * (i == 0 ? -1 : 0);
      const double z = (q < 0. ? 0. : q) * H;
      const double par[1] = {H};
      const double num = midpoint(ring, par, z, H, 200000);
      char name[80];
      sprintf(name, "w(z; H) at z / H = %.2f", z / H);
      check(name, prfm_kernel_projected(z, H), num, 1e-6);
    }
    check("w(0; H) H = 1.4", prfm_kernel_projected(0., H) * H, 1.4, 1e-12);
  }

  /* Bias of a sech^2 layer */
  {
    const double xs[6] = {0.5, 1., 2., 2.6, 10., 100.};
    for (int i = 0; i < 6; i++) {
      const double par[1] = {xs[i]};
      const double umax = xs[i] < 40. ? xs[i] : 40.;
      const double num = midpoint(sech2_w, par, 0., umax, 100000);
      char name[80];
      sprintf(name, "f(H_k / H = %g)", xs[i]);
      check(name, prfm_kernel_column_bias(xs[i]), num, 1e-6);
    }
  }

  /* Uniform slab of half-thickness Z: Sigma_est / Sigma */
  {
    const double H = 1.3;
    const double zs[7] = {1e-3, 0.1, 0.25, 0.5, 0.75, 1., 2.};
    for (int i = 0; i < 7; i++) {
      const double Z = zs[i] * H;
      const double par[2] = {H, Z};
      const double rho_sph = midpoint(slab_shell, par, 0., H, 400000);
      const double ratio = rho_sph / prfm_kernel_projected(0., H) / (2. * Z);
      const double exact =
          2. * kernel_projected_cumulative(zs[i]) / (2. * zs[i] * 1.4);
      char name[80];
      sprintf(name, "slab Z / H_k = %g: Sigma_est / Sigma", zs[i]);
      check(name, ratio, exact, 1e-6);
      if (zs[i] >= 1.) {
        sprintf(name, "slab Z / H_k = %g: H_k / (2.8 Z)", zs[i]);
        check(name, ratio, 1. / (2.8 * zs[i]), 1e-6);
      }
      if (zs[i] < 0.01) check("thin slab: Sigma_est / Sigma", ratio, 1., 1e-5);
    }
  }

  /* Equilibrium correction recovers the layer */
  {
    const double Sg = 20. * Sigma_unit;
    const double rho_d = 0.01 * rho_unit;
    const double xs[6] = {1.5, 3., 10., 1.2, 3., 10.};
    for (int i = 0; i < 6; i++) {
      /* Star dominated, then gas only */
      const double Ss = i < 3 ? 50. * Sigma_unit : 0.;
      struct prfm_state st;
      prfm_int(Sg, Ss, rho_d, -1., c_d, &ok22, 1., 5, &st);
      const double H_k = xs[i] * st.H_g;
      const double Sest = prfm_kernel_column_bias(xs[i]) * Sg;
      const double got = prfm_kernel_column_equilibrium(
          Sest, H_k, Ss, rho_d, -1., c_d, &ok22, 1., 5, 20, &st);
      char name[80];
      sprintf(name, "equilibrium correction, Sigma_* = %g, H_k / H_g = %g",
              Ss / Sigma_unit, xs[i]);
      check(name, got, Sg, 1e-6);
    }
  }

  if (n_fail) {
    printf("%d checks failed\n", n_fail);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
