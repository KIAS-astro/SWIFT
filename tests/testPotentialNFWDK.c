/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2026 Jeong-Gyu Kim (omyojj@gmail.com)
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
 * Unit test of the NFW + Diemer & Kravtsov (2014) external potential
 * (src/potential/nfw_dk), for the halo of Smith et al. (2024, MNRAS 527,
 * 1216, Table 1): M200 = 1e11 Msun, c = 10, Planck 2018 (h = 0.6766,
 * Omega_m = 0.3111), s_e = 1.5, b_e = 1. Runs only in a build configured
 * with --with-ext-potential=nfw-dk; otherwise it reports a skip.
 *
 * Checks:
 *  1. R_200 = 97.94 kpc (S24a Table 1: 97.9 kpc), to 1e-3: the expected value
 *     uses G and Mpc in cgs from the PDG, which differ from SWIFT's in the
 *     fifth digit.
 *  2. R_200m encloses 200 times the mean matter density in the NFW part, to
 *     1e-9 (bisection to round-off).
 *  3. The NFW part encloses M200 at R_200, to 1e-9.
 *  4. Enclosed mass equals the integral of 4 pi r^2 rho over r, to 1e-5
 *     (trapezoid rule on 2e5 log-spaced points; error ~ (d ln r)^2 ~ 1e-9
 *     relative, set by the r^-1.5 term near the inner limit).
 *  5. The acceleration equals -dPhi/dr from a centred difference of the
 *     potential, to 1e-6 (difference step 1e-4 r: truncation ~1e-8 and
 *     round-off ~1e-12 / 1e-4; a_grav is single precision, ~1e-7).
 */

#include <config.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "swift.h"

static int n_fail = 0;

static void check(const char *name, const double measured,
                  const double expected, const double tol) {
  const double rel = expected != 0. ? fabs(measured / expected - 1.)
                                    : fabs(measured - expected);
  const int ok = rel <= tol;
  printf("%s %-58s measured %.10g expected %.10g rel %.2e (tol %.0e)\n",
         ok ? "ok  " : "FAIL", name, measured, expected, rel, tol);
  if (!ok) n_fail++;
}

int main(int argc, char *argv[]) {

#ifndef EXTERNAL_POTENTIAL_NFW_DK
  printf("testPotentialNFWDK: skipped (not built with nfw-dk)\n");
  return 0;
#else
  struct unit_system us;
  units_init(&us, 1.98848e43, 3.085678e21, 3.085678e16, 1., 1.);
  struct swift_params params;
  parser_init("none", &params);
  struct phys_const phys_const;
  phys_const_init(&us, &params, &phys_const);

  parser_set_param(&params, "NFWDKPotential:position:[0., 0., 0.]");
  parser_set_param(&params, "NFWDKPotential:useabspos:1");
  parser_set_param(&params, "NFWDKPotential:timestep_mult:0.01");
  parser_set_param(&params, "NFWDKPotential:concentration:10.");
  parser_set_param(&params, "NFWDKPotential:M_200:10.");
  parser_set_param(&params, "NFWDKPotential:epsilon:0.");
  parser_set_param(&params, "NFWDKPotential:h:0.6766");
  parser_set_param(&params, "NFWDKPotential:Omega_m:0.3111");

  static struct space s;
  static struct external_potential pot;
  potential_init_backend(&params, &phys_const, &us, &s, &pot);
  potential_print_backend(&pot);

  /* 1. R_200 from rho_crit = 3 H0^2 / (8 pi G), cgs */
  const double G_cgs = 6.67430e-8, Mpc = 3.08567758149e24, Msun = 1.98841e33;
  const double H0 = 0.6766 * 1e7 / Mpc;
  const double rho_crit = 3. * H0 * H0 / (8. * M_PI * G_cgs);
  const double R200_cgs =
      cbrt(3. * 1e11 * Msun / (4. * M_PI * 200. * rho_crit));
  check("R_200 [kpc]", pot.R_200, R200_cgs / 3.08567758149e21, 1e-3);

  /* 2. R_200m */
  {
    const double x = pot.R_200m / pot.r_s;
    const double m = pot.M_s * (log(1. + x) - x / (1. + x));
    const double vol = 4. * M_PI / 3. * pow(pot.R_200m, 3);
    check("NFW mean density inside R_200m / rho_m", m / vol / pot.rho_m, 200.,
          1e-9);
  }

  /* 3. NFW mass at R_200 */
  {
    const double x = pot.R_200 / pot.r_s;
    check("NFW mass inside R_200 / M_200",
          pot.M_s * (log(1. + x) - x / (1. + x)) / pot.M_200, 1., 1e-9);
  }

  /* 4. Enclosed mass against the integral of the density */
  {
    const int n = 200000;
    const double r_lo = 1e-6, radii[4] = {1., 10., 97.9, 600.};
    for (int k = 0; k < 4; k++) {
      const double dl = log(radii[k] / r_lo) / n;
      double sum = 0.;
      for (int i = 0; i <= n; i++) {
        const double r = r_lo * exp(i * dl);
        const double f = 4. * M_PI * r * r * r * nfw_dk_density(&pot, r);
        sum += (i == 0 || i == n) ? 0.5 * f : f;
      }
      char name[96];
      sprintf(name, "M(<%g kpc) against integral of 4 pi r^2 rho", radii[k]);
      check(name, nfw_dk_enclosed_mass(&pot, radii[k]),
            sum * dl + nfw_dk_enclosed_mass(&pot, r_lo), 1e-5);
    }
  }

  /* 5. Acceleration against the potential */
  {
    const double radii[5] = {0.5, 5., 50., 300., 600.};
    for (int k = 0; k < 5; k++) {
      struct gpart g;
      bzero(&g, sizeof(struct gpart));
      const double r = radii[k];
      g.x[0] = r / sqrt(3.);
      g.x[1] = r / sqrt(3.);
      g.x[2] = r / sqrt(3.);
      external_gravity_acceleration(0., &pot, &phys_const, &g);
      const double a = sqrt(g.a_grav[0] * g.a_grav[0] +
                            g.a_grav[1] * g.a_grav[1] +
                            g.a_grav[2] * g.a_grav[2]);
      const double dr = 1e-4 * r;
      const double dphi = (nfw_dk_potential_over_G(&pot, r + dr) -
                           nfw_dk_potential_over_G(&pot, r - dr)) /
                          (2. * dr);
      char name[96];
      sprintf(name, "|a| / (dPhi/dr) at r = %g kpc", r);
      check(name, a, dphi, 1e-6);
    }
  }

  if (n_fail) {
    printf("%d checks failed\n", n_fail);
    return 1;
  }
  printf("All checks passed\n");
  return 0;
#endif
}
