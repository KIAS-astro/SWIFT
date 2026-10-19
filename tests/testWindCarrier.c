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
 * Checks the wind-carrier functions of src/wind_physics.h
 * (wind-carrier-design.md):
 *  - the number of carriers per step: on a grid of 1e6 uniform numbers its
 *    mean is p_c (1e-5, the grid spacing) and it is floor(p_c) or
 *    floor(p_c) + 1; so the launched mass per step, n m_c with
 *    m_c = m_g / N_c, has mean eta_M mdot_* dt for N_c = 1, 8, 64 (1e-5);
 *  - its variance is (p_c - floor(p_c)) (1 - p_c + floor(p_c)), so the
 *    relative scatter of the launched mass at fixed expectation falls as
 *    N_c^-1/2 when p_c < 1 (1e-5);
 *  - the deposit into 48 gas neighbours with cubic-spline kernel shares
 *    (kernel_hydro.h): shares sum to 1 and each neighbour receives
 *    w_j m_c (1e-12); total mass, momentum, energy (kinetic + thermal +
 *    potential in a uniform field) and metal mass of carrier plus gas are
 *    the same before and after (1e-12 relative, double round-off over 48
 *    sums); the gas velocity is the mass-weighted mean (1e-12).
 */

#include <config.h>

/* Local includes. */
#include "kernel_hydro.h"
#include "wind_physics.h"

/* System includes. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int n_fail = 0;

static void check(const char *name, const double measured,
                  const double expected, const double tol) {
  const double rel = expected != 0. ? fabs(measured / expected - 1.)
                                    : fabs(measured - expected);
  const int ok = rel <= tol;
  printf("%s %-62s measured %.12g expected %.12g rel %.2e (tol %.0e)\n",
         ok ? "ok  " : "FAIL", name, measured, expected, rel, tol);
  if (!ok) n_fail++;
}

/* Mean and variance of the carrier count over a grid of uniform numbers */
static void count_moments(const double p_c, double *mean, double *var,
                          int *n_min, int *n_max) {
  const int n = 1000000;
  double s = 0., s2 = 0.;
  *n_min = 1 << 30;
  *n_max = -1;
  for (int i = 0; i < n; i++) {
    const int k = wind_carrier_count(p_c, (i + 0.5) / n);
    s += k;
    s2 += (double)k * k;
    if (k < *n_min) *n_min = k;
    if (k > *n_max) *n_max = k;
  }
  *mean = s / n;
  *var = s2 / n - *mean * *mean;
}

static double uniform(void) { return rand() / ((double)RAND_MAX + 1.); }

int main(int argc, char *argv[]) {

  /* The carrier count */
  const double pcs[5] = {0.003, 0.25, 0.7, 2.4, 7.0};
  for (int i = 0; i < 5; i++) {
    double mean, var;
    int n_min, n_max;
    count_moments(pcs[i], &mean, &var, &n_min, &n_max);
    const double f = pcs[i] - floor(pcs[i]);
    char name[96];
    sprintf(name, "count(p_c = %g): mean", pcs[i]);
    check(name, mean, pcs[i], 1e-5);
    sprintf(name, "count(p_c = %g): variance", pcs[i]);
    check(name, var, f * (1. - f), 1e-5);
    sprintf(name, "count(p_c = %g): min = floor(p_c)", pcs[i]);
    check(name, n_min, floor(pcs[i]), 0.);
    sprintf(name, "count(p_c = %g): max = floor(p_c) + 1", pcs[i]);
    check(name, n_max, floor(pcs[i]) + (f > 0. ? 1. : 0.), 0.);
  }

  /* Launched mass per step for N_c = 1, 8, 64 at the same mdot_w dt:
   * m_g = 1, eta_M mdot_* dt = 0.004 */
  const double m_g = 1., mdot_w_dt = 0.004;
  const int Ncs[3] = {1, 8, 64};
  double rel_scatter[3];
  for (int i = 0; i < 3; i++) {
    const double m_c = m_g / Ncs[i];
    double mean, var;
    int n_min, n_max;
    count_moments(mdot_w_dt / m_c, &mean, &var, &n_min, &n_max);
    char name[96];
    sprintf(name, "N_c = %d: mean launched mass / (eta_M mdot_* dt)", Ncs[i]);
    check(name, mean * m_c / mdot_w_dt, 1., 1e-5);
    rel_scatter[i] = sqrt(var) * m_c / mdot_w_dt;
  }
  check("scatter(N_c = 8) / scatter(N_c = 1) vs sqrt((1 - p8)/(8(1 - p1)))",
        rel_scatter[1] / rel_scatter[0],
        sqrt((1. - 8. * mdot_w_dt) / (8. * (1. - mdot_w_dt))), 1e-5);
  check("scatter(N_c = 64) / scatter(N_c = 1) vs sqrt((1-p64)/(64(1-p1)))",
        rel_scatter[2] / rel_scatter[0],
        sqrt((1. - 64. * mdot_w_dt) / (64. * (1. - mdot_w_dt))), 1e-5);

  /* The deposit: a carrier at the origin with 48 gas neighbours */
  srand(12345);
  const int n = 48;
  const double h = 1.3;
  const double m_c = 0.37, u_c = 812.5, Z_c = 0.031;
  const double v_c[3] = {120., -35., 410.};
  const double a[3] = {-0.8, 1.7, -12.3};
  double x[48][3], m[48], v[48][3], u[48], Z[48], w[48];
  double w_sum = 0.;
  for (int j = 0; j < n; j++) {
    double r;
    do {
      for (int k = 0; k < 3; k++) x[j][k] = (2. * uniform() - 1.) * h;
      r = sqrt(x[j][0] * x[j][0] + x[j][1] * x[j][1] + x[j][2] * x[j][2]);
    } while (r >= kernel_gamma * h);
    float wf;
    kernel_eval((float)(r / h), &wf);
    w[j] = wf;
    w_sum += w[j];
    m[j] = 1. + 0.3 * uniform();
    for (int k = 0; k < 3; k++) v[j][k] = 50. * (2. * uniform() - 1.);
    u[j] = 20. + 10. * uniform();
    Z[j] = 0.01 * uniform();
  }

  /* Totals before: potential phi = -a . x (uniform field a) */
  double M0 = m_c, P0[3], E0 = 0., MZ0 = m_c * Z_c;
  double vc2 = 0.;
  for (int k = 0; k < 3; k++) {
    P0[k] = m_c * v_c[k];
    vc2 += v_c[k] * v_c[k];
  }
  E0 = m_c * (0.5 * vc2 + u_c); /* the carrier sits at x = 0, phi = 0 */
  for (int j = 0; j < n; j++) {
    double v2 = 0., phi = 0.;
    for (int k = 0; k < 3; k++) {
      v2 += v[j][k] * v[j][k];
      phi -= a[k] * x[j][k];
      P0[k] += m[j] * v[j][k];
    }
    M0 += m[j];
    E0 += m[j] * (0.5 * v2 + u[j] + phi);
    MZ0 += m[j] * Z[j];
  }

  /* Deposit */
  double share_sum = 0., max_share_err = 0., max_v_err = 0.;
  double M1 = 0., P1[3] = {0., 0., 0.}, E1 = 0., MZ1 = 0.;
  for (int j = 0; j < n; j++) {
    const double share = wind_carrier_share(w[j], w_sum);
    share_sum += share;
    const double dm = share * m_c;
    max_share_err = fmax(max_share_err, fabs(dm / (w[j] / w_sum * m_c) - 1.));
    double a_dx = 0.;
    for (int k = 0; k < 3; k++) a_dx += a[k] * (0. - x[j][k]);
    double v_new[3];
    const double E_th =
        wind_carrier_deposit(m[j], v[j], u[j], dm, v_c, u_c, a_dx, v_new);
    const double m_new = m[j] + dm;
    const double Z_new = wind_carrier_mix(m[j], Z[j], dm, Z_c);
    double v2 = 0., phi = 0.;
    for (int k = 0; k < 3; k++) {
      const double v_ref = (m[j] * v[j][k] + dm * v_c[k]) / m_new;
      max_v_err = fmax(max_v_err, fabs(v_new[k] - v_ref) / fabs(v_c[k]));
      v2 += v_new[k] * v_new[k];
      phi -= a[k] * x[j][k];
      P1[k] += m_new * v_new[k];
    }
    M1 += m_new;
    E1 += 0.5 * m_new * v2 + E_th + m_new * phi;
    MZ1 += m_new * Z_new;
  }
  check("deposit: sum of shares", share_sum, 1., 1e-12);
  check("deposit: max |dm_j / (w_j m_c) - 1|", max_share_err, 0., 1e-12);
  check("deposit: gas velocity = mass-weighted mean (max rel err)", max_v_err,
        0., 1e-12);
  check("deposit: total mass", M1, M0, 1e-12);
  for (int k = 0; k < 3; k++) {
    char name[96];
    sprintf(name, "deposit: total momentum [%d]", k);
    check(name, P1[k], P0[k], 1e-12);
  }
  check("deposit: total energy (kinetic + thermal + potential)", E1, E0,
        1e-12);
  check("deposit: total metal mass", MZ1, MZ0, 1e-12);

  if (n_fail) {
    printf("%d checks failed\n", n_fail);
    return 1;
  }
  printf("All checks passed\n");
  return 0;
}
