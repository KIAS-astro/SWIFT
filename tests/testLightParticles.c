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
 * Checks src/light_physics.h (light wind particles):
 *  - spawn count: on a grid of parent masses, reservoirs (negative ones
 *    included) and uniform numbers, n is floor(m_pend / m_l) or one more
 *    unless the parent's mass limits it, and a parent that spawns keeps at
 *    least m_l (exact integers); over 1e5 uniform numbers the mean of n is
 *    m_pend / m_l (1e-5, the grid spacing);
 *  - creation bookkeeping: a parent and its n clones hold the parent's mass
 *    exactly; the reservoir left plus the spawned mass equals the reservoir
 *    before (1e-15, double rounding); the momentum and energy of parent plus
 *    clones change by exactly the launch kicks and the launch energy
 *    (1e-12 relative);
 *  - merging: for 1e5 random pairs (mass ratios up to 1e3, velocities up to
 *    1e3, internal energies over six decades) the merged particle conserves
 *    mass, momentum, total energy (kinetic + internal) and metal mass to
 *    1e-12 of the larger term (double arithmetic; the worst case seen is
 *    printed), the centre of mass is kept, and the heat equals
 *    (1/2) mu |v_a - v_b|^2;
 *  - clock (Bennett et al. 2025, eqs. 6, 12, 13): the dye falls to f_thresh
 *    after 0.2 R_200 / (k v_B) (1e-12); the target mass is m_l above
 *    f_thresh, m_g at f_ref and below, and m_g (f_ref / f_w)^1/2 between
 *    (1e-15).
 */

#include <config.h>

/* Local includes. */
#include "light_physics.h"

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
  printf("%s %-60s measured %.15g expected %.15g rel %.2e (tol %.0e)\n",
         ok ? "ok  " : "FAIL", name, measured, expected, rel, tol);
  if (!ok) n_fail++;
}

static double uniform(void) { return (rand() + 0.5) / ((double)RAND_MAX + 1.); }

int main(int argc, char *argv[]) {

  srand(12345);

  /* Spawn count on a grid: n is floor(x) or floor(x) + 1, x = m_pend / m_l,
   * unless the parent's mass limits it; the parent keeps at least m_l */
  const double m_l = 0.1;
  int count_errors = 0;
  for (int a = 0; a < 200; a++) {
    for (int b = -20; b < 200; b++) {
      for (int c = 0; c < 10; c++) {
        const double m_p = 0.05 + 0.01 * a;
        const double m_pend = 0.0037 * b;
        const double u = (c + 0.5) / 10.;
        const long long n = light_spawn_count(m_p, m_pend, m_l, u);
        const double x = m_pend > 0. ? m_pend / m_l : 0.;
        const long long n_keep =
            m_p < 2. * m_l ? 0 : (long long)floor((m_p - m_l) / m_l);
        const int ok_res = n == (long long)floor(x) ||
                           n == (long long)floor(x) + 1;
        const int ok = n >= 0 && (n == n_keep ? n <= floor(x) + 1 : ok_res) &&
                       (n == 0 || m_p - n * m_l >= m_l * (1. - 1e-12));
        if (!ok) count_errors++;
      }
    }
  }
  check("spawn count: grid points violating the rule", count_errors, 0, 0.);

  /* Expected count: on a grid of 1e5 uniform numbers the mean of n equals
   * m_pend / m_l (to the grid spacing) when the parent does not limit it */
  {
    const int ng = 100000;
    const double m_pend = 0.537;
    double sum = 0.;
    for (int c = 0; c < ng; c++)
      sum += light_spawn_count(10., m_pend, m_l, (c + 0.5) / ng);
    check("spawn count: mean n = m_pend / m_l", sum / ng, m_pend / m_l, 1e-5);
  }

  /* Creation bookkeeping: a parent with velocity v_p spawns n clones, each
   * launched with a kick dv n_i and internal energy u_w */
  {
    const double m_p = 1.0, m_pend = 0.537, E_pend = 0.537 * 3.7e5;
    const double v_p[3] = {12., -40., 3.};
    const long long n = light_spawn_count(m_p, m_pend, m_l, 0.5);
    const double e_w = E_pend / m_pend;
    const double f_kin = 0.25, gamma = 5. / 3.;
    const double dv = sqrt(2. * f_kin * e_w), u_w = (1. - f_kin) * e_w / gamma;
    double M = m_p - n * m_l, P[3], E_kin = 0., kick_P[3] = {0., 0., 0.};
    for (int k = 0; k < 3; k++) P[k] = M * v_p[k];
    double v2p = v_p[0] * v_p[0] + v_p[1] * v_p[1] + v_p[2] * v_p[2];
    E_kin = 0.5 * M * v2p;
    double E_th_clones = 0., E_kick_cross = 0.;
    for (long long c = 0; c < n; c++) {
      const double sign = (c % 3 == 0) ? 1. : -1.;
      double v[3] = {v_p[0], v_p[1], v_p[2] + sign * dv};
      M += m_l;
      for (int k = 0; k < 3; k++) P[k] += m_l * v[k];
      kick_P[2] += m_l * sign * dv;
      E_kin += 0.5 * m_l * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
      E_kick_cross += m_l * sign * dv * v_p[2];
      E_th_clones += m_l * u_w;
    }
    check("creation: n clones from a 0.537 reservoir of m_l = 0.1", n, 5, 0.);
    check("creation: parent + clones mass", M, m_p, 1e-15);
    check("creation: reservoir left + spawned", (m_pend - n * m_l) + n * m_l,
          m_pend, 1e-15);
    check("creation: z momentum change = sum of kicks",
          P[2] - m_p * v_p[2], kick_P[2], 1e-12);
    check("creation: x momentum unchanged", P[0], m_p * v_p[0], 1e-15);
    const double E_expected = 0.5 * m_p * v2p + E_kick_cross +
                              n * m_l * (0.5 * dv * dv);
    check("creation: kinetic energy = parent + kicks", E_kin, E_expected,
          1e-12);
    check("creation: launch energy (kick^2/2 + gamma u_w) per mass = e_w",
          (n * m_l * 0.5 * dv * dv + gamma * E_th_clones) / (n * m_l), e_w,
          1e-12);
  }

  /* Merging */
  double worst_m = 0., worst_p = 0., worst_E = 0., worst_Z = 0.,
         worst_x = 0., worst_heat = 0.;
  for (int t = 0; t < 100000; t++) {
    double ma = pow(10., -3. * uniform()), mb = pow(10., -3. * uniform());
    double xa[3], xb[3], va[3], vb[3];
    for (int k = 0; k < 3; k++) {
      xa[k] = 10. * uniform();
      xb[k] = xa[k] + uniform() - 0.5;
      va[k] = 2e3 * (uniform() - 0.5);
      vb[k] = 2e3 * (uniform() - 0.5);
    }
    double ua = pow(10., 6. * uniform()), ub = pow(10., 6. * uniform());
    const double Za = 0.02 * uniform(), Zb = 0.02 * uniform();

    const double M0 = ma + mb;
    double P0[3], X0[3], v2rel = 0.;
    for (int k = 0; k < 3; k++) {
      P0[k] = ma * va[k] + mb * vb[k];
      X0[k] = ma * xa[k] + mb * xb[k];
      v2rel += (va[k] - vb[k]) * (va[k] - vb[k]);
    }
    const double Ka = 0.5 * ma * (va[0] * va[0] + va[1] * va[1] + va[2] * va[2]);
    const double Kb = 0.5 * mb * (vb[0] * vb[0] + vb[1] * vb[1] + vb[2] * vb[2]);
    const double E0 = Ka + Kb + ma * ua + mb * ub;
    const double MZ0 = ma * Za + mb * Zb;

    double m = ma, x[3] = {xa[0], xa[1], xa[2]}, v[3] = {va[0], va[1], va[2]},
           u = ua;
    const double heat = light_merge_state(&m, x, v, &u, mb, xb, vb, ub);
    const double Z = (ma * Za + mb * Zb) / m;

    const double scaleP = ma * 1e3 + mb * 1e3;
    double dP = 0., dX = 0.;
    for (int k = 0; k < 3; k++) {
      dP = fmax(dP, fabs(m * v[k] - P0[k]) / scaleP);
      dX = fmax(dX, fabs(m * x[k] - X0[k]) / (M0 * 10.));
    }
    const double E1 = 0.5 * m * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) +
                      m * u;
    worst_m = fmax(worst_m, fabs(m - M0) / M0);
    worst_p = fmax(worst_p, dP);
    worst_E = fmax(worst_E, fabs(E1 - E0) / fmax(Ka + Kb, ma * ua + mb * ub));
    worst_Z = fmax(worst_Z, fabs(m * Z - MZ0) / fmax(MZ0, 1e-300));
    worst_x = fmax(worst_x, dX);
    worst_heat = fmax(worst_heat,
                      fabs(heat - 0.5 * ma * mb / M0 * v2rel) /
                          fmax(0.5 * ma * mb / M0 * v2rel, 1e-300));
  }
  check("merge: worst relative mass error", worst_m, 0., 1e-12);
  check("merge: worst momentum error / (sum m) 1e3", worst_p, 0., 1e-12);
  check("merge: worst total-energy error / larger term", worst_E, 0., 1e-12);
  check("merge: worst metal-mass error", worst_Z, 0., 1e-12);
  check("merge: worst centre-of-mass error / (M L)", worst_x, 0., 1e-12);
  check("merge: worst heat / ((1/2) mu dv^2) - 1", worst_heat, 0., 1e-12);

  /* Clock */
  {
    const double kpc = 3.0857e21, kms = 1e5, Myr = 3.15576e13;
    const double R200 = 97.9 * kpc, k = 0.25, t0 = 10. * Myr;
    const double v_B = 1200. * kms, f_thresh = 1000., f_ref = 0.1;
    const double f0 = light_dye_init(f_thresh, R200, k, v_B, t0);
    const double t_flag = 0.2 * R200 / (k * v_B);
    check("clock: f_w after 0.2 R_200 / (k v_B) equals f_thresh",
          f0 * exp(-t_flag / t0), f_thresh, 1e-12);
    const double m_g = 1., ml = 0.1;
    check("clock: target above f_thresh is m_l",
          light_target_mass(2. * f_thresh, ml, m_g, f_thresh, f_ref), ml,
          1e-15);
    check("clock: target at f_ref is m_g",
          light_target_mass(f_ref, ml, m_g, f_thresh, f_ref), m_g, 1e-15);
    check("clock: target below f_ref is m_g",
          light_target_mass(0.01, ml, m_g, f_thresh, f_ref), m_g, 1e-15);
    check("clock: target at f_w = 0.4 is m_g (0.1 / 0.4)^1/2",
          light_target_mass(0.4, ml, m_g, f_thresh, f_ref), 0.5, 1e-15);
    check("clock: target at f_w = 100 is max(m_l, m_g / sqrt(1000))",
          light_target_mass(100., ml, m_g, f_thresh, f_ref), ml, 1e-15);
  }

  printf("%s: %d failed\n", n_fail ? "FAIL" : "PASS", n_fail);
  return n_fail ? 1 : 0;
}
