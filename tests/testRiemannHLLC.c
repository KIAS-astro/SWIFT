/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (C) 2016 Bert Vandenbroucke (bert.vandenbroucke@gmail.com).
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
#include <config.h>

/* Some standard headers. */
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Force use of the HLLC Riemann solver */
#undef RIEMANN_SOLVER_TRRS
#undef RIEMANN_SOLVER_EXACT
#undef RIEMANN_SOLVER_HLLC
#define RIEMANN_SOLVER_HLLC 1

/* Local headers. */
#include "riemann/riemann_hllc.h"
#include "tools.h"

const float max_abs_error = 1e-3f;
const float max_rel_error = 1e-3f;
const float min_threshold = 1e-2f;

/**
 * @brief Checks whether two numbers are opposite of each others.
 */
int are_symmetric(float a, float b) {

  /* Check that the signs are different */
  if ((a * b) > 0.f) {
    message("Identical signs a=%.8e b=%.8e", a, b);
    return 0;
  }

  const float abs_a = fabsf(a);
  const float abs_b = fabsf(b);

  const float abs_error = fabsf(abs_a - abs_b);

  /* Check that we do not breach the absolute error limit. Fluxes between
     colliding states reach ~1e3 (rho v^2 with |v| up to 10 on each side),
     where single-precision round-off alone is ~1e-4, so the limit grows with
     the size of the flux. */
  if (abs_error > max_abs_error * fmaxf(1.f, 0.5f * (abs_a + abs_b))) {
    message("Absolute error too large a=%.8e b=%.8e abs=%.8e", a, b, abs_error);
    return 0;
  }

  /* Avoid FPEs... */
  if (fabsf(abs_a + abs_b) == 0.f) {
    return 1;
  }

  /* Avoid things close to 0 */
  if ((abs_a < min_threshold) || (abs_b < min_threshold)) {
    return 1;
  }

  const float rel_error = 0.5f * abs_error / fabsf(abs_a + abs_b);

  /* Check that we do not breach the relative error limit */
  if (rel_error > max_rel_error) {
    message("Relative error too large a=%.8e b=%.8e rel=%.8e", a, b, rel_error);
    return 0;
  }

  /* All good */
  return 1;
}

/**
 * @brief Check the symmetry of the HLLC Riemann solver for a random setup
 */
void check_riemann_symmetry(void) {
  float WL[5], WR[5], n_unit1[3], n_unit2[3], n_norm, vij[3], totflux1[5],
      totflux2[5];

  WL[0] = random_uniform(0.1f, 1.0f);
  WL[1] = random_uniform(-10.0f, 10.0f);
  WL[2] = random_uniform(-10.0f, 10.0f);
  WL[3] = random_uniform(-10.0f, 10.0f);
  WL[4] = random_uniform(0.1f, 1.0f);
  WR[0] = random_uniform(0.1f, 1.0f);
  WR[1] = random_uniform(-10.0f, 10.0f);
  WR[2] = random_uniform(-10.0f, 10.0f);
  WR[3] = random_uniform(-10.0f, 10.0f);
  WR[4] = random_uniform(0.1f, 1.0f);

  n_unit1[0] = random_uniform(-1.0f, 1.0f);
  n_unit1[1] = random_uniform(-1.0f, 1.0f);
  n_unit1[2] = random_uniform(-1.0f, 1.0f);

  n_norm = sqrtf(n_unit1[0] * n_unit1[0] + n_unit1[1] * n_unit1[1] +
                 n_unit1[2] * n_unit1[2]);
  n_unit1[0] /= n_norm;
  n_unit1[1] /= n_norm;
  n_unit1[2] /= n_norm;

  n_unit2[0] = -n_unit1[0];
  n_unit2[1] = -n_unit1[1];
  n_unit2[2] = -n_unit1[2];

  vij[0] = random_uniform(-10.0f, 10.0f);
  vij[1] = random_uniform(-10.0f, 10.0f);
  vij[2] = random_uniform(-10.0f, 10.0f);

  riemann_solve_for_flux(WL, WR, n_unit1, vij, totflux1);
  riemann_solve_for_flux(WR, WL, n_unit2, vij, totflux2);

  if (!are_symmetric(totflux1[0], totflux2[0]) ||
      !are_symmetric(totflux1[1], totflux2[1]) ||
      !are_symmetric(totflux1[2], totflux2[2]) ||
      !are_symmetric(totflux1[3], totflux2[3]) ||
      !are_symmetric(totflux1[4], totflux2[4])) {
    message("WL=[%.8e, %.8e, %.8e, %.8e, %.8e]", WL[0], WL[1], WL[2], WL[3],
            WL[4]);
    message("WR=[%.8e, %.8e, %.8e, %.8e, %.8e]", WR[0], WR[1], WR[2], WR[3],
            WR[4]);
    message("n_unit1=[%.8e, %.8e, %.8e]", n_unit1[0], n_unit1[1], n_unit1[2]);
    message("vij=[%.8e, %.8e, %.8e]\n", vij[0], vij[1], vij[2]);
    message(
        "Flux solver asymmetric: [%.6e,%.6e,%.6e,%.6e,%.6e] == "
        "[%.6e,%.6e,%.6e,%.6e,%.6e]\n",
        totflux1[0], totflux1[1], totflux1[2], totflux1[3], totflux1[4],
        totflux2[0], totflux2[1], totflux2[2], totflux2[3], totflux2[4]);
    error("Asymmetry in flux solution!");
  } else {
    /* message( */
    /*     "Flux solver symmetric: [%.3e,%.3e,%.3e,%.3e,%.3e] == " */
    /*     "[%.3e,%.3e,%.3e,%.3e,%.3e]\n", */
    /*     totflux1[0], totflux1[1], totflux1[2], totflux1[3], totflux1[4], */
    /*     totflux2[0], totflux2[1], totflux2[2], totflux2[3], totflux2[4]); */
  }
}

/**
 * @brief Check the star-region pressure that the MFM flux uses against the
 * exact solution of the Riemann problem.
 *
 * Exact values (gamma = 5/3) from Toro (2009), ch. 4, computed with an
 * independent solver. An approximate solver on unreconstructed states is
 * allowed a factor 2; the linearised (PVRS) pressure fails the cold
 * collisions, where it scales with the sound speed instead of rho v^2
 * (0.05 of the exact value at v = +-20).
 */
void check_riemann_middle_state_pressure(void) {

  /* rho_L, v_L, P_L, rho_R, v_R, P_R, exact P* */
  const float cases[6][7] = {{1.f, 0.f, 1.f, 0.125f, 0.f, 0.1f, 0.2939f},
                             {1.f, 0.f, 1e3f, 1.f, 0.f, 1.f, 446.3f},
                             {1.f, 0.f, 1e5f, 1.f, 0.f, 1.f, 4.456e4f},
                             {1.f, 2.f, 1.f, 1.f, -2.f, 1.f, 7.379f},
                             {1.f, 20.f, 1.f, 1.f, -20.f, 1.f, 535.6f},
                             {1.f, 0.f, 1e4f, 10.f, 0.f, 1.f, 7320.f}};
  const float n_unit[3] = {1.f, 0.f, 0.f};
  const float vij[3] = {0.f, 0.f, 0.f};

  for (int i = 0; i < 6; i++) {
    const float WL[5] = {cases[i][0], cases[i][1], 0.f, 0.f, cases[i][2]};
    const float WR[5] = {cases[i][3], cases[i][4], 0.f, 0.f, cases[i][5]};
    float totflux[5];
    riemann_solve_for_middle_state_flux(WL, WR, n_unit, vij, totflux);
    const float ratio = totflux[1] / cases[i][6];
    if (ratio < 0.5f || ratio > 2.f)
      error("Case %d: star pressure %.4e, exact %.4e (ratio %.3f)", i,
            totflux[1], cases[i][6], ratio);
  }
}

/**
 * @brief Check the HLLC Riemann solver
 */
int main(int argc, char *argv[]) {

  /* Initialize CPU frequency, this also starts time. */
  unsigned long long cpufreq = 0;
  clocks_set_cpufreq(cpufreq);

/* Choke on FP-exceptions */
#ifdef HAVE_FE_ENABLE_EXCEPT
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif

  /* Get some randomness going */
  const int seed = time(NULL);
  message("Seed = %d", seed);
  srand(seed);

  /* star pressure against the exact solution */
  check_riemann_middle_state_pressure();

  /* symmetry test */
  for (int i = 0; i < 100000; i++) {
    check_riemann_symmetry();
  }

  return 0;
}
