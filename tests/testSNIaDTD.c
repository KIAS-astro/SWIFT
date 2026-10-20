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

/* Config parameters. */
#include <config.h>

/* Standard includes. */
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Local includes. */
#include "swift.h"

#if defined(FEEDBACK_EAGLE_THERMAL) || defined(FEEDBACK_EAGLE_KINETIC) || \
    defined(FEEDBACK_EAGLE_THERMAL_KINETIC)

/* Defined in src/feedback/EAGLE/enrichment.h, compiled into the library */
double eagle_feedback_number_of_SNIa(const double M_init, const double t0,
                                     const double t1,
                                     const struct feedback_props *props);

static int n_fail = 0;

static void check(const char *name, const double measured,
                  const double expected, const double tol) {
  const double rel = fabs(measured / expected - 1.);
  const int ok = rel <= tol;
  printf("%s %-58s measured %.12g expected %.12g rel %.2e\n",
         ok ? "ok  " : "FAIL", name, measured, expected, rel);
  if (!ok) n_fail++;
}

/**
 * @brief Number of SNIa per unit initial stellar mass of the exponential
 * delay-time distribution (eagle_feedback_number_of_SNIa) in its two
 * conventions, with nu = 1.54e-3 per Msun, tau = 2 Gyr, t_d = 40 Myr
 * (COLIBRE, Schaye et al. 2026):
 *  - EAGLE, exponential in star age t: N(t1, t2) = nu [e^-t1/tau - e^-t2/tau],
 *    total from t_d to infinity nu e^-t_d/tau = 0.9802 nu;
 *  - COLIBRE, exponential in t - t_d: N = nu [e^-(t1-t_d)/tau -
 *    e^-(t2-t_d)/tau], total nu;
 *  - for any window the ratio EAGLE / COLIBRE is e^-t_d/tau;
 *  - 2000 consecutive steps from t_d to 13.6 Gyr sum to the single interval.
 * The expected values use the same float parameters as the code, so the
 * tolerance (1e-12) is double-precision rounding.
 */
int main(int argc, char *argv[]) {

  struct feedback_props fp;
  memset(&fp, 0, sizeof(fp));
  fp.SNIa_DTD = eagle_feedback_SNIa_DTD_exponential;
  fp.SNIa_DTD_exp_norm = 1.54e-3f;
  fp.SNIa_DTD_exp_timescale_Gyr = 2.f;
  fp.SNIa_DTD_exp_timescale_Gyr_inv = 1.f / fp.SNIa_DTD_exp_timescale_Gyr;
  fp.SNIa_DTD_delay_Gyr = 0.04f;
  fp.mass_to_solar_mass = 1.f;

  const double nu = fp.SNIa_DTD_exp_norm;
  const double itau = fp.SNIa_DTD_exp_timescale_Gyr_inv;
  const double td = fp.SNIa_DTD_delay_Gyr;
  const double t_inf = 1e3;
  const double t1 = 0.0978, t2 = 0.1488;

  for (int colibre = 0; colibre <= 1; colibre++) {
    fp.SNIa_DTD_exp_from_delay = colibre;
    const char *conv = colibre ? "COLIBRE" : "EAGLE";
    const double shift = colibre ? td : 0.;
    char name[96];

    sprintf(name, "%s: total SNIa per Msun / nu", conv);
    check(name, eagle_feedback_number_of_SNIa(1., td, t_inf, &fp) / nu,
          colibre ? 1. : exp(-td * itau), 1e-12);

    sprintf(name, "%s: SNIa per Msun, age %g-%g Gyr", conv, t1, t2);
    check(name, eagle_feedback_number_of_SNIa(1., t1, t2, &fp),
          nu * (exp(-(t1 - shift) * itau) - exp(-(t2 - shift) * itau)), 1e-12);

    sprintf(name, "%s: 2 Msun star makes twice the SNIa", conv);
    check(name, eagle_feedback_number_of_SNIa(2., t1, t2, &fp),
          2. * eagle_feedback_number_of_SNIa(1., t1, t2, &fp), 1e-14);

    const int n_steps = 2000;
    const double t_end = 13.6;
    double sum = 0.;
    for (int i = 0; i < n_steps; i++) {
      const double a = td + (t_end - td) * i / n_steps;
      const double b = td + (t_end - td) * (i + 1) / n_steps;
      sum += eagle_feedback_number_of_SNIa(1., a, b, &fp);
    }
    sprintf(name, "%s: %d steps to 13.6 Gyr sum to one interval", conv,
            n_steps);
    check(name, sum, eagle_feedback_number_of_SNIa(1., td, t_end, &fp), 1e-12);
  }

  /* The conventions differ by the constant factor e^-t_d/tau */
  fp.SNIa_DTD_exp_from_delay = 0;
  const double n_eagle = eagle_feedback_number_of_SNIa(1., t1, t2, &fp);
  fp.SNIa_DTD_exp_from_delay = 1;
  const double n_colibre = eagle_feedback_number_of_SNIa(1., t1, t2, &fp);
  check("EAGLE / COLIBRE = exp(-t_d / tau)", n_eagle / n_colibre,
        exp(-td * itau), 1e-12);
  check("EAGLE / COLIBRE = 0.9802 (4 digits)", n_eagle / n_colibre, 0.9802,
        5e-5 / 0.9802);

  if (n_fail) error("%d checks failed", n_fail);
  message("all checks passed");
  return 0;
}

#else

/* Don't do anything if not using an EAGLE feedback model */
int main(int argc, char *argv[]) { return 0; }

#endif
