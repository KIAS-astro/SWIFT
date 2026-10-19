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
#include <stdlib.h>

/* Local includes. */
#include "swift.h"
#include "cosmology.h"
#include "rays_struct.h"
#include "rays.h"

/**
 * @brief Test the conservative pair kick of the kinetic SN feedback
 * (ray_kinetic_feedback_compute_kick_velocity, Chaikin et al. 2023).
 *
 * For random pairs of gas particles (masses, positions and velocities) around
 * a moving star, the two kicks must
 *  - change the total momentum by zero (eq. 13),
 *  - add exactly the pair energy to the kinetic energy (eq. 12),
 *  - point along the line joining the two particles (eq. 20), in opposite
 *    directions, pushing the particles apart along that line when each lies
 *    near its own ray, as the ray search selects them.
 */
int main(int argc, char *argv[]) {

  const int n_pairs = 100000;
  srand(20261020);

  struct cosmology cosmo;
  cosmology_init_no_cosmo(&cosmo);

  double max_dp = 0., max_dE = 0., max_angle = 0.;
  int n_not_apart = 0;

  for (int k = 0; k < n_pairs; k++) {

    /* The ray direction, drawn as in the feedback loops: cos(theta) and phi
     * uniform. The true particle lies near the ray and the mirror particle near
     * the opposite ray, as the arc-length search selects them; each is offset
     * from its ray by up to 0.3 of its distance from the star. */
    const double rand_theta = rand() / (double)RAND_MAX;
    const double rand_phi = rand() / (double)RAND_MAX;
    const double cos_t = 2. * rand_theta - 1., phi = 2. * M_PI * rand_phi - M_PI;
    const double sin_t = sqrt(1. - cos_t * cos_t);
    const double ray[3] = {sin_t * cos(phi), sin_t * sin(phi), cos_t};
    const double d_true = 0.2 + 0.8 * rand() / (double)RAND_MAX;
    const double d_mirr = 0.2 + 0.8 * rand() / (double)RAND_MAX;

    struct ray_data_extra ext_true, ext_mirr;
    float v_star[3];
    for (int i = 0; i < 3; i++) {
      ext_true.x[i] =
          d_true * (ray[i] + 0.3 * (2. * rand() / (double)RAND_MAX - 1.));
      ext_mirr.x[i] =
          -d_mirr * (ray[i] + 0.3 * (2. * rand() / (double)RAND_MAX - 1.));
      /* gas and star velocities up to 3 times the pair kick speed */
      ext_true.v[i] = 6.f * rand() / (float)RAND_MAX - 3.f;
      ext_mirr.v[i] = 6.f * rand() / (float)RAND_MAX - 3.f;
      v_star[i] = 2.f * rand() / (float)RAND_MAX - 1.f;
    }
    /* masses within a factor 10 of each other */
    const double m_true = pow(10., rand() / (double)RAND_MAX);
    const double m_mirr = pow(10., rand() / (double)RAND_MAX);
    const double E_pair = 0.5 * (m_true + m_mirr);

    float dv_true[3], dv_mirr[3], abs_true, abs_mirr;
    ray_kinetic_feedback_compute_kick_velocity(
        dv_true, &abs_true, &ext_true, &ext_mirr, ray_feedback_kinetic_true,
        E_pair, &cosmo, m_true, v_star, rand_theta, rand_phi, m_true, m_mirr);
    ray_kinetic_feedback_compute_kick_velocity(
        dv_mirr, &abs_mirr, &ext_true, &ext_mirr, ray_feedback_kinetic_mirr,
        E_pair, &cosmo, m_mirr, v_star, rand_theta, rand_phi, m_true, m_mirr);

    double dp[3], sep[3], dE = 0., sep2 = 0., kick2 = 0., dot = 0.;
    for (int i = 0; i < 3; i++) {
      dp[i] = m_true * dv_true[i] + m_mirr * dv_mirr[i];
      dE += 0.5 * m_true *
                ((ext_true.v[i] + dv_true[i]) * (ext_true.v[i] + dv_true[i]) -
                 ext_true.v[i] * ext_true.v[i]) +
            0.5 * m_mirr *
                ((ext_mirr.v[i] + dv_mirr[i]) * (ext_mirr.v[i] + dv_mirr[i]) -
                 ext_mirr.v[i] * ext_mirr.v[i]);
      sep[i] = ext_true.x[i] - ext_mirr.x[i];
      sep2 += sep[i] * sep[i];
      kick2 += dv_true[i] * dv_true[i];
      dot += dv_true[i] * sep[i];
    }

    /* momentum change relative to the momentum of one kick */
    const double p_kick = m_true * abs_true;
    const double rel_dp =
        sqrt(dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2]) / p_kick;
    const double rel_dE = fabs(dE / E_pair - 1.);
    /* angle between the true particle's kick and the separation vector */
    const double c = fabs(dot) / sqrt(sep2 * kick2);
    const double angle = acos(fmin(c, 1.));

    if (rel_dp > max_dp) max_dp = rel_dp;
    if (rel_dE > max_dE) max_dE = rel_dE;
    if (angle > max_angle) max_angle = angle;
    if (dot < 0.) n_not_apart++;
  }

  message("pairs: %d", n_pairs);
  message("max |sum m dv| / (m |dv|): %e", max_dp);
  message("max |dE / E_pair - 1|: %e", max_dE);
  message("max angle(kick, r_true - r_mirr) [rad]: %e", max_angle);
  message("pairs kicked towards each other: %d", n_not_apart);

  /* single-precision kicks and velocities */
  if (max_dp > 1e-5) error("Momentum not conserved: %e", max_dp);
  if (max_dE > 1e-3) error("Energy not conserved: %e", max_dE);
  if (max_angle > 1e-3) error("Kick not along the pair separation: %e", max_angle);
  if (n_not_apart > 0) error("%d pairs kicked towards each other", n_not_apart);

  return 0;
}
