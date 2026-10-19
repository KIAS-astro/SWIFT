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
#ifndef SWIFT_LIGHT_PHYSICS_H
#define SWIFT_LIGHT_PHYSICS_H

/**
 * @file src/light_physics.h
 * @brief Light wind particles: the pure functions of spawning, the
 * refinement clock and merging (light-particles-design.md Sec. 8).
 *
 * Spawning: at every tree rebuild a parent holding a reservoir m_pending
 * spawns n light particles of mass m_l, n being m_pending / m_l rounded at
 * random to one of the two nearest integers with the expected value kept,
 * limited so that it keeps at least m_l itself; the reservoir is then
 * emptied.
 * Clock (Bennett et al. 2025, eqs. 6, 12, 13): at recoupling the dye is
 *   f_w = f_thresh exp[0.2 R_200 / (k v_B t_0)]
 * and it decays as exp(-dt / t_0); the target mass is m_l while
 * f_w >= f_thresh and max[m_l, m_g (f_ref / f_w)^1/2] below.
 * Merging of two particles a and b: mass, momentum, metal mass and total
 * (kinetic + internal) energy are conserved; the kinetic energy of the
 * relative motion, (1/2) mu |v_a - v_b|^2 with mu = m_a m_b / (m_a + m_b),
 * becomes internal energy.
 */

#include <math.h>

/**
 * @brief Number of light particles a parent spawns from its reservoir.
 *
 * With x = m_pending / m_l, n = floor(x) plus one with probability
 * x - floor(x) (Sullivan et al. 2026, eq. 9), so that the expected spawned
 * mass equals the reservoir and no reservoir waits to fill a whole m_l. The
 * caller empties the reservoir after every draw: carrying the remainder
 * (x - n) m_l to the next draw would bias the spawned mass upwards, since a
 * negative remainder cannot spawn a negative number. n is limited so that
 * the parent keeps at least m_l.
 *
 * @param m_parent Mass of the parent, reservoir included.
 * @param m_pending Mass in the reservoir.
 * @param m_l Mass of a light particle.
 * @param u Uniform number in [0, 1).
 * @return n >= 0 with m_parent - n m_l >= m_l (or n = 0).
 */
static inline long long light_spawn_count(const double m_parent,
                                          const double m_pending,
                                          const double m_l, const double u) {
  if (m_pending <= 0. || m_parent < 2. * m_l) return 0;
  const double x = m_pending / m_l;
  const double fl = floor(x);
  const long long n_res = (long long)fl + (u < x - fl ? 1 : 0);
  const long long n_keep = (long long)floor((m_parent - m_l) / m_l);
  return n_res < n_keep ? n_res : n_keep;
}

/**
 * @brief Does a parent too light to spawn (m_parent < 2 m_l) launch itself?
 *
 * Such a parent (typically a recoupled light particle back in star-forming
 * gas) launches whole with probability m_pending / m_parent, so that the
 * expected launched mass still equals its reservoir.
 *
 * @param u Uniform number in [0, 1), the one given to light_spawn_count().
 */
static inline int light_launch_whole(const double m_parent,
                                     const double m_pending, const double m_l,
                                     const double u) {
  if (m_pending <= 0. || m_parent >= 2. * m_l) return 0;
  return u < m_pending / m_parent;
}

/**
 * @brief Dye of a light particle at recoupling (Bennett et al. 2025, eq. 12).
 *
 * The particle stays above f_thresh for the time 0.2 R_200 / (k v_B) to cross
 * 0.2 R_200 at k v_B.
 */
static inline double light_dye_init(const double f_thresh, const double R200,
                                    const double k, const double v_B,
                                    const double t0) {
  return f_thresh * exp(0.2 * R200 / (k * v_B * t0));
}

/**
 * @brief Target mass of a light particle with dye f_w (Bennett et al. 2025,
 * eq. 13 with m_0 / 100 replaced by m_l).
 */
static inline double light_target_mass(const double f_w, const double m_l,
                                       const double m_g, const double f_thresh,
                                       const double f_ref) {
  if (f_w >= f_thresh) return m_l;
  if (f_w <= f_ref) return m_g;
  const double m = m_g * sqrt(f_ref / f_w);
  return m > m_l ? m : m_l;
}

/**
 * @brief Merge particle b into particle a.
 *
 * Positions must be on one periodic image. Velocities and internal energies
 * are per unit mass.
 *
 * @param m_a, x_a, v_a, u_a (in/out) Particle a, replaced by the merged
 * particle.
 * @param m_b, x_b, v_b, u_b Particle b.
 * @return The kinetic energy of the relative motion turned into heat.
 */
static inline double light_merge_state(double *m_a, double x_a[3],
                                       double v_a[3], double *u_a,
                                       const double m_b, const double x_b[3],
                                       const double v_b[3], const double u_b) {
  const double ma = *m_a;
  const double M = ma + m_b;
  double dv2 = 0.;
  for (int k = 0; k < 3; k++) {
    const double dv = v_a[k] - v_b[k];
    dv2 += dv * dv;
    x_a[k] = (ma * x_a[k] + m_b * x_b[k]) / M;
    v_a[k] = (ma * v_a[k] + m_b * v_b[k]) / M;
  }
  const double heat = 0.5 * ma * m_b / M * dv2;
  *u_a = (ma * *u_a + m_b * u_b + heat) / M;
  *m_a = M;
  return heat;
}

#endif /* SWIFT_LIGHT_PHYSICS_H */
