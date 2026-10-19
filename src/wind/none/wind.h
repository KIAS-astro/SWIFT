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
#ifndef SWIFT_WIND_NONE_H
#define SWIFT_WIND_NONE_H

/**
 * @file src/wind/none/wind.h
 * @brief No decoupled wind. Every function is empty or returns the value
 * that leaves the code path of a run without wind unchanged.
 */

#include "inline.h"
#include "timeline.h"

#include <float.h>

struct part;
struct xpart;
struct spart;
struct cosmology;
struct hydro_props;
struct wind_props;

/**
 * @brief Is this gas particle decoupled from the hydrodynamics? Never.
 */
__attribute__((always_inline)) INLINE static int part_is_decoupled(
    const struct part *p) {
  return 0;
}

INLINE static void wind_first_init_part(struct part *p, struct xpart *xp,
                                        const struct wind_props *props,
                                        const double time) {}

/**
 * @brief Does a star form? The comparison of the star formation modules,
 * u < p_*.
 *
 * @return 1 for a star, 0 otherwise.
 */
__attribute__((always_inline)) INLINE static int wind_star_formation_draw(
    const struct wind_props *props, const struct part *p,
    const struct xpart *xp, const double p_star, const double u) {
  return p_star > u;
}

INLINE static int wind_should_launch(const struct part *p,
                                     const struct xpart *xp,
                                     const struct wind_props *props,
                                     const integertime_t ti_current,
                                     const double dt_star) {
  return 0;
}

INLINE static void wind_launch(struct part *p, struct xpart *xp,
                               const struct wind_props *props,
                               const struct cosmology *cosmo,
                               const double time,
                               const integertime_t ti_current) {}

INLINE static int wind_should_recouple(const struct part *p,
                                       const struct wind_props *props,
                                       const double time) {
  return 0;
}

INLINE static void wind_recouple(struct part *p, struct xpart *xp,
                                 const struct wind_props *props,
                                 const struct cosmology *cosmo,
                                 const double time) {}

__attribute__((always_inline)) INLINE static void wind_drift_part(
    struct part *p, const struct xpart *xp, const double dt_drift) {}

INLINE static float wind_timestep(const struct wind_props *props) {
  return FLT_MAX;
}

/* Wind carriers: none */

__attribute__((always_inline)) INLINE static int spart_is_wind_carrier(
    const struct spart *sp) {
  return 0;
}

__attribute__((always_inline)) INLINE static int wind_feedback_is_active(
    const struct spart *sp) {
  return 1;
}

INLINE static void wind_first_init_spart(struct spart *sp,
                                         const struct wind_props *props,
                                         const double time) {}

INLINE static int wind_number_of_carriers(const struct part *p,
                                          const struct xpart *xp,
                                          const struct wind_props *props,
                                          const integertime_t ti_current,
                                          const double dt_star, double *m_c,
                                          int *whole) {
  *m_c = 0.;
  *whole = 0;
  return 0;
}

INLINE static void wind_carrier_launch(
    struct spart *sp, struct part *p, struct xpart *xp,
    const struct wind_props *props, const struct cosmology *cosmo,
    double m_c, const int whole, const double time,
    const integertime_t ti_current, const int index) {}

INLINE static void wind_carrier_launch_failed(void) {}

__attribute__((always_inline)) INLINE static void wind_carrier_drift(
    struct spart *sp, const double dt_drift, const struct wind_props *props,
    const double time) {}

__attribute__((always_inline)) INLINE static void wind_carrier_init_spart(
    struct spart *sp) {}

__attribute__((always_inline)) INLINE static void wind_carrier_iact_density(
    const float r2, const float hi, struct spart *si) {}

INLINE static void wind_carrier_iact_apply(
    const float r2, const float dx[3], const float hi, const struct spart *si,
    struct part *pj, struct xpart *xpj, const struct cosmology *cosmo,
    const struct hydro_props *hydro_props) {}

INLINE static int wind_carrier_end_step(struct spart *sp, const double time) {
  return 0;
}

#endif /* SWIFT_WIND_NONE_H */
