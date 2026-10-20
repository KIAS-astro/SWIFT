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
#ifndef SWIFT_WIND_NONE_PROPERTIES_H
#define SWIFT_WIND_NONE_PROPERTIES_H

/**
 * @file src/wind/none/wind_properties.h
 * @brief No decoupled wind: no properties.
 */

#include "inline.h"

#include <stdio.h>

struct phys_const;
struct unit_system;
struct swift_params;
struct engine;

/**
 * @brief Properties of the wind model: none.
 */
struct wind_props {};

INLINE static void wind_props_init(struct wind_props *props,
                                   const struct phys_const *phys_const,
                                   const struct unit_system *us,
                                   struct swift_params *params) {}

INLINE static void wind_props_print(const struct wind_props *props) {}

INLINE static void wind_struct_dump(const struct wind_props *props,
                                    FILE *stream) {}

INLINE static void wind_struct_restore(struct wind_props *props,
                                       FILE *stream) {}

INLINE static void wind_print_stats(const struct engine *e) {}

INLINE static void wind_engine_pass(struct engine *e) {}

#endif /* SWIFT_WIND_NONE_PROPERTIES_H */
