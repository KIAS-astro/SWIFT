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
#ifndef SWIFT_WIND_NONE_IO_H
#define SWIFT_WIND_NONE_IO_H

/**
 * @file src/wind/none/wind_io.h
 * @brief No decoupled wind: nothing to read or write.
 */

#include "io_properties.h"

INLINE static int wind_read_particles(struct part *parts,
                                      struct io_props *list) {
  return 0;
}

INLINE static int wind_write_particles(const struct part *parts,
                                       const struct xpart *xparts,
                                       struct io_props *list) {
  return 0;
}

#endif /* SWIFT_WIND_NONE_IO_H */
