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
#ifndef SWIFT_WIND_H
#define SWIFT_WIND_H

/**
 * @file src/wind.h
 * @brief Branches between the decoupled-wind models.
 */

/* Config parameters. */
#include <config.h>

#if defined(WIND_NONE)
#include "./wind/none/wind.h"
#elif defined(WIND_TIGRESS)
#include "./wind/TIGRESS/wind.h"
#else
#error "Invalid choice of wind model."
#endif

#endif /* SWIFT_WIND_H */
