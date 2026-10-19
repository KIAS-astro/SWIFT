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
#ifndef SWIFT_WIND_NONE_STRUCT_H
#define SWIFT_WIND_NONE_STRUCT_H

/**
 * @file src/wind/none/wind_struct.h
 * @brief No decoupled wind: no particle data.
 */

/**
 * @brief Wind data carried by a gas particle: none.
 */
struct wind_part_data {};

/**
 * @brief Wind-carrier data carried by a star particle: none.
 */
struct wind_spart_data {};

#endif /* SWIFT_WIND_NONE_STRUCT_H */
