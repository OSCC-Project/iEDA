// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#pragma once

#include "boost_definition.h"

namespace ieda_solver {

typedef gtl::direction_1d GeometryDirection1D;
typedef gtl::direction_2d GeometryDirection2D;
typedef gtl::orientation_2d GeometryOrientation;
typedef gtl::interval_data<int> GeometryInterval;

#define K_HORIZONTAL gtl::HORIZONTAL
#define K_VERTICAL gtl::VERTICAL
#define WEST gtl::WEST
#define EAST gtl::EAST
#define NORTH gtl::NORTH
#define SOUTH gtl::SOUTH

typedef GtlRect GeometryRect;

#define getWireDirection(wire) gtl::guess_orientation(wire)

#define getWireWidth(wire, dir) gtl::delta(wire, dir)

#define BLOAT(rect_to_change, direction, value) gtl::bloat(rect_to_change, direction, value)
#define SHRINK(rect_to_change, direction, value) gtl::shrink(rect_to_change, direction, value)

#define rectManhattanDistance(rect1, rect2) gtl::manhattan_distance(rect1, rect2)
#define rectEuclideanDistance(rect1, rect2) gtl::square_euclidean_distance(rect1, rect2)

#define oppositeRegion(rect1, rect2) gtl::generalized_intersect(rect1, rect2)

#define lowLeftX(rect_to_get_property) gtl::xl(rect_to_get_property)
#define lowLeftY(rect_to_get_property) gtl::yl(rect_to_get_property)
#define upRightX(rect_to_get_property) gtl::xh(rect_to_get_property)
#define upRightY(rect_to_get_property) gtl::yh(rect_to_get_property)

}  // namespace ieda_solver
