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

#define HORIZONTAL gtl::HORIZONTAL
#define VERTICAL gtl::VERTICAL
#define WEST gtl::WEST
#define EAST gtl::EAST
#define NORTH gtl::NORTH
#define SOUTH gtl::SOUTH

typedef GtlRect GeometryRect;

#define getWireDirection(wire) gtl::guess_orientation(wire)

#define getWireWidth(wire, dir) gtl::delta(wire, dir)

inline std::array<GeometryRect, 2> getExpandRects(const GeometryRect& rect, int within, GeometryOrientation dir)
{
  auto interval_wire = rect.get(dir.get_perpendicular());
  auto interval_within = rect.get(dir);
  GeometryInterval interval_high(gtl::get(interval_within, gtl::HIGH), gtl::get(interval_within, gtl::HIGH) + within);
  GeometryInterval interval_low(gtl::get(interval_within, gtl::LOW) - within, gtl::get(interval_within, gtl::LOW));
  GeometryRect rect_high;
  rect_high.set(dir, interval_high);
  rect_high.set(dir.get_perpendicular(), interval_wire);
  GeometryRect rect_low;
  rect_low.set(dir, interval_low);
  rect_low.set(dir.get_perpendicular(), interval_wire);

  return {rect_high, rect_low};
}

#define bloat(rect, direction, value) gtl::bloat(rect, direction, value)
#define shrink(rect, direction, value) gtl::shrink(rect, direction, value)

#define manhattanDistance(rect1, rect2) gtl::manhattan_distance(rect1, rect2)

#define oppositeRegion(rect1, rect2) gtl::generalized_intersect(rect1, rect2)

}  // namespace ieda_solver
