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

typedef GtlPolygon90Set GeometryPolygonSet;

#define get_interact(polygon_set1, polygon_set2) gtl::interact(polygon_set1, polygon_set2)

#define getDefaultRectangles(output, polygon_set) gtl::get_rectangles(output, polygon_set)
#define getRectangles(output, polygon_set, direction) gtl::get_rectangles(output, polygon_set, direction)
#define getMaxRectangles(output, polygon_set) gtl::get_max_rectangles(output, polygon_set)
#define getPolygons(output, polygon_set) gtl::get_polygons(output, polygon_set)

#define ENVELOPE(rect, polygon_set) gtl::extents(rect, polygon_set)

#define growAnd(polygon_set, value) gtl::grow_and(polygon_set, value)
#define growAndHorizontal(polygon_set, value) gtl::grow_and(polygon_set, gtl::orientation_2d(HORIZONTAL), value)
#define growAndVertical(polygon_set, value) gtl::grow_and(polygon_set, gtl::orientation_2d(VERTICAL), value)

#define getArea(polygon_set) gtl::area(polygon_set)

}  // namespace ieda_solver
