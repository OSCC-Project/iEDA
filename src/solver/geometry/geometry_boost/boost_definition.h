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

#include <boost/geometry.hpp>
#include <boost/geometry/algorithms/detail/intersection/interface.hpp>
#include <boost/geometry/geometries/adapted/boost_polygon.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/multi_polygon.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/polygon/polygon.hpp>

#include "Boost.hpp"

namespace ieda_solver {
namespace gtl = boost::polygon;
using namespace boost::polygon::operators;
typedef int32_t CoordType;
typedef gtl::point_data<CoordType> GtlPoint;
typedef gtl::segment_data<CoordType> GtlSegment;
typedef gtl::rectangle_data<CoordType> GtlRect;
typedef gtl::polygon_90_data<CoordType> GtlPolygon90;
typedef gtl::polygon_90_with_holes_data<CoordType> GtlPolygon90WithHoles;
typedef gtl::polygon_90_set_data<CoordType> GtlPolygon90Set;
typedef gtl::polygon_data<CoordType> GtlPolygon;
typedef gtl::polygon_set_data<CoordType> GtlPolygonSet;
typedef gtl::polygon_with_holes_data<CoordType> GtlPolygonWithHoles;

namespace bg = boost::geometry;
typedef boost::geometry::model::d2::point_xy<CoordType> BgPoint;
typedef boost::geometry::model::segment<BgPoint> BgSegment;
typedef boost::geometry::model::box<BgPoint> BgRect;
typedef boost::geometry::model::polygon<BgPoint> BgPolygon;
typedef boost::geometry::model::multi_polygon<BgPolygon> BgMultiPolygon;

}  // namespace ieda_solver