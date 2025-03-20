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

#include <boost/foreach.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/polygon/polygon.hpp>

namespace gtl = boost::polygon;
using namespace boost::polygon::operators;
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

using GTLPointInt = gtl::point_data<int32_t>;
using GTLRectInt = gtl::rectangle_data<int32_t>;
using GTLPolyInt = gtl::polygon_90_data<int32_t>;
using GTLHolePolyInt = gtl::polygon_90_with_holes_data<int32_t>; 
using GTLPolySetInt = gtl::polygon_90_set_data<int32_t>;

using BGPointInt = bg::model::d2::point_xy<int32_t>;
using BGMultiPointInt = bg::model::multi_point<BGPointInt>;
using BGSegmentInt = bg::model::segment<BGPointInt>;
using BGLineInt = bg::model::linestring<BGPointInt>;
using BGMultiLineInt = bg::model::multi_linestring<BGLineInt>;
using BGRectInt = bg::model::box<BGPointInt>;
using BGPolyInt = bg::model::polygon<BGPointInt>;
using BGMultiPolyInt = bg::model::multi_polygon<BGPolyInt>;

using BGPointDBL = bg::model::d2::point_xy<double>;
using BGMultiPointDBL = bg::model::multi_point<BGPointDBL>;
using BGSegmentDBL = bg::model::segment<BGPointDBL>;
using BGLineDBL = bg::model::linestring<BGPointDBL>;
using BGMultiLineDBL = bg::model::multi_linestring<BGLineDBL>;
using BGRectDBL = bg::model::box<BGPointDBL>;
using BGPolyDBL = bg::model::polygon<BGPointDBL>;
using BGMultiPolyDBL = bg::model::multi_polygon<BGPolyDBL>;

