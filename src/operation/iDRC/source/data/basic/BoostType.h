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
#ifndef IDRC_SRC_DB_BOOST_TYPE_H_
#define IDRC_SRC_DB_BOOST_TYPE_H_

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/polygon/polygon.hpp>

namespace idrc {
class DrcEdge;
class DrcRect;

using namespace boost::polygon::operators;
namespace bp = boost::polygon;
typedef boost::polygon::point_data<int> BoostPoint;
typedef boost::polygon::segment_data<int> BoostSegment;
typedef boost::polygon::polygon_90_data<int> BoostPolygon;
typedef boost::polygon::polygon_90_with_holes_data<int> PolygonWithHoles;
typedef boost::polygon::polygon_90_set_data<int> PolygonSet;
typedef boost::polygon::rectangle_data<int> BoostRect;

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
typedef bg::model::d2::point_xy<int, bg::cs::cartesian> RTreePoint;
typedef bg::model::segment<RTreePoint> RTreeSegment;
typedef bg::model::box<RTreePoint> RTreeBox;
typedef std::pair<RTreeSegment, DrcEdge*> rtree_edge_value;
typedef std::pair<RTreeBox, DrcRect*> rtree_rect_value;

typedef std::map<std::string, bgi::rtree<std::pair<RTreeBox, DrcRect*>, bgi::quadratic<16>>> LayerNameToRTreeMap;
typedef std::map<int, bgi::rtree<std::pair<RTreeBox, DrcRect*>, bgi::quadratic<16>>> LayerIdToRTreeMap;
typedef bgi::rtree<std::pair<RTreeBox, DrcRect*>, bgi::quadratic<16>> RectRTree;
}  // namespace idrc

#endif