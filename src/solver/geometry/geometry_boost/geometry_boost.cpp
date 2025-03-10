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

#include "geometry_boost.h"

#include <boost/geometry.hpp>

namespace ieda_solver {

GeometryBoost::GeometryBoost()
{
  _polyset.clean();
}

GeometryBoost::~GeometryBoost()
{
}

void GeometryBoost::addRect(int llx, int lly, int urx, int ury)
{
  GtlRect rect(llx, lly, urx, ury);

  _polyset += rect;
   gtl::bloat(rect,1);
  _polyset_overlap += rect;

  /// update bounding box
  updateBoundingBox(llx, lly, urx, ury);
}

bool GeometryBoost::isIntersect(int llx, int lly, int urx, int ury)
{
  GtlPolygon90Set self_set(_polyset);  /// self polyset
  self_set.clean();
  GtlPolygon90Set target_set;
  GtlRect rect(llx, lly, urx, ury);
  target_set += rect;

  auto& interact_poly = self_set.interact(target_set);

  //   std::vector<GeometryPolygon> overlap_list;
  //   interact_poly.get(overlap_list);
  return interact_poly.empty() ? false : true;

  //   GeometryPolygonSet target_set;
  //   GtlRect rect(llx - 1, lly - 1, urx + 1, ury + 1);
  //   target_set += rect;
  //   get_interact(target_set, _polyset);

  //   //   std::vector<GeometryPolygon> overlap_list;
  //   //   target_set.get(overlap_list);
  //   return target_set.empty() ? false : true;
}

/**
 * get points from polygon list
 * @param
 * std::pair<int, int> : define point x, y
 * std::vector<std::pair<int, int>> : define point list
 * std::vector<std::vector<std::pair<int, int>>> : define polygon list
 */
// std::vector<std::vector<std::pair<int, int>>> GeometryBoost::get_polygons_points()
// {
//   std::vector<std::vector<std::pair<int, int>>> polygons_points;

//   auto& polygon_list = get_polygon_list();
//   polygons_points.reserve(polygon_list.size());

//   for (auto& polygon : polygon_list) {
//     std::vector<std::pair<int, int>> pt_list;
//     for (auto& pt : polygon) {
//       pt_list.push_back(std::make_pair(pt.x(), pt.y()));
//     }
//     polygons_points.emplace_back(pt_list);
//   }

//   return polygons_points;
// }
/**
 * get boost points from polygon list
 * @param
 * uint64_t : total point number
 * GtlPoint : boost point
 * std::vector<GtlPoint> : point list for one polygon
 * std::vector<std::vector<GtlPoint>> : define all point list in the polygon list
 */
// std::pair<uint64_t, std::vector<std::vector<GtlPoint>>> GeometryBoost::get_boost_polygons_points()
// {
//   auto& polygon_list = get_polygon_list();

//   std::vector<std::vector<GtlPoint>> polygons_points;
//   polygons_points.reserve(polygon_list.size());
//   for (auto& polygon : polygon_list) {
//     polygons_points.emplace_back(polygon.coords_);
//   }

//   uint64_t number = _polyset.size();
//   /// return total number & point list number for all polygons
//   return std::make_pair(number, polygons_points);
// }

void GeometryBoost::addGeometry(EngineGeometry* geometry)
{
  if (geometry == nullptr) {
    return;
  }
  auto* boost_geometry = dynamic_cast<GeometryBoost*>(geometry);

  _polyset_overlap += boost_geometry->get_polyset_overlap();

  boost_geometry->get_polyset().clean();

  // _overlap_set += boost_geometry->get_polyset() & _polyset;

  _polyset += boost_geometry->get_polyset();
}

void GeometryBoost::addPolyset(GeometryPolygonSet& polyset)
{
  _polyset_overlap += polyset;
  
  polyset.clean();

  _polyset += polyset;
}

std::vector<GeometryPolygon>& GeometryBoost::getLayoutPolygons()
{
  if (_polygon_list.empty()) {
    _polyset.clean();
    _polyset.get(_polygon_list);
  }
  return _polygon_list;
}

void GeometryBoost::initPolygonRTree()
{
  //   auto& polygon_list = getLayoutPolygons();
  //   for (auto polygon : polygon_list) {
  //     ieda_solver::GeometryRect rect;
  //     ieda_solver::ENVELOPE(rect, polygon);

  //     ieda_solver::BgRect rtree_rect(gtl::ll(rect).x(), gtl::ll(rect).y(), gtl::ur(rect).x(), gtl::ur(rect).y());
  //     _polygon_rtree.insert(std::make_pair(rtree_rect, polygon));
  //   }
}

std::vector<GeometryPolygon> GeometryBoost::getOverlap(EngineGeometry* other)
{
  std::vector<GeometryPolygon> overlap_list;

  if (other == nullptr) {
    /// check self overlap
    GtlPolygon90Set set(_polyset);
    auto& intersets = set.self_intersect();
    intersets.get(overlap_list);
  } else {
    /// check overlap with other geometry
    GtlPolygon90Set self_set(_polyset);  /// self polyset

    auto* boost_geometry = dynamic_cast<GeometryBoost*>(other);
    auto& interact_poly = self_set.interact(boost_geometry->get_polyset());
    interact_poly.get(overlap_list);
  }

  return overlap_list;
}

std::vector<GeometryRect>& GeometryBoost::getWires()
{
  if (!_wires_initialized) {
    gtl::get_max_rectangles(_wire_list, _polyset);
    _wires_initialized = true;
  }
  return _wire_list;
}

std::vector<GeometryRect>& GeometryBoost::getRects()
{
  if (!_rect_initialized) {
    _polyset.clean();
    gtl::get_rectangles(_rect_list, _polyset);
    _rect_initialized = true;
  }
  return _rect_list;
}

std::vector<GeometryRect> GeometryBoost::getRectsGrowAnd(int value, GeometryOrientation direction)
{
  std::vector<GeometryRect> grow_rects;

  GeometryPolygonSet polyset;

  auto& polygon_list = getLayoutPolygons();
  for (auto& polygon : polygon_list) {
    GeometryPolygonSet polyset_temp;
    std::vector<GeometryRect> poly_rects;
    gtl::get_rectangles(poly_rects, polygon);
    for (auto rect : poly_rects) {
      bloat(rect, direction, value);
      polyset_temp += rect;
    }

    polyset_temp.clean();

    polyset += polyset_temp;
  }

  polyset.self_intersect();
  polyset.get(grow_rects);

  return grow_rects;
}

int64_t GeometryBoost::getMergeRectArea(int llx, int lly, int urx, int ury)
{
  int64_t area;

  GtlPolygon90Set set(_polyset);
  GtlRect rect(llx, lly, urx, ury);

  set += rect;

  area = (int64_t) getArea(set);

  return area;
}
}  // namespace ieda_solver