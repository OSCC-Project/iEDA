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
/**
 * @file GeomOperator.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "GeomOperator.hh"

#include "log/Log.hh"
namespace icts {
Point GeomOperator::bgToPglPoint(const bg_Point& p)
{
  return Point(p.x(), p.y());
}

Segment GeomOperator::bgToPglSegment(const bg_Segment& s)
{
  return Segment(bgToPglPoint(s.front()), bgToPglPoint(s.back()));
}

Polygon GeomOperator::bgToPglPolygon(const bg_Polygon& p)
{
  Polygon poly;
  for (auto& point : p.outer()) {
    poly.add_point(bgToPglPoint(point));
  }
  return poly;
}

bg_Point GeomOperator::pglToBgPoint(const Point& p)
{
  return bg_Point(p.x(), p.y());
}

bg_Segment GeomOperator::pglToBgSegment(const Segment& s)
{
  return bg_Segment({pglToBgPoint(s.low()), pglToBgPoint(s.high())});
}

bg_Polygon GeomOperator::pglToBgPolygon(const Polygon& p)
{
  bg_Polygon bg_poly;
  for (auto& point : p.get_points()) {
    bg_poly.outer().push_back(pglToBgPoint(point));
  }
  if (!bg::is_valid(bg_poly)) {
    bg::correct(bg_poly);
  }
  return bg_poly;
}

std::vector<Point> GeomOperator::intersectionPointByBg(const Polygon& poly_a, const Polygon& poly_b)
{
  auto bg_poly_a = pglToBgPolygon(poly_a);
  auto bg_poly_b = pglToBgPolygon(poly_b);

  std::vector<bg_Point> intersection;
  bg::intersection(bg_poly_a, bg_poly_b, intersection);
  std::vector<Point> points;
  for (auto point : intersection) {
    points.push_back(Point(point.x(), point.y()));
  }
  return points;
}

std::vector<Point> GeomOperator::intersectionPointByBg(const Polygon& poly, const Segment& seg)
{
  auto bg_poly = pglToBgPolygon(poly);
  auto bg_seg = pglToBgSegment(seg);

  std::vector<bg_Point> intersection;
  bg::intersection(bg_poly, bg_seg, intersection);
  std::vector<Point> points;
  for (auto point : intersection) {
    points.push_back(Point(point.x(), point.y()));
  }
  return points;
}

Point GeomOperator::intersectionPointByBg(const Segment& seg_a, const Segment& seg_b)
{
  auto bg_seg_a = pglToBgSegment(seg_a);
  auto bg_seg_b = pglToBgSegment(seg_b);
  std::vector<bg_Point> intersection;
  bg::intersection(bg_seg_a, bg_seg_b, intersection);
  if (intersection.empty()) {
    return Point(0, 0);
  }
  return Point(intersection.front().x(), intersection.front().y());
}

Polygon GeomOperator::intersectionByBg(const Polygon& poly_a, const Polygon& poly_b)
{
  if (isPoint(poly_a) || isPoint(poly_b)) {
    auto a_points = poly_a.get_points();
    auto b_points = poly_b.get_points();
    auto a_point = a_points.front();
    auto b_point = b_points.front();
    if (a_point == b_point) {
      return Polygon({a_point});
    } else {
      return Polygon({});
    }
  }
  auto bg_poly_a = pglToBgPolygon(poly_a);
  auto bg_poly_b = pglToBgPolygon(poly_b);
  std::vector<bg_Polygon> intersection;
  bg::intersection(bg_poly_a, bg_poly_b, intersection);
  if (intersection.empty()) {
    auto is_points = intersectionPointByBg(poly_a, poly_b);
    return Polygon(is_points);
  }
  Polygon poly;
  for (auto point : intersection.front().outer()) {
    poly.add_point(Point(point.x(), point.y()));
  }
  return poly;
}

Polygon GeomOperator::intersectionByBg(const Polygon& poly, const Segment& seg)
{
  auto bg_poly = pglToBgPolygon(poly);
  auto bg_seg = pglToBgSegment(seg);
  std::vector<bg_Point> intersection;
  bg::intersection(bg_poly, bg_seg, intersection);
  Polygon is_poly;
  for (auto& point : intersection) {
    is_poly.add_point(Point(point.x(), point.y()));
  }
  return is_poly;
}

Segment GeomOperator::intersectJS(const Segment& js_i, const Segment& js_j, const int& radius_by_j)
{
  if (radius_by_j == 0) {
    return js_j;
  }
  if (radius_by_j == pgl::manhattan_distance(js_i, js_j)) {
    return js_i;
  }
  if (js_i.low() == js_i.high() && js_j.low() == js_j.high() && pgl::rectilinear(js_i.low(), js_j.low())) {
    Point point;
    if (js_i.low().x() == js_j.low().x()) {
      point = Point(js_i.low().x(),
                    js_j.low().y() + radius_by_j * ((js_i.low().y() - js_j.low().y()) / std::abs((js_i.low().y() - js_j.low().y()))));
      return Segment(point, point);
    }
    point = Point(js_j.low().x() + radius_by_j * ((js_i.low().x() - js_j.low().x()) / std::abs((js_i.low().x() - js_j.low().x()))),
                  js_j.low().y());
    return Segment(point, point);
  }

  Polygon poly_j;
  pgl::tilted_rect_region(poly_j, js_j, radius_by_j);
  Polygon sdr;
  calcSDR(sdr, js_i, js_j);
  auto is_points = intersectionPointByBg(sdr, poly_j);
  if (is_points.size() > 2) {
    // it means sdr radius plus 1, should remove some points
    auto dist = pgl::manhattan_distance(js_i, js_j);
    for (auto itr = is_points.begin(); itr != is_points.end();) {
      if (pgl::manhattan_distance(*itr, js_i) > dist) {
        is_points.erase(itr);
      } else {
        ++itr;
      }
    }
  }
  auto js = Segment(is_points[0], is_points[1]);
  return pgl::fixJoinSegment(js);
}

void GeomOperator::calcSDR(Polygon& sdr, const Segment& seg_i, const Segment& seg_j)
{
  if (seg_i == seg_j) {
    sdr = seg_i.low() == seg_i.high() ? Polygon({seg_i.low()}) : Polygon({seg_i.low(), seg_i.high()});
    return;
  }
  if (seg_i.low() == seg_i.high() && seg_j.low() == seg_j.high() && pgl::rectilinear(seg_i.low(), seg_j.low())) {
    sdr = Polygon({seg_i.low(), seg_j.low()});
    return;
  }
  Rectangle bbox;
  std::vector<Point> points{seg_i.low(), seg_i.high(), seg_j.low(), seg_j.high()};
  pgl::convex_hull(points);
  pgl::extents(bbox, points);

  auto box = Polygon({bbox.low(), Point(bbox.low().x(), bbox.high().y()), bbox.high(), Point(bbox.high().x(), bbox.low().y()), bbox.low()});
  if (seg_i.low() == seg_i.high() && seg_j.low() == seg_j.high()) {
    sdr = box;
    return;
  }
  CtsPolygon<int64_t> trr_a, trr_b;
  auto radius = pgl::manhattan_distance(seg_i, seg_j);
  if (radius == 0) {
    auto point = intersectionPointByBg(seg_i, seg_j);
    sdr = Polygon({point});
    return;
  }

  pgl::tilted_rect_region(trr_a, seg_i, radius);
  pgl::tilted_rect_region(trr_b, seg_j, radius);

  auto first_region = intersectionByBg(box, trr_a);
  sdr = intersectionByBg(first_region, trr_b);
  LOG_FATAL_IF(sdr.empty()) << "sdr is empty";
}

bool GeomOperator::isPoint(const Polygon& poly)
{
  auto points = poly.get_points();
  auto unique_points = std::set<Point>(points.begin(), points.end());
  return unique_points.size() == 1;
}

bool GeomOperator::isSegment(const Polygon& poly)
{
  auto points = poly.get_points();
  auto unique_points = std::set<Point>(points.begin(), points.end());
  return unique_points.size() == 2;
}

}  // namespace icts