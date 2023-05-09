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

#include <algorithm>
#include <cassert>
#include <limits>
#include <stack>
#include <vector>

#include "pgl.h"

namespace icts {
namespace pgl {
template <typename T>
using Pair = std::pair<T, T>;

// compute the cross product of two vectors that starting at the origin.
template <typename T>
static inline double cross_product(const CtsPoint<T> &lhs,
                                   const CtsPoint<T> &rhs) {
  return 1.0 * static_cast<double>(lhs.x()) * static_cast<double>(rhs.y()) -
         1.0 * static_cast<double>(lhs.y()) * static_cast<double>(rhs.x());
}

// calculate the cross product of two vector.
template <typename T>
static inline double cross_product(const CtsPoint<T> &start1,
                                   const CtsPoint<T> &end1,
                                   const CtsPoint<T> &start2,
                                   const CtsPoint<T> &end2) {
  return cross_product(end1 - start1, end2 - start2);
}

template <typename T>
static inline double cross_product(const CtsSegment<T> &vec1,
                                   const CtsSegment<T> &vec2) {
  return cross_product(vec1.high() - vec1.low(), vec2.high() - vec2.low());
}

template <typename T>
static inline double cross_product(const CtsPoint<T> &p1, const CtsPoint<T> &p2,
                                   const CtsPoint<T> &p3) {
  return cross_product(p2 - p1, p3 - p1);
}

template <typename T>
inline double dot_product(const CtsPoint<T> &lhs, const CtsPoint<T> &rhs) {
  return 1.0 * lhs.x() * rhs.x() + 1.0 * lhs.y() * rhs.y();
}
template <typename T>
inline double dot_product(const CtsPoint<T> &start1, const CtsPoint<T> &end1,
                          const CtsPoint<T> &start2, const CtsPoint<T> &end2) {
  return dot_product(end1 - start1, end2 - start2);
}
template <typename T>
static inline double dot_product(const CtsSegment<T> &vec1,
                                 const CtsSegment<T> &vec2) {
  return dot_product(vec1.high() - vec1.low(), vec2.high() - vec2.low());
}

template <typename T>
static inline CtsInterval<T> interval(const CtsSegment<T> &seg,
                                      gtl::orientation_2d orient) {
  return CtsInterval<T>(seg.low().get(orient), seg.high().get(orient));
}

template <typename T>
static inline bool longest_segment(CtsSegment<T> &longest,
                                   const std::vector<CtsSegment<T>> &segs) {
  if (segs.empty()) {
    return false;
  }
  longest = segs[0];
  for (auto &seg : segs) {
    if (gtl::euclidean_distance(seg.low(), seg.high()) >
        gtl::euclidean_distance(longest.low(), longest.high())) {
      longest = seg;
    }
  }
  return true;
}

template <typename T>
inline bool integral_coord(const T &coord) {
  return coord == static_cast<int>(coord);
}

template <typename T>
inline bool integral_coord(const CtsPoint<T> &point) {
  return integral_coord(point.x()) && integral_coord(point.y());
}

// intersect of two geometric objects.
template <typename T>
inline bool intersect(CtsSegment<T> &res, const CtsSegment<T> &seg1,
                      const CtsSegment<T> &seg2) {
  if (gtl::intersects(seg1, seg2) == false) {
    return false;
  }
  if (0 == cross_product(seg1.low(), seg1.high(), seg2.low(), seg2.high())) {
    std::vector<CtsPoint<T>> points = {seg1.low(), seg1.high(), seg2.low(),
                                       seg2.high()};
    std::sort(points.begin(), points.end());
    res.low(points[1]);
    res.high(points[2]);
    return true;
  }

  double area_l =
      cross_product(seg2.low(), seg2.high(), seg2.low(), seg1.low());
  double area_r =
      cross_product(seg2.low(), seg2.high(), seg2.low(), seg1.high());
  double area_ratio = std::abs(area_l) / (std::abs(area_l) + std::abs(area_r));
  double x = seg1.low().x() + (seg1.high().x() - seg1.low().x()) * area_ratio;
  double y = seg1.low().y() + (seg1.high().y() - seg1.low().y()) * area_ratio;
  CtsPoint<T> point(x, y);
  res.low(point);
  res.high(point);
  return true;
}

template <typename T>
static inline bool bound_intersects(std::vector<CtsSegment<T>> &inter_set,
                                    const CtsPolygon<T> &lhs,
                                    const CtsPolygon<T> &rhs) {
  auto segs_l = lhs.get_edges();
  auto segs_r = rhs.get_edges();
  CtsSegment<T> seg;

  bool is_intersect = false;
  for (auto &seg_l : segs_l) {
    for (auto &seg_r : segs_r) {
      if (intersect(seg, seg_l, seg_r)) {
        inter_set.emplace_back(seg);
        is_intersect = true;
      }
    }
  }
  return is_intersect;
}

template <typename T>
static inline bool bound_intersects(std::vector<CtsSegment<T>> &res,
                                    const CtsSegment<T> &seg,
                                    const CtsPolygon<T> &poly) {
  auto edges = poly.get_edges();
  bool is_intersect = false;
  CtsSegment<T> ans;
  for (auto &edge : edges) {
    if (intersect(ans, seg, edge)) {
      res.emplace_back(ans);
      is_intersect = true;
    }
  }
  return is_intersect;
}

template <typename T>
static inline bool bound_intersect(CtsSegment<T> &res, const CtsSegment<T> &seg,
                                   const CtsPolygon<T> &poly) {
  std::vector<CtsSegment<T>> segs;

  bool ret = pgl::bound_intersects(segs, seg, poly);
  longest_segment(res, segs);
  return ret;
}

template <typename T>
static inline bool bound_intersect(CtsSegment<T> &seg, const CtsPolygon<T> &lhs,
                                   const CtsPolygon<T> &rhs) {
  std::vector<CtsSegment<T>> segs;

  bool is_intersect = bound_intersects(segs, lhs, rhs);
  pgl::longest_segment(seg, segs);
  return is_intersect;
}

// calculate the bounding box of segment
template <typename T>
static inline CtsRectangle<T> extends(const CtsSegment<T> &seg) {
  T min_x = std::min(seg.low().x(), seg.high().x());
  T min_y = std::min(seg.low().y(), seg.high().y());
  T max_x = std::max(seg.low().x(), seg.high().x());
  T max_y = std::max(seg.low().y(), seg.high().y());
  CtsRectangle<T> rect(min_x, min_y, max_x, max_y);
  return rect;
}

template <typename T>
static inline CtsRectangle<T> extends(const CtsSegment<T> &seg1,
                                      const CtsSegment<T> &seg2) {
  std::vector<CtsPoint<T>> points{seg1.low(), seg1.high(), seg2.low(),
                                  seg2.high()};
  T min_x = std::numeric_limits<T>::max();
  T min_y = std::numeric_limits<T>::max();
  T max_x = std::numeric_limits<T>::min();
  T max_y = std::numeric_limits<T>::min();
  for (auto &point : points) {
    min_x = std::min(min_x, point.x());
    min_y = std::min(min_y, point.y());
    max_x = std::max(max_x, point.x());
    max_y = std::max(max_y, point.y());
  }
  CtsRectangle<T> rect(min_x, min_y, max_x, max_y);
  return rect;
}

template <typename T>
static inline CtsRectangle<T> extends(const CtsPolygon<T> &poly) {
  std::vector<CtsPoint<T>> points = poly.get_points();
  T min_x = std::numeric_limits<T>::max();
  T min_y = std::numeric_limits<T>::max();
  T max_x = std::numeric_limits<T>::min();
  T max_y = std::numeric_limits<T>::min();
  for (auto &point : points) {
    min_x = std::min(min_x, point.x());
    min_y = std::min(min_y, point.y());
    max_x = std::max(max_x, point.x());
    max_y = std::max(max_y, point.y());
  }
  CtsRectangle<T> rect(min_x, min_y, max_x, max_y);
  return rect;
}

template <typename T>
static inline void extents(CtsInterval<T> &interval, const T &coord) {
  T low = coord;
  T high = coord;
  if (coord != static_cast<int>(coord)) {
    low = std::floor(coord);
    high = std::ceil(coord);
  } else {
    low = coord - 1;
    high = coord + 1;
  }
  interval.low(low);
  interval.high(high);
}

template <typename T>
static inline void extents(CtsRectangle<T> &rect, const CtsPoint<T> &point) {
  CtsInterval<T> x_interval, y_interval;
  extents(x_interval, point.x());
  extents(y_interval, point.y());
  rect = CtsRectangle<T>(x_interval, y_interval);
}

template <typename T>
static inline bool bound_integral_points(std::vector<CtsPoint<T>> &points,
                                         const CtsRectangle<T> &rect) {
  if (!integral_coord(rect.low()) || !integral_coord(rect.high())) {
    assert(0);
  }

  CtsInterval<T> x_interval = rect.get(gtl::HORIZONTAL);
  CtsInterval<T> y_interval = rect.get(gtl::VERTICAL);
  for (T coord_x = x_interval.low(); coord_x <= x_interval.high(); coord_x++) {
    for (T coord_y = y_interval.low(); coord_y <= y_interval.high();
         coord_y++) {
      points.emplace_back(Point(coord_x, coord_y));
    }
  }
  return true;
}

// compute the slope of a line
template <typename T>
static inline double slope(const CtsSegment<T> &seg) {
  if (seg.low().x() == seg.high().x()) {
    return std::numeric_limits<T>::max();
  }
  return 1.0 * (seg.high().y() - seg.low().y()) /
         (seg.high().x() - seg.low().x());
}

// linear functions
template <typename T>
static inline void linearFunction(T &value, const CtsPoint<T> &p,
                                  const CtsPoint<T> &q, T x) {
  assert(p.x() == q.x());
  value = p.y() + slope(CtsSegment<T>(p, q)) * (x - p.x());
}

template <typename T>
static inline void inverseLinearFunction(T &value, const CtsPoint<T> &p,
                                         const CtsPoint<T> &q, T y) {
  assert(p.y() == q.y());
  value = p.x() + (y - p.y()) / slope(CtsSegment<T>(p, q));
}

// determine whether a line segment is horizontal.
template <typename T>
static inline bool horizontal(const CtsSegment<T> &seg) {
  return seg.low().y() == seg.high().y();
}
template <typename T>
static inline bool vertical(const CtsSegment<T> &seg) {
  return seg.low().x() == seg.high().x();
}
template <typename T>
static inline bool rectilinear(const CtsSegment<T> &seg) {
  return horizontal(seg) || vertical(seg);
}
template <typename T>
static inline bool manhattan_arc(const CtsSegment<T> &seg) {
  return std::abs(seg.high().y() - seg.low().y()) ==
         std::abs(seg.high().x() - seg.low().x());
}

// determine both point horizontal.
template <typename T>
static inline bool horizontal(const CtsPoint<T> &p1, const CtsPoint<T> &p2) {
  return p1.y() == p2.y();
}
template <typename T>
static inline bool vertical(const CtsPoint<T> &p1, const CtsPoint<T> &p2) {
  return p1.x() == p2.x();
}
template <typename T>
static inline bool rectilinear(const CtsPoint<T> &p1, const CtsPoint<T> &p2) {
  return horizontal(p1, p2) || vertical(p1, p2);
}
template <typename T>
static inline bool manhattan_arc(const CtsPoint<T> &p1, const CtsPoint<T> &p2) {
  return std::abs(p1.y() - p2.y()) == std::abs(p1.x() - p2.x());
}

template <typename T>
static inline bool contains(const CtsInterval<T> &interval, T coord) {
  return interval.low() <= coord && coord <= interval.high();
}

template <typename T>
static inline bool contains(const CtsPolygon<T> &bound,
                            const CtsPolygon<T> &poly) {
  auto points = poly.get_points();
  for (auto point : points) {
    if (!gtl::contains(bound, point)) {
      return false;
    }
  }
  return true;
}

template <typename T>
static inline bool contains(const CtsPolygon<T> &bound,
                            const CtsRectangle<T> &rect) {
  auto p1 = rect.low();
  auto p2 = rect.high();
  auto p3 = CtsPoint<T>(p1.x(), p2.y());
  auto p4 = CtsPoint<T>(p2.x(), p1.y());
  std::vector<CtsPoint<T>> points = {p1, p2, p3, p4};
  for (auto point : points) {
    if (!gtl::contains(bound, point)) {
      return false;
    }
  }
  return true;
}

template <typename T>
static inline bool contains(const CtsPolygon<T> &poly,
                            const CtsPoint<T> &point) {
  std::vector<CtsPoint<T>> points;
  std::copy(poly.begin(), poly.end(), std::back_inserter(points));

  if (points.size() == 1) {
    return points.front() == point;
  }
  if (points.size() == 2) {
    CtsSegment<T> seg(points.front(), points.back());
    return gtl::contains(seg, point);
  }

  if (points.front() != points.back()) {
    points.push_back(points.front());
  }
  for (size_t i = 0; i < points.size() - 1; i++) {
    auto cross = pgl::cross_product(points[i], point, points[i], points[i + 1]);
    CtsRectangle<T> rect(points[i], points[i + 1]);
    auto x_intval = rect.get(gtl::HORIZONTAL);
    auto y_intval = rect.get(gtl::VERTICAL);
    if (cross == 0 && pgl::contains(x_intval, point.x()) &&
        pgl::contains(y_intval, point.y())) {
      return true;
    }
  }
  //   auto origin_cross =
  //       pgl::cross_product(points.back(), point, points.back(),
  //       points.front());
  //   for (size_t i = 0; i < points.size() - 1; i++) {
  //     auto cur_cross =
  //         pgl::cross_product(points[i], point, points[i], points[i + 1]);
  //     if (cur_cross * origin_cross <= 0) {
  //       return false;
  //     }
  //   }
  // fix
  auto cur_cross = pgl::cross_product(points[0], point, points[0], points[1]);
  for (size_t i = 1; i < points.size() - 1; i++) {
    auto next_cross =
        pgl::cross_product(points[i], point, points[i], points[i + 1]);
    if (cur_cross * next_cross <= 0) {
      return false;
    }
    cur_cross = next_cross;
  }

  return true;
}

// calculate the length of interval.
template <typename T>
static inline T length(const CtsInterval<T> &interval) {
  return interval.high() - interval.low();
}

// calculate the manhattan distance between points.
template <typename T>
static inline T manhattan_distance(const CtsPoint<T> &lhs,
                                   const CtsPoint<T> &rhs) {
  return abs(lhs.x() - rhs.x()) + abs(lhs.y() - rhs.y());
}

// calculate the manhattan distance between point and segment.
template <typename T>
static inline T manhattan_distance(const CtsPoint<T> &point,
                                   const CtsSegment<T> &segment) {
  if (segment.low() == segment.high()) {
    return manhattan_distance(point, segment.low());
  }

  if (rectilinear(segment)) {
    gtl::orientation_2d orient =
        horizontal(segment) ? gtl::HORIZONTAL : gtl::VERTICAL;
    CtsInterval<T> itrval = interval(segment, orient);
    if (gtl::contains(itrval, point.get(orient))) {
      orient.turn_90();
      return std::abs(point.get(orient) - segment.low().get(orient));
    }
  }

  CtsRectangle<T> bbox = extends(segment);
  if (gtl::contains(bbox, point) == true) {
    auto k = slope(segment);
    auto vertical_dist = std::abs(
        point.y() - (segment.low().y() + k * (point.x() - segment.low().x())));
    auto horizon_dist = std::abs(
        point.x() - (segment.low().x() + (point.y() - segment.low().y()) / k));
    return static_cast<T>(std::min(vertical_dist, horizon_dist));
  }

  return std::min(manhattan_distance(point, segment.low()),
                  manhattan_distance(point, segment.high()));
}

// calculate the manhattan distance between two segments
template <typename T>
static inline T manhattan_distance(const CtsSegment<T> &lhs,
                                   const CtsSegment<T> &rhs) {
  T distance = 0;
  if (gtl::intersects(lhs, rhs) == false) {
    T temp1 = std::min(manhattan_distance(lhs.low(), rhs),
                       manhattan_distance(lhs.high(), rhs));
    T temp2 = std::min(manhattan_distance(rhs.low(), lhs),
                       manhattan_distance(rhs.high(), lhs));
    distance = std::min(temp1, temp2);
  }
  return distance;
}

// tilted rectangle region
template <typename T1, typename T2, typename T3>
static inline bool tilted_rect_region(CtsPolygon<T1> &region,
                                      const CtsSegment<T2> &core,
                                      const T3 &radius) {
  CtsPoint<T2> low = gtl::low(core);
  CtsPoint<T2> high = gtl::high(core);
  if (low > high) {
    std::swap(low, high);
  }
  if (manhattan_arc(core) == false) {
    return false;
  }

  std::vector<CtsPoint<T2>> points;
  if (gtl::x(high) > gtl::x(low)) {
    points.emplace_back(CtsPoint<T2>(gtl::x(low) - radius, gtl::y(low)));
    points.emplace_back(CtsPoint<T2>(gtl::x(low), gtl::y(low) - radius));
    points.emplace_back(CtsPoint<T2>(gtl::x(high) + radius, gtl::y(high)));
    points.emplace_back(CtsPoint<T2>(gtl::x(high), gtl::y(high) + radius));
    points.emplace_back(CtsPoint<T2>(gtl::x(low) - radius, gtl::y(low)));
  } else {
    points.emplace_back(CtsPoint<T2>(gtl::x(low), gtl::y(low) - radius));
    points.emplace_back(CtsPoint<T2>(gtl::x(low) + radius, gtl::y(low)));
    points.emplace_back(CtsPoint<T2>(gtl::x(high), gtl::y(high) + radius));
    points.emplace_back(CtsPoint<T2>(gtl::x(high) - radius, gtl::y(high)));
    points.emplace_back(CtsPoint<T2>(gtl::x(low), gtl::y(low) - radius));
  }
  region.set(points.begin(), points.end());
  return true;
}

template <typename T>
static inline bool tilted_rect_region(CtsPolygon<T> &region,
                                      const CtsPoint<T> &core, T radius) {
  return tilted_rect_region(region, CtsSegment<T>(core, core), radius);
}

template <typename T>
static inline CtsPoint<T> closest_point(const CtsPoint<T> &point,
                                        const CtsSegment<T> &seg) {
  if (seg.low() == seg.high()) {
    return seg.low();
  }

  if (rectilinear(seg)) {
    gtl::orientation_2d orient =
        horizontal(seg) ? gtl::HORIZONTAL : gtl::VERTICAL;
    CtsInterval<T> itrval = interval(seg, orient);
    if (gtl::contains(itrval, point.get(orient))) {
      CtsPoint<T> ans = point;
      orient.turn_90();
      ans.set(orient, seg.low().get(orient));
      return ans;
    }
  }

  CtsRectangle<T> bbox = extends(seg);
  if (gtl::contains(bbox, point) == true) {
    double k = slope(seg);

    CtsPoint<T> point1 = point;
    CtsPoint<T> point2 = point;
    point1.set(gtl::HORIZONTAL,
               seg.low().x() + (point.y() - seg.low().y()) / k);
    point2.set(gtl::VERTICAL, seg.low().y() + (point.x() - seg.low().x()) * k);

    if (manhattan_distance(point, point1) < manhattan_distance(point, point2)) {
      return point1;
    } else {
      return point2;
    }
  }

  if (manhattan_distance(point, seg.low()) <
      manhattan_distance(point, seg.high())) {
    return seg.low();
  } else {
    return seg.high();
  }
}

// find a point from segment a that
template <typename T>
static inline Pair<CtsPoint<T>> closest_point_pair(const CtsSegment<T> &seg_a,
                                                   const CtsSegment<T> &seg_b) {
  auto dist = pgl::manhattan_distance(seg_a, seg_b);

  if (pgl::manhattan_distance(seg_a.low(), seg_b) == dist) {
    return std::make_pair(seg_a.low(), pgl::closest_point(seg_a.low(), seg_b));
  }
  if (pgl::manhattan_distance(seg_a.high(), seg_b) == dist) {
    return std::make_pair(seg_a.high(),
                          pgl::closest_point(seg_a.high(), seg_b));
  }
  if (pgl::manhattan_distance(seg_b.low(), seg_a) == dist) {
    return std::make_pair(pgl::closest_point(seg_b.low(), seg_a), seg_b.low());
  }
  return std::make_pair(pgl::closest_point(seg_b.high(), seg_a), seg_b.high());
}

// closest edge of two polygon
template <typename T>
static inline std::vector<Pair<CtsSegment<T>>> closest_edges(
    const CtsPolygon<T> &poly_a, const CtsPolygon<T> &poly_b) {
  std::vector<CtsSegment<T>> segs_a = poly_a.get_edges();
  std::vector<CtsSegment<T>> segs_b = poly_b.get_edges();

  typedef std::pair<CtsSegment<T>, CtsSegment<T>> ElemType;
  std::vector<ElemType> results;
  std::vector<ElemType> edge_pairs;

  for (auto &seg_a : segs_a) {
    for (auto &seg_b : segs_b) {
      edge_pairs.emplace_back(std::make_pair(seg_a, seg_b));
    }
  }

  auto itr =
      std::min_element(edge_pairs.begin(), edge_pairs.end(),
                       [](const ElemType &lhs, const ElemType &rhs) {
                         return manhattan_distance(lhs.first, lhs.second) <
                                manhattan_distance(rhs.first, rhs.second);
                       });
  auto poly_dist = manhattan_distance(itr->first, itr->second);
  std::copy_if(edge_pairs.begin(), edge_pairs.end(),
               std::back_inserter(results), [&poly_dist](const ElemType &elem) {
                 return manhattan_distance(elem.first, elem.second) ==
                        poly_dist;
               });

  return results;
}

template <typename T>
inline void extents(CtsRectangle<T> &box,
                    const std::vector<CtsPoint<T>> &points) {
  T min_x = std::numeric_limits<T>::max();
  T min_y = std::numeric_limits<T>::max();
  T max_x = std::numeric_limits<T>::min();
  T max_y = std::numeric_limits<T>::min();
  for (auto &point : points) {
    min_x = std::min(min_x, point.x());
    min_y = std::min(min_y, point.y());
    max_x = std::max(max_x, point.x());
    max_y = std::max(max_y, point.y());
  }
  box = CtsRectangle<T>(min_x, min_y, max_x, max_y);
}

// find the convex polygon that contains the given point sets.
template <typename Point>
inline void convex_hull(std::vector<Point> &points) {
  auto cmp = [](const Point &lhs, const Point &rhs) {
    if (lhs.x() == rhs.x()) {
      return lhs.y() < rhs.y();
    }
    return lhs.x() < rhs.x();
  };
  std::sort(points.begin(), points.end(), cmp);
  // delete duplicate points
  points.erase(std::unique(points.begin(), points.end()), points.end());
  size_t n = points.size();
  std::vector<Point> hull;
  for (size_t i = 0; i < n; ++i) {
    while (hull.size() >= 2 &&
           cross_product(hull[hull.size() - 2], hull[hull.size() - 1],
                         points[i]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(points[i]);
  }
  size_t t = hull.size() - 1;
  for (auto i = static_cast<int>(n) - 2; i >= 0; --i) {
    while (hull.size() - t >= 2 &&
           cross_product(hull[hull.size() - 2], hull[hull.size() - 1],
                         points[i]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(points[i]);
  }
  hull.pop_back();
  points = hull;
}
template <typename T>
static inline CtsPoint<T> mid(const CtsPoint<T> &p1, const CtsPoint<T> &p2) {
  auto x = (p1.x() + p2.x()) / 2;
  auto y = (p1.y() + p2.y()) / 2;
  return CtsPoint<T>(x, y);
}
template <typename T>
static inline void simplify_polygon(CtsPolygon<T> &polygon) {
  if (polygon.size() == 1) {
    return;
  }
  if (polygon.size() == 2) {
    auto p1 = *polygon.begin();
    auto p2 = *(polygon.begin() + 1);
    if (p1 == p2) {
      polygon = CtsPolygon<T>({p1});
    }
    return;
  }
  std::vector<CtsPoint<T>> points;
  for (auto itr = polygon.begin(); itr != polygon.end() - 2; ++itr) {
    auto pre = *itr;
    auto cur = *(itr + 1);
    auto next = *(itr + 2);
    if (cur != mid(pre, next)) {
      points.emplace_back(cur);
    }
  }
  if (*(polygon.end() - 1) != mid(*(polygon.end() - 2), *polygon.begin())) {
    points.emplace_back(*(polygon.end() - 1));
  }
  if (*polygon.begin() != mid(*(polygon.end() - 1), *(polygon.begin() + 1))) {
    points.emplace_back(*polygon.begin());
  }
  if (points.size() > 0) {
    convex_hull(points);
  } else {
    points = {*polygon.begin()};
  }
  polygon = CtsPolygon<T>(points);
}

template <typename T>
static inline std::vector<CtsPoint<T>> edge_to_points(
    const CtsSegment<T> &edge) {
  std::vector<CtsPoint<T>> points;
  int sep_num = rectilinear(edge)
                    ? std::max(abs(edge.low().x() - edge.high().x()),
                               abs(edge.low().y() - edge.high().y()))
                    : std::min(abs(edge.low().x() - edge.high().x()),
                               abs(edge.low().y() - edge.high().y()));
  if (sep_num == 0) {
    points.emplace_back(edge.low());
    return points;
  }
  auto sep_x = (edge.high().x() - edge.low().x()) / sep_num;
  auto sep_y = (edge.high().y() - edge.low().y()) / sep_num;
  for (int i = 0; i <= sep_num; ++i) {
    double x = edge.low().x() + sep_x * i;
    double y = edge.low().y() + sep_y * i;
    if (integral_coord(x) && integral_coord(y)) {
      points.emplace_back(CtsPoint<T>(static_cast<T>(x), static_cast<T>(y)));
    }
  }
  return points;
}

template <typename T>
static inline CtsSegment<T> fixJoinSegment(const CtsSegment<T> &seg) {
  if (manhattan_arc(seg)) {
    return seg;
  }
  auto a = seg.low();
  auto b = seg.high();
  auto delta_x = b.x() - a.x();
  auto delta_y = b.y() - a.y();
  auto direction_x = delta_x > 0 ? 1 : -1;
  auto direction_y = delta_y > 0 ? 1 : -1;
  if (std::abs(delta_x) > std::abs(delta_y)) {
    return CtsSegment<T>(a,
                         CtsPoint<T>(a.x() + direction_x * std::abs(delta_y),
                                     a.y() + direction_y * std::abs(delta_y)));
  }
  return CtsSegment<T>(a, CtsPoint<T>(a.x() + direction_x * std::abs(delta_x),
                                      a.y() + direction_y * std::abs(delta_x)));
}

template <typename T>
static inline Point center(const CtsPolygon<T> &polygon) {
  double x = 0;
  double y = 0;
  for (auto point : polygon) {
    x += point.x();
    y += point.y();
  }
  return Point(x / polygon.size(), y / polygon.size());
}

template <typename T>
static inline void bound_margin_points(
    std::vector<std::vector<CtsPoint<T>>> &points,
    const CtsPolygon<T> &polygon) {
  for (auto edge : polygon.get_edges()) {
    auto cur_edge_points = edge_to_points(edge);
    points.emplace_back(cur_edge_points);
  }
}
template <typename T>
inline bool slope_legal(const CtsSegment<T> &seg) {
  return slope(seg) == 0 || slope(seg) == 1 || slope(seg) == -1 ||
         slope(seg) == std::numeric_limits<T>::max();
}

template <typename Coord>
inline bool polygon_slope_legal(const CtsPolygon<Coord> &poly) {
  for (auto edge : poly.get_edges()) {
    if (!slope_legal(edge)) {
      return false;
    }
  }
  return true;
}
template <typename Coord>
inline CtsPoint<Coord> straight_intersect(const Coord &k,
                                          const CtsPoint<Coord> &p,
                                          const CtsSegment<Coord> &seg) {
  auto res_x = 0;
  auto res_y = 0;
  auto k_t = slope(seg);
  auto p_t = seg.low();

  if (k == k_t) {
    if (p == seg.low()) {
      return seg.high();
    } else if (p == seg.high()) {
      return seg.low();
    }
    std::cout << "error : k == k_t" << std::endl;
  }
  if (k == std::numeric_limits<Coord>::max() ||
      k_t == std::numeric_limits<Coord>::max()) {
    res_x = k == std::numeric_limits<Coord>::max() ? p.x() : p_t.x();
    res_y = k == std::numeric_limits<Coord>::max()
                ? k_t * (res_x - p_t.x()) + p_t.y()
                : k * (res_x - p.x()) + p.y();
  } else {
    res_x = (k * p.x() - k_t * p_t.x() + p_t.y() - p.y()) / (k - k_t);
    res_y = k * (res_x - p.x()) + p.y();
  }

  return CtsPoint<Coord>(res_x, res_y);
}

template <typename Coord>
inline CtsPolygon<Coord> round_off_simple_coord(const CtsPolygon<Coord> &poly) {
  auto edges = poly.get_edges();
  std::cout << "dist :" << manhattan_distance(edges[0].low(), edges[0].high())
            << std::endl;
  if (edges.size() == 1 &&
      manhattan_distance(edges[0].low(), edges[0].high()) > 0.5) {
    return poly;
  }
  // remove the same edges
  std::vector<CtsSegment<Coord>> edges_set;
  for (auto &edge : edges) {
    if (std::find(edges_set.begin(), edges_set.end(), edge) ==
        edges_set.end()) {
      edges_set.emplace_back(edge);
    }
  }
  std::vector<CtsPoint<Coord>> vertexs;
  for (auto itr = edges_set.begin(); itr != edges_set.end(); ++itr) {
    auto edge = *itr;
    if (manhattan_distance(edge.low(), edge.high()) <= 0.5) {
      if (integral_coord(edge.low())) {
        vertexs.emplace_back(edge.low());
        if (itr != edges_set.end() - 1) {
          ++itr;
        } else {
          if (vertexs.size() > 0 && vertexs.front() == edge.high()) {
            vertexs.erase(vertexs.begin());
          }
        }
      }
    } else {
      vertexs.emplace_back(edge.low());
    }
  }
  if (vertexs.size() == 0) {
    vertexs.emplace_back(edges_set[0].high());
  }
  if (vertexs.size() == 1) {
    vertexs.emplace_back(vertexs[0]);
  }
  // if (vertexs.size() == 3) {
  //   std::vector<double> res;
  //   for (size_t i = 0; i < edges.size(); ++i) {
  //     res.push_back(gtl::euclidean_distance(edges[i].low(),
  //     edges[i].high()));
  //   }
  //   std::sort(res.begin(), res.end());
  //   auto min_dist = res[0];
  //   auto mid_dist = res[1];
  //   auto max_dist = res[2];
  //   if (max_dist == 0) {
  //     return CtsPolygon<Coord>({edges[0].low(), edges[0].high()});
  //   }
  //   if (1.0 * (min_dist / max_dist) < 0.5) {
  //     for (auto edge : edges) {
  //       if (gtl::euclidean_distance(edge.low(), edge.high()) == mid_dist) {
  //         return CtsPolygon<Coord>({edge.low(), edge.high()});
  //       }
  //     }
  //   }
  // }
  return CtsPolygon<Coord>(vertexs);
}

template <typename Coord1, typename Coord2>
inline bool round_off_line(CtsPolygon<Coord1> &new_poly,
                           const CtsPolygon<Coord2> &old_poly) {
  if (old_poly.get_edges().size() != 1) {
    assert(false);
  }
  auto line = old_poly.get_edges()[0];
  if (!slope_legal(line)) {
    new_poly = CtsPolygon<Coord1>({line.low(), line.low()});
  }
  std::vector<CtsPoint<Coord1>> vertexs;
  auto p1_x = line.low().x();
  auto p1_y = line.low().y();
  auto p2_x = line.high().x();
  auto p2_y = line.high().y();
  if (line.low() == line.high()) {
    vertexs.emplace_back(CtsPoint<Coord1>(std::ceil(p1_x), std::ceil(p1_y)));
    vertexs.emplace_back(CtsPoint<Coord1>(std::ceil(p2_x), std::ceil(p2_y)));
    new_poly = CtsPolygon<Coord1>(vertexs.begin(), vertexs.end());
    return true;
  }
  if ((integral_coord(p1_x) == integral_coord(p2_y)) &&
      (integral_coord(p2_x) == integral_coord(p1_y)) &&
      (integral_coord(p1_x) != integral_coord(p2_x))) {
    auto k = slope(line);
    p1_x = integral_coord(p1_x) ? p1_x
                                : (k == 1 ? std::floor(p1_x) : std::ceil(p1_x));
    p1_y = integral_coord(p1_y) ? p1_y
                                : (k == 1 ? std::floor(p1_y) : std::ceil(p1_y));
    p2_x = integral_coord(p2_x) ? p2_x : std::ceil(p2_x);
    p2_y = integral_coord(p2_y) ? p2_y : std::ceil(p2_y);
  } else {
    p1_x = std::ceil(p1_x);
    p1_y = std::ceil(p1_y);
    p2_x = std::ceil(p2_x);
    p2_y = std::ceil(p2_y);
  }
  vertexs.emplace_back(CtsPoint<Coord1>(p1_x, p1_y));
  vertexs.emplace_back(CtsPoint<Coord1>(p2_x, p2_y));
  new_poly = CtsPolygon<Coord1>(vertexs.begin(), vertexs.end());
  return true;
}

// template <typename Coord1, typename Coord2>
// inline bool round_off_triangle(CtsPolygon<Coord1> &new_poly,
//                                const CtsPolygon<Coord2> &old_poly) {
//   if (old_poly.get_edges().size() != 3) {
//     assert(false);
//   }
//   auto min_edge_length = std::numeric_limits<Coord2>::max();
//   for (auto edge : old_poly.get_edges()) {
//     min_edge_length = std::min(
//         min_edge_length, gtl::euclidean_distance(edge.low(), edge.high()));
//   }
//   std::vector<CtsPoint<Coord1>> vertexs;
//   CtsPoint<Coord1> p, p1, p2;
//   for (auto edge : old_poly.get_edges()) {
//     auto k = slope(edge);
//     if (k != 0 && k != std::numeric_limits<Coord2>::max()) {
//       p1 = edge.low();
//       p2 = edge.high();
//     }
//     vertexs.emplace_back(edge.low());
//   }
//   for (auto point : vertexs) {
//     if (point != p1 && point != p2) {
//       p = point;
//     }
//   }
//   if (gtl::euclidean_distance(p, p1) == min_edge_length) {
//   }
// }

template <typename Coord1, typename Coord2>
inline bool round_off_float(CtsPolygon<Coord1> &new_poly,
                            const CtsPolygon<Coord2> &old_poly) {
  std::vector<CtsPoint<Coord1>> integral_points;
  std::vector<CtsPoint<Coord2>> un_integral_points;
  for (auto edge : old_poly.get_edges()) {
    if (integral_coord(edge.low())) {
      integral_points.emplace_back(edge.low());
    } else {
      un_integral_points.emplace_back(edge.low());
    }
  }
  if (un_integral_points.empty()) {
    new_poly = old_poly;
    return true;
  }

  for (auto point : un_integral_points) {
    CtsRectangle<Coord2> rect;
    extents(rect, point);

    std::vector<CtsPoint<Coord2>> points;
    bound_integral_points(points, rect);

    for (auto &point : points) {
      if (contains(old_poly, point)) {
        integral_points.push_back(point);
      }
    }
  }

  convex_hull(integral_points);
  // remove the same point
  std::vector<CtsPoint<Coord1>> point_set;
  for (auto &point : integral_points) {
    if (std::find(point_set.begin(), point_set.end(), point) ==
        point_set.end()) {
      point_set.emplace_back(point);
    }
  }
  new_poly = CtsPolygon<Coord1>(point_set);
  return true;
}

template <typename Coord>
inline bool round_off_direction(CtsPolygon<Coord> &poly) {
  std::vector<CtsPoint<Coord>> vertexs;
  for (auto edge : poly.get_edges()) {
    vertexs.push_back(edge.low());
  }
  while (!polygon_slope_legal(poly)) {
    auto edges = poly.get_edges();
    auto head = edges.front();
    auto tail = edges.back();
    edges.insert(edges.begin(), tail);
    edges.emplace_back(head);
    for (auto itr = edges.begin() + 1; itr != edges.end() - 1; ++itr) {
      auto pre_edge = *(itr - 1);
      auto cur_edge = *itr;
      auto next_edge = *(itr + 1);
      if (!slope_legal(cur_edge)) {
        if (!slope_legal(pre_edge) || !slope_legal(next_edge)) {
          std::cout << "error" << std::endl;
        }
        auto k = slope(cur_edge);
        auto fix_k = 0;
        // compute the most closly legal slope
        if (std::abs(k) >= 2) {
          fix_k = std::numeric_limits<Coord>::max();
        } else if (std::abs(k) >= 0.5) {
          fix_k = k > 0 ? 1 : -1;
        } else {
          fix_k = 0;
        }
        auto p1 = cur_edge.low();
        auto p2 = cur_edge.high();
        // compute p1's fix segment
        auto is_p_l = straight_intersect(fix_k, p1, pre_edge);
        auto is_p_r = straight_intersect(fix_k, p1, next_edge);
        if (!contains(poly, is_p_l) || !contains(poly, is_p_r)) {
          is_p_l = straight_intersect(fix_k, p2, pre_edge);
          is_p_r = straight_intersect(fix_k, p2, next_edge);
        }
        if (p1 == is_p_l || p1 == is_p_r) {
          for (auto itr = vertexs.begin(); itr != vertexs.end(); ++itr) {
            if (*itr == p2) {
              *itr = p1 == is_p_l ? is_p_r : is_p_l;
              break;
            }
          }
        } else {
          for (auto itr = vertexs.begin(); itr != vertexs.end(); ++itr) {
            if (*itr == p1) {
              *itr = p2 == is_p_l ? is_p_r : is_p_l;
              break;
            }
          }
        }
        poly = CtsPolygon<Coord>(vertexs.begin(), vertexs.end());
      }
    }
  }
  return true;
}
template <typename Coord1, typename Coord2>
inline bool round_off_coord(CtsPolygon<Coord1> &new_poly,
                            const CtsPolygon<Coord2> &old_poly) {
  auto old_fix_poly = round_off_simple_coord(old_poly);
  if (old_fix_poly.get_edges()[0] ==
      CtsSegment<Coord2>(CtsPoint<Coord2>(3447677.5, 452652),
                         CtsPoint<Coord2>(3447677.5, 452153))) {
    std::cout << "error" << std::endl;
  }
  // repair polygon's line which length just 0.5
  if (old_fix_poly.get_edges().size() == 1) {
    return round_off_line(new_poly, old_fix_poly);
  }
  // repair polygon's point which location is float type
  if (!round_off_float(new_poly, old_fix_poly)) {
    return false;
  }
  // // repair polygon which is triangle
  // if (old_fix_poly.get_edges().size() == 3) {
  //   return round_off_triangle(new_poly, old_fix_poly);
  // }
  // round off direction
  return round_off_direction(new_poly);
}
template <typename T, typename t>
static inline CtsSegment<T> forwardShift(const CtsSegment<T> &segment,
                                         const t &step) {
  auto p1 = segment.low();
  auto p2 = segment.high();
  return CtsSegment<T>(CtsPoint<T>(p1.x() + step, p1.y() + step),
                       CtsPoint<T>(p2.x() + step, p2.y() + step));
}

template <typename T, typename t>
static inline CtsPolygon<T> forwardShift(const CtsPolygon<T> &polygon,
                                         const t &step) {
  auto points = polygon.get_points();
  std::vector<CtsPoint<T>> shift_points;
  for (auto &point : points) {
    shift_points.emplace_back(CtsPoint<T>(point.x() + step, point.y() + step));
  }
  return CtsPolygon<T>(shift_points);
}

template <typename T, typename t>
static inline CtsRectangle<T> forwardShift(const CtsRectangle<T> &rect,
                                           const t &step) {
  return CtsRectangle<T>(rect.low().x() + step, rect.low().y() + step,
                         rect.high().x() + step, rect.high().y() + step);
}

template <typename T, typename t>
static inline CtsSegment<T> backwardShift(const CtsSegment<T> &segment,
                                          const t &step) {
  auto p1 = segment.low();
  auto p2 = segment.high();
  return CtsSegment<T>(CtsPoint<T>(p1.x() - step, p1.y() - step),
                       CtsPoint<T>(p2.x() - step, p2.y() - step));
}

template <typename T, typename t>
static inline CtsPolygon<T> backwardShift(const CtsPolygon<T> &polygon,
                                          const t &step) {
  auto points = polygon.get_points();
  std::vector<CtsPoint<T>> shift_points;
  for (auto &point : points) {
    shift_points.emplace_back(CtsPoint<T>(point.x() - step, point.y() - step));
  }
  return CtsPolygon<T>(shift_points);
}

template <typename T, typename t>
static inline CtsPolygon<T> backwardShift(const CtsRectangle<T> &rect,
                                          const t &step) {
  auto p1 = rect.low();
  auto p2 = rect.high();
  return CtsPolygon<T>({
      CtsPoint<T>(p1.x() - step, p1.y() - step),
      CtsPoint<T>(p1.x() - step, p2.y() - step),
      CtsPoint<T>(p2.x() - step, p2.y() - step),
      CtsPoint<T>(p2.x() - step, p1.y() - step),
      CtsPoint<T>(p1.x() - step, p1.y() - step),
  });
}
// template <typename T>
// static inline std::pair<CtsSegment<T>, CtsSegment<T>> joinSegment(
//     const CtsPolygon<T> &mr_i, const CtsPolygon<T> &mr_j) {
//   auto intersection = intersectionByBg(mr_i, mr_j);
//   if(intersection.size() > 0) {
//     auto seg = longest_segment(intersection);
//     return std::make_pair(seg, seg);
//   }
//   auto mr_i_t = CtsPolygon<int64_t>(mr_i.get_points());
//   auto mr_j_t = CtsPolygon<int64_t>(mr_j.get_points());
//   // merge region intersect
//   if (!gtl::empty(mr_i_t & mr_j_t)) {
//     gtl::scale_up(mr_i_t, SCALE_FACTOR);
//     gtl::scale_up(mr_j_t, SCALE_FACTOR);
//     CtsPolygonSet<int64_t> polyset;
//     polyset += mr_i_t & mr_j_t;
//     auto poly_t = polyset.front();
//     gtl::scale_down(poly_t, SCALE_FACTOR);
//     auto poly = CtsPolygon<T>(poly_t.get_points());
//     CtsSegment<T> join_seg;
//     longest_segment(join_seg, poly.get_edges());
//     if (rectilinear(join_seg)) {
//       join_seg = {join_seg.low(), join_seg.low()};
//     }
//     return std::make_pair(join_seg, join_seg);
//   }
//   // line merge region intersect
//   if (mr_i.size() == 2 || mr_j.size() == 2) {
//     bg_Segment line_mr;
//     bg_Polygon poly_mr;
//     if (mr_i.size() == 2) {
//       line_mr = bg_Segment(
//           {bg_Point(mr_i.get_points()[0].x(), mr_i.get_points()[0].y()),
//            bg_Point(mr_i.get_points()[1].x(), mr_i.get_points()[1].y())});
//       for (auto &p : mr_j.get_points()) {
//         poly_mr.outer().push_back(bg_Point(p.x(), p.y()));
//       }
//     } else {
//       line_mr = bg_Segment(
//           {bg_Point(mr_j.get_points()[0].x(), mr_j.get_points()[0].y()),
//            bg_Point(mr_j.get_points()[1].x(), mr_j.get_points()[1].y())});
//       for (auto &p : mr_i.get_points()) {
//         poly_mr.outer().push_back(bg_Point(p.x(), p.y()));
//       }
//     }
//     std::vector<bg_Point> intersection;
//     bg::intersection(poly_mr, line_mr, intersection);
//     if (!intersection.empty()) {
//       auto join_seg =
//           intersection.size() == 1
//               ? CtsSegment<T>(
//                     CtsPoint<T>(intersection[0].x(), intersection[0].y()),
//                     CtsPoint<T>(intersection[0].x(), intersection[0].y()))
//               : CtsSegment<T>(
//                     CtsPoint<T>(intersection[0].x(), intersection[0].y()),
//                     CtsPoint<T>(intersection[1].x(), intersection[1].y()));
//       return std::make_pair(join_seg, join_seg);
//     }
//   }
//   // not intersect
//   auto edge_pair = closestEdge(mr_i, mr_j);
//   auto join_seg_i = edge_pair.first;
//   auto join_seg_j = edge_pair.second;
//   // 计算最近边中的有效的部分，有效部分作为joining segment
//   // two closest edge are manhattan arc
//   if (!manhattan_arc(join_seg_i) || !manhattan_arc(join_seg_j)) {
//     auto point_pair = closest_point_pair(join_seg_i, join_seg_j);
//     join_seg_i = CtsSegment<T>(point_pair.first, point_pair.first);
//     join_seg_j = CtsSegment<T>(point_pair.second, point_pair.second);
//   }
//   CtsPolygon<T> trr;
//   auto radius = manhattan_distance(join_seg_i, join_seg_j);
//   assert(tilted_rect_region(trr, join_seg_j, radius));
//   assert(bound_intersect(join_seg_i, join_seg_i, trr));
//   assert(tilted_rect_region(trr, join_seg_i, radius));
//   assert(bound_intersect(join_seg_j, join_seg_j, trr));
//   return std::make_pair(join_seg_i, join_seg_j);
// }

template <typename T>
static inline Pair<CtsSegment<T>> closestEdge(const CtsPolygon<T> &mr_a,
                                              const CtsPolygon<T> &mr_b) {
  auto edge_pairs = closest_edges(mr_a, mr_b);
  assert(!edge_pairs.empty());
  for (auto &edge_pair : edge_pairs) {
    if (cross_product(edge_pair.first, edge_pair.second) == 0 &&
        manhattan_arc(edge_pair.first) && manhattan_arc(edge_pair.second)) {
      return edge_pair;
    }
  }
  for (auto &edge_pair : edge_pairs) {
    if (cross_product(edge_pair.first, edge_pair.second) == 0) {
      return edge_pair;
    }
  }
  for (auto &edge_pair : edge_pairs) {
    if (dot_product(edge_pair.first, edge_pair.second) == 0) {
      return edge_pair;
    }
  }
  return edge_pairs.front();
}

template <typename T>
static inline Pair<CtsPoint<T>> cutSegment(const CtsSegment<T> &seg,
                                           const T &dist_i, const T &dist_j) {
  auto first = seg.low();
  auto second = seg.high();
  CtsPoint<T> p1;
  CtsPoint<T> p2;
  if (first.x() == second.x()) {
    p1.x(first.x());
    p2.x(second.x());
  } else {
    p1.x(first.x() - dist_i * ((first.x() - second.x()) /
                               std::abs((first.x() - second.x()))));
    p2.x(second.x() + dist_j * ((first.x() - second.x()) /
                                std::abs((first.x() - second.x()))));
  }
  if (first.y() == second.y()) {
    p1.y(first.y());
    p2.y(second.y());
  } else {
    p1.y(first.y() - dist_i * ((first.y() - second.y()) /
                               std::abs((first.y() - second.y()))));
    p2.y(second.y() + dist_j * ((first.y() - second.y()) /
                                std::abs((first.y() - second.y()))));
  }
  return std::make_pair(p1, p2);
}

template <typename T>
static inline CtsPoint<T> cutSegment(const CtsSegment<T> &seg,
                                     const T &dist_i) {
  auto first = seg.low();
  auto second = seg.high();
  CtsPoint<T> p1;
  if (first.x() == second.x()) {
    p1.x(first.x());
  } else {
    p1.x(first.x() - dist_i * ((first.x() - second.x()) /
                               std::abs((first.x() - second.x()))));
  }
  if (first.y() == second.y()) {
    p1.y(first.y());
  } else {
    p1.y(first.y() - dist_i * ((first.y() - second.y()) /
                               std::abs((first.y() - second.y()))));
  }
  return p1;
}
}  // namespace pgl
}  // namespace icts