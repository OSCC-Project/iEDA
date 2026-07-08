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
 * @file Geometry.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-16
 * @brief Geometry helpers for topology algorithms.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

#include "Point.hh"
#include "Rect.hh"
#include "Region.hh"

namespace icts::geometry {

/**
 * @brief Manhattan (L1) distance between two points.
 *
 * Accepts any coordinate type T (int, double, etc.).
 * Returns the same coordinate type T to preserve the caller's distance semantics.
 */
template <typename T>
inline auto Manhattan(const Point<T>& a, const Point<T>& b) -> T
{
  using std::abs;
  return abs(a.get_x() - b.get_x()) + abs(a.get_y() - b.get_y());
}

template <typename T>
inline auto Manhattan(const Point<T>& point, const Rect<T>& rect) -> T
{
  return Manhattan(point, rect.clamp(point));
}

template <typename T>
inline auto ProjectNearest(const Region<T>& region, const Point<T>& point) -> std::optional<Point<T>>
{
  return region.project_nearest(point);
}

/**
 * @brief Center of mass (arithmetic mean) of a collection of points.
 *
 * The getter extracts a Point from each element in the collection.
 * Return type is always floating-point:
 *   - integer coords  → Point<double>
 *   - floating coords → Point<same floating type>
 */
template <typename Value, typename PointGetter>
inline auto CalcCenter(const std::vector<Value>& values, PointGetter getter)
{
  using PointT = std::decay_t<decltype(getter(std::declval<const Value&>()))>;
  using CoordT = typename PointT::CoordType;
  using FloatT = std::conditional_t<std::is_floating_point_v<CoordT>, CoordT, double>;

  if (values.empty()) {
    return Point<FloatT>{};
  }
  FloatT sum_x = 0;
  FloatT sum_y = 0;
  for (const auto& value : values) {
    auto p = getter(value);
    sum_x += static_cast<FloatT>(p.get_x());
    sum_y += static_cast<FloatT>(p.get_y());
  }
  const auto count = static_cast<FloatT>(values.size());
  return Point<FloatT>(sum_x / count, sum_y / count);
}

/**
 * @brief Coordinate-wise median of a collection of points.
 *
 * The getter extracts a Point from each element in the collection.
 * Return type matches the getter's Point coordinate type:
 *   - getter returns Point<int>    → returns Point<int>
 *   - getter returns Point<double> → returns Point<double>
 */
template <typename Value, typename PointGetter>
inline auto CalcMedian(const std::vector<Value>& values, PointGetter getter)
{
  using PointT = std::decay_t<decltype(getter(std::declval<const Value&>()))>;
  using CoordT = typename PointT::CoordType;

  if (values.empty()) {
    return PointT{};
  }
  std::vector<CoordT> xs;
  std::vector<CoordT> ys;
  xs.reserve(values.size());
  ys.reserve(values.size());
  for (const auto& value : values) {
    auto p = getter(value);
    xs.push_back(p.get_x());
    ys.push_back(p.get_y());
  }

  const auto mid_offset = static_cast<std::ptrdiff_t>(xs.size() / 2);
  auto x_mid_iter = xs.begin() + mid_offset;
  auto y_mid_iter = ys.begin() + mid_offset;
  std::nth_element(xs.begin(), x_mid_iter, xs.end());
  std::nth_element(ys.begin(), y_mid_iter, ys.end());
  return PointT(*x_mid_iter, *y_mid_iter);
}

/**
 * @brief Project a point onto the closest position on an L1 circle, snapped to an integer grid.
 *
 * This function is specific to integer (DBU) coordinates because it performs
 * floor/ceil/round snapping to find the best integer grid point.
 *
 * @param center  Center of the L1 circle
 * @param point   Point to project
 * @param r       Radius of the L1 circle
 * @param min_x, min_y, max_x, max_y  Bounding box constraints
 */
inline auto ProjectToL1Circle(const Point<int>& center, const Point<int>& point, double r, int min_x = 0, int min_y = 0,
                              int max_x = std::numeric_limits<int>::max(), int max_y = std::numeric_limits<int>::max()) -> Point<int>
{
  if (min_x > max_x) {
    std::swap(min_x, max_x);
  }
  if (min_y > max_y) {
    std::swap(min_y, max_y);
  }

  auto clampi = [](int v, int lo, int hi) -> int { return std::max(lo, std::min(v, hi)); };
  auto clampd = [](double v, double lo, double hi) -> double { return std::max(lo, std::min(v, hi)); };

  if (r <= 0.0) {
    return Point<int>(clampi(center.get_x(), min_x, max_x), clampi(center.get_y(), min_y, max_y));
  }

  auto inside = [&](double x, double y) -> bool { return x >= min_x && x <= max_x && y >= min_y && y <= max_y; };

  const double cx = center.get_x();
  const double cy = center.get_y();
  const double tx = point.get_x();
  const double ty = point.get_y();

  auto to_int_best = [&](double x, double y) -> Point<int> {
    const int fx = static_cast<int>(std::floor(x));
    const int cx2 = static_cast<int>(std::ceil(x));
    const int fy = static_cast<int>(std::floor(y));
    const int cy2 = static_cast<int>(std::ceil(y));
    const int rx = static_cast<int>(std::lround(x));
    const int ry = static_cast<int>(std::lround(y));

    const std::array<std::pair<int, int>, 5> candidates = {{{fx, fy}, {fx, cy2}, {cx2, fy}, {cx2, cy2}, {rx, ry}}};

    bool has = false;
    int best_ix = clampi(rx, min_x, max_x);
    int best_iy = clampi(ry, min_y, max_y);
    double best_l1err = std::numeric_limits<double>::infinity();
    double best_d2 = std::numeric_limits<double>::infinity();

    for (const auto& c : candidates) {
      int ix = c.first;
      int iy = c.second;
      if (ix < min_x || ix > max_x || iy < min_y || iy > max_y)
        continue;

      const double l1 = std::fabs(ix - cx) + std::fabs(iy - cy);
      const double l1err = std::fabs(l1 - r);

      const double dx = ix - x;
      const double dy = iy - y;
      const double d2 = dx * dx + dy * dy;

      if (!has || l1err < best_l1err || (l1err == best_l1err && d2 < best_d2)) {
        has = true;
        best_l1err = l1err;
        best_d2 = d2;
        best_ix = ix;
        best_iy = iy;
      }
    }

    return Point<int>(best_ix, best_iy);
  };

  bool found = false;
  double best_x = cx, best_y = cy;
  double best_d2 = std::numeric_limits<double>::infinity();

  auto relax = [&](double x, double y) -> void {
    if (!inside(x, y))
      return;
    const double dx = x - tx;
    const double dy = y - ty;
    const double d2 = dx * dx + dy * dy;
    if (!found || d2 < best_d2) {
      best_d2 = d2;
      best_x = x;
      best_y = y;
      found = true;
    }
  };

  for (int su : {+1, -1})
    for (int sv : {+1, -1}) {
      double t_left = 0.0;
      double t_right = r;

      auto intersect_t = [&](double a, double b, double lo, double hi) -> bool {
        if (std::fabs(a) < 1e-12)
          return (b >= lo && b <= hi);
        double t1 = (lo - b) / a;
        double t2 = (hi - b) / a;
        if (t1 > t2)
          std::swap(t1, t2);
        t_left = std::max(t_left, t1);
        t_right = std::min(t_right, t2);
        return t_left <= t_right;
      };

      if (!intersect_t(su, cx, min_x, max_x))
        continue;
      if (!intersect_t(-sv, cy + sv * r, min_y, max_y))
        continue;

      const double ax = cx;
      const double ay = cy + sv * r;
      const double dx = su;
      const double dy = -sv;
      const double num = (tx - ax) * dx + (ty - ay) * dy;
      const double den = dx * dx + dy * dy;
      double t = num / den;
      t = clampd(t, t_left, t_right);

      relax(cx + su * t, cy + sv * (r - t));
    }

  if (found) {
    return to_int_best(best_x, best_y);
  }

  const double x = clampd(tx, min_x, max_x);
  const double y = clampd(ty, min_y, max_y);
  return Point<int>(clampi(static_cast<int>(std::lround(x)), min_x, max_x), clampi(static_cast<int>(std::lround(y)), min_y, max_y));
}

}  // namespace icts::geometry
