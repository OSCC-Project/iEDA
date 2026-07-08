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
 * @file Components.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-03-08
 * @brief Shared geometry and timing helpers for bound-skew tree routing
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

#include "bound_skew_tree/config/BSTRoutingConfig.hh"
namespace icts::bst {

template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

/**
 * @brief Global constant
 *
 */
constexpr static size_t kHead = 0;
constexpr static size_t kTail = 1;
constexpr static size_t kLeft = 0;
constexpr static size_t kRight = 1;
constexpr static size_t kMin = 0;
constexpr static size_t kMax = 1;
constexpr static size_t kX = 0;
constexpr static size_t kY = 1;
constexpr static size_t kH = 0;
constexpr static size_t kV = 1;
constexpr static double kEpsilon = 1e-7;

/**
 * @brief Global function
 *
 */
#define FOR_EACH_BST_SIDE(side) for (size_t side = 0; side < 2; ++side)

template <Numeric T1, Numeric T2>
constexpr static auto Equal(const T1& a, const T2& b, const double& epsilon = kEpsilon) -> bool
{
  return std::abs(a - b) < epsilon;
}

class Point
{
 public:
  Point() = default;
  Point(const double& t_x, const double& t_y, const double& t_max, const double& t_min, const double& t_val)
      : x(t_x), y(t_y), max(t_max), min(t_min), val(t_val)
  {
  }
  Point(const double& t_x, const double& t_y) : x(t_x), y(t_y) {}

  auto operator+(const Point& other) const -> Point { return Point(x + other.x, y + other.y); }
  auto operator-(const Point& other) const -> Point { return Point(x - other.x, y - other.y); }
  auto operator*(const double& scale) const -> Point { return Point(x * scale, y * scale); }
  auto operator/(const double& scale) const -> Point { return Point(x / scale, y / scale); }
  auto operator+=(const Point& other) -> Point
  {
    x += other.x;
    y += other.y;
    return *this;
  }
  auto operator-=(const Point& other) -> Point
  {
    x -= other.x;
    y -= other.y;
    return *this;
  }
  auto operator*=(const double& scale) -> Point
  {
    x *= scale;
    y *= scale;
    return *this;
  }
  auto operator/=(const double& scale) -> Point
  {
    x /= scale;
    y /= scale;
    return *this;
  }

  double x = 0;
  double y = 0;
  double max = 0;
  double min = 0;
  double val = 0;
};

/**
 * @brief type alias
 *
 */
using JoinSegment = std::vector<Point>;
using Points = std::vector<Point>;
using Region = std::vector<Point>;
using Line = std::array<Point, 2>;
using PointPair = std::array<Point, 2>;
template <typename T>
using Side = std::array<T, 2>;

class Area
{
 public:
  Area(const size_t& id) { _name = "steiner_" + std::to_string(id); };
  Area(const std::string& name, const Point& location, const double& cap_load, const double& sub_len = 0.0,
       const BSTRoutingRCPattern& rc_pattern = BSTRoutingRCPattern::kHV, const bool is_fixed_terminal = false)
      : _name(name),
        _cap_load(cap_load),
        _sub_len(sub_len),
        _rc_pattern(rc_pattern),
        _location(location),
        _is_fixed_terminal(is_fixed_terminal)
  {
    if (is_fixed_terminal) {
      _merge_region.push_back(_location);
      _convex_hull.push_back(_location);
    }
  }

  Area(const std::string& name, const double& x, const double& y, const double& cap_load, const double& min_delay = 0.0,
       const double& max_delay = 0.0, const double& sub_len = 0.0, const BSTRoutingRCPattern& rc_pattern = BSTRoutingRCPattern::kHV,
       const bool is_fixed_terminal = false)
      : Area(name, Point(x, y, max_delay, min_delay, cap_load), cap_load, sub_len, rc_pattern, is_fixed_terminal)
  {
  }
  // get
  auto get_name() const -> const std::string& { return _name; }
  auto get_cap_load() const -> const double& { return _cap_load; }
  auto get_sub_len() const -> const double& { return _sub_len; }
  auto get_edge_len(const size_t& side) const -> const double& { return _edge_len[side]; }
  auto get_radius() const -> const double& { return _radius; }
  auto get_rc_pattern() const -> const BSTRoutingRCPattern& { return _rc_pattern; }

  auto get_location() const -> const Point& { return _location; }
  auto get_parent() const -> Area* { return _parent; }
  auto get_left() const -> Area* { return _left; }
  auto get_right() const -> Area* { return _right; }
  auto get_line(const size_t& side) const -> Line { return _lines[side]; }
  auto get_lines() const -> Side<Line> { return _lines; }
  auto get_merge_region() const -> Region { return _merge_region; }
  auto getMergeRegionLines() const -> std::vector<Line>
  {
    std::vector<Line> lines;
    for (size_t i = 0; i < _merge_region.size(); ++i) {
      auto j = (i + 1) % _merge_region.size();
      lines.push_back({_merge_region[i], _merge_region[j]});
    }
    return lines;
  }

  auto get_convex_hull() const -> Region { return _convex_hull; }
  auto getConvexHullLines() const -> std::vector<Line>
  {
    std::vector<Line> lines;
    for (size_t i = 0; i < _convex_hull.size(); ++i) {
      auto j = (i + 1) % _convex_hull.size();
      lines.push_back({_convex_hull[i], _convex_hull[j]});
    }
    return lines;
  }
  // set
  auto set_name(const std::string& name) -> void { _name = name; }
  auto set_cap_load(const double& cap_load) -> void { _cap_load = cap_load; }
  auto set_sub_len(const double& sub_len) -> void { _sub_len = sub_len; }
  auto set_edge_len(const size_t& side, const double& edge_len) -> void { _edge_len[side] = edge_len; }
  auto set_radius(const double& radius) -> void { _radius = radius; }
  auto set_rc_pattern(const BSTRoutingRCPattern& rc_pattern) -> void { _rc_pattern = rc_pattern; }

  auto set_location(const Point& location) -> void { _location = location; }
  auto set_parent(Area* parent) -> void { _parent = parent; }
  auto set_left(Area* left) -> void { _left = left; }
  auto set_right(Area* right) -> void { _right = right; }
  auto set_line(const size_t& side, const Line& line) -> void { _lines[side] = line; }
  auto set_merge_region(const Region& merge_region) -> void { _merge_region = merge_region; }
  auto set_convex_hull(const Region& convex_hull) -> void { _convex_hull = convex_hull; }

  // add
  auto add_merge_region_point(const Point& point) -> void { _merge_region.push_back(point); }
  auto add_convex_hull_point(const Point& point) -> void { _convex_hull.push_back(point); }

  auto is_fixed_terminal() const -> bool { return _is_fixed_terminal; }
  auto set_is_fixed_terminal(bool is_fixed_terminal) -> void { _is_fixed_terminal = is_fixed_terminal; }

 private:
  std::string _name;
  double _cap_load = 0;
  double _sub_len = 0;
  Side<double> _edge_len = {0, 0};
  double _radius = 0;
  BSTRoutingRCPattern _rc_pattern = BSTRoutingRCPattern::kHV;

  Point _location;
  Area* _parent = nullptr;
  Area* _left = nullptr;
  Area* _right = nullptr;
  Side<Line> _lines;
  Region _merge_region;
  Region _convex_hull;
  bool _is_fixed_terminal = false;
};

struct Match
{
  Area* left = nullptr;
  Area* right = nullptr;
  double merge_cost = 0.0;
};

class Interval
{
 public:
  Interval() = default;
  Interval(const double& val) : _low(val), _high(val) {}
  Interval(const double& low, const double& high) : _low(low), _high(high) {}
  Interval(const Interval& other) : _low(other._low), _high(other._high) {}

  auto low() const -> const double& { return _low; }
  auto high() const -> const double& { return _high; }

  auto is_empty() const -> bool { return _low > _high; }
  auto is_point() const -> bool { return _low == _high; }

  auto enclose(const double& val) -> void
  {
    if (is_empty()) {
      _low = val;
      _high = val;
    } else {
      _low = std::min(_low, val);
      _high = std::max(_high, val);
    }
  }
  auto enclose(const Interval& other) -> void
  {
    if (!other.is_empty()) {
      enclose(other.low());
      enclose(other.high());
    }
  }

  auto isEnclosed(const double& val) const -> bool { return _low <= val && val <= _high; }
  auto isEnclosed(const Interval& other) const -> bool { return _low <= other.low() && other.high() <= _high; }

  auto width() const -> double { return is_empty() ? 0 : _high - _low; }

 private:
  double _low = 1;
  double _high = 0;
};

class TransformedRect
{
 public:
  TransformedRect() = default;
  TransformedRect(const double& x_low, const double& x_high, const double& y_low, const double& y_high)
      : _x_low(x_low), _x_high(x_high), _y_low(y_low), _y_high(y_high)
  {
  }
  TransformedRect(const Point& point, const double& radius) { makeDiamond(point, radius); }

  auto init() -> void
  {
    _x_low = _y_low = 1;
    _x_high = _y_high = 0;
  }
  auto x_low() const -> const double& { return _x_low; }
  auto x_high() const -> const double& { return _x_high; }
  auto y_low() const -> const double& { return _y_low; }
  auto y_high() const -> const double& { return _y_high; }
  auto x_low(const double& val) -> void { _x_low = val; }
  auto x_high(const double& val) -> void { _x_high = val; }
  auto y_low(const double& val) -> void { _y_low = val; }
  auto y_high(const double& val) -> void { _y_high = val; }

  auto is_empty() const -> bool
  {
    auto x_interval = Interval(_x_low, _x_high);
    auto y_interval = Interval(_y_low, _y_high);
    return x_interval.is_empty() || y_interval.is_empty();
  }
  auto makeDiamond(const Point& point, const double& radius) -> void
  {
    auto val = point.x - point.y;
    _x_low = val - radius;
    _x_high = val + radius;
    val = point.x + point.y;
    _y_low = val - radius;
    _y_high = val + radius;
  }
  auto enclose(const TransformedRect& other) -> void
  {
    if (is_empty()) {
      _x_low = other._x_low;
      _x_high = other._x_high;
      _y_low = other._y_low;
      _y_high = other._y_high;
    } else {
      _x_low = std::min(_x_low, other._x_low);
      _x_high = std::max(_x_high, other._x_high);
      _y_low = std::min(_y_low, other._y_low);
      _y_high = std::max(_y_high, other._y_high);
    }
  }

  auto width(const size_t& side) const -> double
  {
    if (side == 0) {
      return _x_high - _x_low;
    } else {
      return _y_high - _y_low;
    }
  }

  auto diameter() const -> double { return std::max(width(0), width(1)); }

  static auto intersect(const TransformedRect& trr1, const TransformedRect& trr2) -> TransformedRect;

 private:
  auto check() -> void;

  auto correction() -> void;
  double _x_low = 1;
  double _x_high = 0;
  double _y_low = 1;
  double _y_high = 0;
};
}  // namespace icts::bst
