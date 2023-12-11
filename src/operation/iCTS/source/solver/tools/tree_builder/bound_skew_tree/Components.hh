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
 */
#pragma once

#include "TimingPropagator.hh"
#include "log/Log.hh"
namespace icts {
namespace bst {
/**
 * @brief Tool namespace
 *
 */
using Timing = TimingPropagator;

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
#define FOR_EACH_SIDE(side) for (size_t side = 0; side < 2; ++side)

template <Numeric T1, Numeric T2>
constexpr static bool Equal(const T1& a, const T2& b, const double& epsilon = kEpsilon)
{
  return std::abs(a - b) < epsilon;
}

class Pt
{
 public:
  Pt() = default;
  Pt(const double& t_x, const double& t_y, const double& t_max, const double& t_min, const double& t_val)
      : x(t_x), y(t_y), max(t_max), min(t_min), val(t_val)
  {
  }
  Pt(const double& t_x, const double& t_y) : x(t_x), y(t_y), max(0), min(0), val(0) {}

  Pt operator+(const Pt& other) const { return Pt(x + other.x, y + other.y); }
  Pt operator-(const Pt& other) const { return Pt(x - other.x, y - other.y); }
  Pt operator*(const double& scale) const { return Pt(x * scale, y * scale); }
  Pt operator/(const double& scale) const { return Pt(x / scale, y / scale); }
  Pt operator+=(const Pt& other)
  {
    x += other.x;
    y += other.y;
    return *this;
  }
  Pt operator-=(const Pt& other)
  {
    x -= other.x;
    y -= other.y;
    return *this;
  }
  Pt operator*=(const double& scale)
  {
    x *= scale;
    y *= scale;
    return *this;
  }
  Pt operator/=(const double& scale)
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
using JoinSegment = std::vector<Pt>;
using Pts = std::vector<Pt>;
using Region = std::vector<Pt>;
using Line = std::array<Pt, 2>;
using PtPair = std::array<Pt, 2>;
template <typename T>
using Side = std::array<T, 2>;

class Area
{
 public:
  Area(const size_t& id) { _name = CTSAPIInst.toString("steiner_", id); };
  Area(Node* node) : _name(node->get_name())
  {
    _pattern = node->get_pattern();
    if (_pattern == RCPattern::kSingle) {
      _pattern = static_cast<RCPattern>(1 + std::rand() % 2);
    }
    auto loc = node->get_location();
    auto x = 1.0 * loc.x() / Timing::getDbUnit();
    auto y = 1.0 * loc.y() / Timing::getDbUnit();
    _location = Pt(x, y, node->get_max_delay(), node->get_min_delay(), node->get_cap_load());
    _sub_len = 1.0 * node->get_sub_len() / Timing::getDbUnit();
    _cap_load = node->get_cap_load();
    if (node->isPin() && node->isLoad()) {
      _mr.push_back(_location);
      _convex_hull.push_back(_location);
    }
  }

  Area(const std::string& name, const double& x, const double& y, const double& cap_load) : _name(name)
  {
    _location = Pt(x, y, 0, 0, cap_load);
    _cap_load = cap_load;
    _mr.push_back(_location);
    _convex_hull.push_back(_location);
  }
  // get
  const std::string& get_name() const { return _name; }
  const double& get_cap_load() const { return _cap_load; }
  const double& get_sub_len() const { return _sub_len; }
  const double& get_edge_len(const size_t& side) const { return _edge_len[side]; }
  const double& get_radius() const { return _radius; }
  const RCPattern& get_pattern() const { return _pattern; }

  const Pt& get_location() const { return _location; }
  Area* get_parent() const { return _parent; }
  Area* get_left() const { return _left; }
  Area* get_right() const { return _right; }
  Line get_line(const size_t& side) const { return _lines[side]; }
  Side<Line> get_lines() const { return _lines; }
  Region get_mr() const { return _mr; }
  std::vector<Line> getMrLines() const
  {
    std::vector<Line> lines;
    for (size_t i = 0; i < _mr.size(); ++i) {
      auto j = (i + 1) % _mr.size();
      lines.push_back({_mr[i], _mr[j]});
    }
    return lines;
  }

  Region get_convex_hull() const { return _convex_hull; }
  std::vector<Line> getConvexHullLines() const
  {
    std::vector<Line> lines;
    for (size_t i = 0; i < _convex_hull.size(); ++i) {
      auto j = (i + 1) % _convex_hull.size();
      lines.push_back({_convex_hull[i], _convex_hull[j]});
    }
    return lines;
  }
  // set
  void set_name(const std::string& name) { _name = name; }
  void set_cap_load(const double& cap_load) { _cap_load = cap_load; }
  void set_sub_len(const double& sub_len) { _sub_len = sub_len; }
  void set_edge_len(const size_t& side, const double& edge_len) { _edge_len[side] = edge_len; }
  void set_radius(const double& radius) { _radius = radius; }
  void set_pattern(const RCPattern& pattern) { _pattern = pattern; }

  void set_location(const Pt& location) { _location = location; }
  void set_parent(Area* parent) { _parent = parent; }
  void set_left(Area* left) { _left = left; }
  void set_right(Area* right) { _right = right; }
  void set_line(const size_t& side, const Line& line) { _lines[side] = line; }
  void set_mr(const Region& mr) { _mr = mr; }
  void set_convex_hull(const Region& convex_hull) { _convex_hull = convex_hull; }

  // add
  void add_mr_point(const Pt& point) { _mr.push_back(point); }
  void add_convex_hull_point(const Pt& point) { _convex_hull.push_back(point); }

 private:
  std::string _name;
  double _cap_load = 0;
  double _sub_len = 0;
  Side<double> _edge_len = {0, 0};
  double _radius = 0;
  RCPattern _pattern = RCPattern::kHV;

  Pt _location;
  Area* _parent = nullptr;
  Area* _left = nullptr;
  Area* _right = nullptr;
  Side<Line> _lines;
  Region _mr;
  Region _convex_hull;
};

struct Match
{
  Area* left;
  Area* right;
  double merge_cost;
};

class Interval
{
 public:
  Interval() = default;
  Interval(const double& val) : _low(val), _high(val) {}
  Interval(const double& low, const double& high) : _low(low), _high(high) {}
  Interval(const Interval& other) : _low(other._low), _high(other._high) {}

  const double& low() const { return _low; }
  const double& high() const { return _high; }

  bool is_empty() const { return _low > _high; }
  bool is_point() const { return _low == _high; }

  void enclose(const double& val)
  {
    if (is_empty()) {
      _low = val;
      _high = val;
    } else {
      _low = std::min(_low, val);
      _high = std::max(_high, val);
    }
  }
  void enclose(const Interval& other)
  {
    if (!other.is_empty()) {
      enclose(other.low());
      enclose(other.high());
    }
  }

  bool isEnclosed(const double& val) const { return _low <= val && val <= _high; }
  bool isEnclosed(const Interval& other) const { return _low <= other.low() && other.high() <= _high; }

  double width() const { return is_empty() ? 0 : _high - _low; }

 private:
  double _low = 1;
  double _high = 0;
};

class Trr
{
 public:
  Trr() = default;
  Trr(const double& x_low, const double& x_high, const double& y_low, const double& y_high)
      : _x_low(x_low), _x_high(x_high), _y_low(y_low), _y_high(y_high)
  {
  }
  Trr(const Pt& point, const double& radius) { makeDiamond(point, radius); }

  void init()
  {
    _x_low = _y_low = 1;
    _x_high = _y_high = 0;
  }
  const double& x_low() const { return _x_low; }
  const double& x_high() const { return _x_high; }
  const double& y_low() const { return _y_low; }
  const double& y_high() const { return _y_high; }
  void x_low(const double& val) { _x_low = val; }
  void x_high(const double& val) { _x_high = val; }
  void y_low(const double& val) { _y_low = val; }
  void y_high(const double& val) { _y_high = val; }

  bool is_empty() const
  {
    auto x_interval = Interval(_x_low, _x_high);
    auto y_interval = Interval(_y_low, _y_high);
    return x_interval.is_empty() || y_interval.is_empty();
  }
  void makeDiamond(const Pt& point, const double& radius)
  {
    auto val = point.x - point.y;
    _x_low = val - radius;
    _x_high = val + radius;
    val = point.x + point.y;
    _y_low = val - radius;
    _y_high = val + radius;
  }
  void enclose(const Trr& other)
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

  double width(const size_t& side) const
  {
    if (side == 0) {
      return _x_high - _x_low;
    } else {
      return _y_high - _y_low;
    }
  }

  double diameter() const { return std::max(width(0), width(1)); }

  Trr intersect(const Trr& trr1, const Trr& trr2)
  {
    auto x_low = std::max(trr1._x_low, trr2._x_low);
    auto x_high = std::min(trr1._x_high, trr2._x_high);
    auto y_low = std::max(trr1._y_low, trr2._y_low);
    auto y_high = std::min(trr1._y_high, trr2._y_high);
    auto trr = Trr(x_low, x_high, y_low, y_high);
    trr.check();
    return trr;
  }

 private:
  void check()
  {
    correction();
    LOG_FATAL_IF(is_empty()) << "TRR is empty, which x_low: " << _x_low << ", x_high: " << _x_high << ", y_low: " << _y_low
                             << ", y_high: " << _y_high;
  }

  void correction()
  {
    auto temp_low = _x_low;
    auto temp_high = _x_high;
    if (Equal(temp_low, temp_high)) {
      _x_low = _x_high = (temp_low + temp_high) / 2;
    }
    temp_low = _y_low;
    temp_high = _y_high;
    if (Equal(temp_low, temp_high)) {
      _y_low = _y_high = (temp_low + temp_high) / 2;
    }
  }
  double _x_low = 1;
  double _x_high = 0;
  double _y_low = 1;
  double _y_high = 0;
};
}  // namespace bst
}  // namespace icts