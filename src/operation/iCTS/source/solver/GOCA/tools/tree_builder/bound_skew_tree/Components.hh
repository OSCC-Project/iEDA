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

#include "GeomOperator.hh"
#include "TimingPropagator.hh"
#include "pgl.h"
namespace icts {
namespace bst {

using Timing = TimingPropagator;
using Geom = GeomOperator;
using Pt = CtsPoint<double>;
using JoinSegment = std::vector<Pt>;
using Pts = std::vector<Pt>;
using Region = std::vector<Pt>;

template <typename T>
using Side = std::array<T, 2>;

#define FOR_EACH_SIDE(side) for (size_t side = 0; side < 2; ++side)

class BstNode
{
 public:
  BstNode() = default;
  BstNode(Node* node)
      : _name(node->get_name()),
        _sub_len(node->get_sub_len()),
        _slew_in(node->get_slew_in()),
        _cap_load(node->get_cap_load()),
        _min_delay(node->get_min_delay()),
        _max_delay(node->get_max_delay())
  {
    auto loc = node->get_location();
    auto x = loc.x();
    auto y = loc.y();
    _location = Pt(1.0 * x / Timing::getDbUnit(), 1.0 * y / Timing::getDbUnit());
    _mr.push_back(_location);
  }
  // get
  const std::string& get_name() const { return _name; }
  const Pt& get_location() const { return _location; }
  const double& get_sub_len() const { return _sub_len; }
  const double& get_cap_load() const { return _cap_load; }
  const double& get_slew_in() const { return _slew_in; }
  const double& get_min_delay() const { return _min_delay; }
  const double& get_max_delay() const { return _max_delay; }
  const double& get_edge_len() const { return _edge_len; }

  JoinSegment get_js(const size_t& side) const { return _sub_js[side]; }
  Region get_mr() const { return _mr; }

  BstNode* get_parent() const { return _parent; }
  std::vector<BstNode*> get_children() const { return {_left, _right}; }

  // set
  void set_name(const std::string& name) { _name = name; }
  void set_location(const Point& location) { _location = location; }
  void set_sub_len(const double& sub_len) { _sub_len = sub_len; }
  void set_cap_load(const double& cap_load) { _cap_load = cap_load; }
  void set_slew_in(const double& slew_in) { _slew_in = slew_in; }
  void set_min_delay(const double& min_delay) { _min_delay = min_delay; }
  void set_max_delay(const double& max_delay) { _max_delay = max_delay; }
  void set_edge_len(const double& edge_len) { _edge_len = edge_len; }

  void set_js(const size_t& side, const JoinSegment& js) { _sub_js[side] = js; }
  void set_js(const Side<JoinSegment>& sub_js) { _sub_js = sub_js; }

  void set_parent(BstNode* parent) { _parent = parent; }
  void set_children(BstNode* left, BstNode* right)
  {
    _left = left;
    _right = right;
  }

  // add
  void add_mr_point(const Pt& point) { _mr.push_back(point); }

 private:
  std::string _name;
  Pt _location;
  double _sub_len = 0;
  double _slew_in = 0;
  double _cap_load = 0;
  double _min_delay = 0;
  double _max_delay = 0;
  double _edge_len = 0;

  BstNode* _parent;
  BstNode* _left;
  BstNode* _right;
  Side<JoinSegment> _sub_js;
  Region _mr;
};

struct Match
{
  BstNode* left;
  BstNode* right;
  double merge_cost;
};

template <Numeric T>
class Interval
{
 public:
  Interval() = default;
  Interval(const T& val) : _low(val), _high(val) {}
  Interval(const T& low, const T& high) : _low(low), _high(high) {}
  Interval(const Interval& other) : _low(other._low), _high(other._high) {}

  const T& low() const { return _low; }
  const T& high() const { return _high; }

  const bool& is_empty() const { return _low > _high; }
  const bool& is_point() const { return _low == _high; }

  void enclose(const T& val)
  {
    if (is_empty()) {
      _low = val;
      _high = val;
    } else {
      _low = std::min(_low, val);
      _high = std::max(_high, val);
    }
  }
  void enclose(const Interval<T>& other)
  {
    if (!other.is_empty()) {
      enclose(other.low());
      enclose(other.high());
    }
  }

  const bool& isEnclosed(const T& val) const { return _low <= val && val <= _high; }
  const bool& isEnclosed(const Interval<T>& other) const { return _low <= other.low() && other.high() <= _high; }

  const T& width() const { return is_empty() ? 0 : _high - _low; }

 private:
  T _low = 1;
  T _high = 0;
};
}  // namespace bst
}  // namespace icts