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

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <stack>
#include <utility>
#include <vector>

#include "DmeNode.h"
#include "Params.h"
#include "Traits.h"
#include "pgl.h"

namespace icts {
using std::map;
using std::stack;
using std::vector;
template <typename T>
using Pair = std::pair<T, T>;

class RectilinearTag;
class ManhattanArcTag;

enum class BstDelayType { kMAX, kMIN };
enum class DelayFuncType {
  kOLD,
  kNEW
};  // kNEW:new delay function, kOLD: old delay function

struct BstDelay {
  typedef double coord_type;

  coord_type _max_t;  // the maximum delay time to sink node
  coord_type _min_t;  // the minimun delay time to sink node

  coord_type get_delay_time(BstDelayType type) const {
    return type == BstDelayType::kMAX ? _max_t : _min_t;
  }
  coord_type skew() const { return _max_t - _min_t; }
  bool operator==(const BstDelay &other) const {
    return _max_t == other._max_t && _min_t == other._min_t;
  }
};

struct PointDelay {
  Point _point;
  BstDelay _delay;
};

template <typename Point>
inline std::ostream &operator<<(std::ostream &os, const PointDelay &pd) {
  std::cout << pd._point << std::endl;
  std::cout << "{" << pd._delay._max_t << ", " << pd._delay._min_t << "}"
            << std::endl;
  return os;
}

class BstDelayFunc {
 public:
  typedef double coord_type;
  typedef CtsPoint<coord_type> point_type;

  BstDelayFunc(const point_type &point_a, const BstDelay &delay_a,
               const point_type &point_b, const BstDelay &delay_b,
               DelayFuncType type = DelayFuncType::kNEW)
      : _point_a(point_a),
        _delay_a(delay_a),
        _point_b(point_b),
        _delay_b(delay_b) {
    _func_type = type;
  }
  BstDelayFunc(const PointDelay &a, const PointDelay &b,
               DelayFuncType type = DelayFuncType::kNEW)
      : _point_a(a._point),
        _delay_a(a._delay),
        _point_b(b._point),
        _delay_b(b._delay) {
    _func_type = type;
  }
  BstDelayFunc(const BstDelayFunc &) = default;
  ~BstDelayFunc() = default;

  BstDelay delayTime(coord_type offset) const {
    return _func_type == DelayFuncType::kNEW ? newDelayTime(offset)
                                             : oldDelayTime(offset);
  }
  BstDelay delayTime(const point_type &point) const {
    return delayTime(pgl::manhattan_distance(_point_a, point));
  }

  // PointDelay pointDelay(const point_type &point) const {
  //     return PointDelay(point, delayTime(point));
  // }

  point_type skewTurnPoint(BstDelayType type) const {
    return _func_type == DelayFuncType::kNEW ? newSkewTurnPoint(type)
                                             : oldSkewTurnPoint(type);
  }

  Pair<point_type> skewTurnPoint() const {
    auto turn_point_max = skewTurnPoint(BstDelayType::kMAX);
    auto turn_point_min = skewTurnPoint(BstDelayType::kMIN);
    return Pair<point_type>(turn_point_max, turn_point_min);
  }

  // PointDelay skewTurnPointDelay(BstDelayType type) const {
  //     auto turn_point = skewTurnPoint(type);
  //     return PointDelay{turn_point, delayTime(turn_point)};
  // }

  coord_type skew(coord_type offset) const {
    BstDelay delay = delayTime(offset);
    return delay._max_t - delay._min_t;
  }

  coord_type minSkew() const {
    auto dist = pgl::manhattan_distance(_point_a, _point_b);
    CtsInterval<coord_type> dist_intval(0, dist);

    auto max_offset = (dist + _delay_b._max_t - _delay_a._max_t) / 2;
    if (gtl::contains(dist_intval, max_offset)) {
      return skew(max_offset);
    }
    auto min_offset = (dist + _delay_b._min_t - _delay_a._min_t) / 2;
    if (gtl::contains(dist_intval, min_offset)) {
      return skew(min_offset);
    }
    return std::min(skew(0), skew(dist));
  }

  // The line segment consisting of two points is required to be horizontal or
  // vertical
  bool boundPoints(vector<point_type> &points, coord_type skew_bound) const {
    assert(_point_a.x() == _point_b.x() || _point_a.y() == _point_b.y());

    std::pair<coord_type, coord_type> offsets;
    if (boundPointOffset(offsets, skew_bound) == false) {
      return false;
    }

    points.emplace_back(makePoint(offsets.first));
    points.emplace_back(makePoint(offsets.second));
    return true;
  }

  bool boundPoints(vector<point_type> &points, coord_type skew_bound,
                   const point_type &inflection_point) const {
    std::pair<coord_type, coord_type> offsets;
    if (boundPointOffset(offsets, skew_bound) == false) {
      return false;
    }
    auto dist = pgl::manhattan_distance(_point_a, _point_b);

    CtsInterval<coord_type> interval(0, dist);
    if (!pgl::contains(interval, offsets.first)) {
      offsets.first = 0;
    }
    if (!pgl::contains(interval, offsets.second)) {
      offsets.second = dist;
    }
    auto point_l = makePoint(offsets.first, inflection_point);
    auto point_r = makePoint(offsets.second, inflection_point);
    points.emplace_back(point_l);
    points.emplace_back(point_r);
    return true;
  }

 private:
  bool boundPointOffset(Pair<coord_type> &offset_pair,
                        coord_type skew_bound) const {
    if (minSkew() > skew_bound) {
      return false;
    }
    auto dist = pgl::manhattan_distance(_point_a, _point_b);
    offset_pair.first = (skew_bound - skew(0)) / -2;
    offset_pair.second = dist + (skew_bound - skew(dist)) / 2;

    return true;
  }

  // The line segment consisting of two points is required to be horizontal or
  // vertical
  point_type makePoint(coord_type offset) const {
    coord_type dist = pgl::manhattan_distance(_point_a, _point_b);
    auto x = _point_a.x() + (_point_b.x() - _point_a.x()) / dist * offset;
    auto y = _point_a.y() + (_point_b.y() - _point_a.y()) / dist * offset;
    return point_type(x, y);
  }

  point_type makePoint(coord_type offset,
                       const point_type &inflection_point) const {
    auto start_point = _point_a;
    auto end_point = inflection_point;
    auto dist = pgl::manhattan_distance(start_point, end_point);

    if (offset > dist) {
      offset = offset - dist;
      start_point = inflection_point;
      end_point = _point_b;
      dist = pgl::manhattan_distance(start_point, end_point);
    }

    auto x =
        start_point.x() + (end_point.x() - start_point.x()) / dist * offset;
    auto y =
        start_point.y() + (end_point.y() - start_point.y()) / dist * offset;
    return point_type(x, y);
  }

  BstDelay newDelayTime(coord_type offset) const {
    auto dist = pgl::manhattan_distance(_point_a, _point_b);
    assert(0 <= offset && offset <= dist);
    auto max_t =
        std::max(_delay_a._max_t + offset, _delay_b._max_t + dist - offset);
    auto min_t =
        std::min(_delay_a._min_t + offset, _delay_b._min_t + dist - offset);
    return BstDelay{max_t, min_t};
  }

  // offset range: [0, dist], dist: the distance between two points
  // The line segment consisting of two points is required to be horizontal or
  // vertical
  BstDelay oldDelayTime(coord_type offset) const {
    auto dist = pgl::manhattan_distance(_point_a, _point_b);
    assert(0 <= offset && offset <= dist);
    auto max_t =
        std::max(_delay_a._max_t - offset, _delay_b._max_t - dist + offset);
    auto min_t =
        std::min(_delay_a._min_t + offset, _delay_b._min_t + dist - offset);
    return BstDelay{max_t, min_t};
  }

  point_type newSkewTurnPoint(BstDelayType type) const {
    auto dist = pgl::manhattan_distance(_point_a, _point_b);
    auto max_offset = (dist + _delay_b._max_t - _delay_a._max_t) / 2;
    auto min_offset = (dist + _delay_b._min_t - _delay_a._min_t) / 2;
    auto offset = (type == BstDelayType::kMAX) ? max_offset : min_offset;

    return makePoint(offset);
  }

  point_type oldSkewTurnPoint(BstDelayType type) const {
    if (pgl::manhattan_arc(_point_a, _point_b)) {
      return type == BstDelayType::kMAX ? _point_a : _point_b;
    }

    auto dist = pgl::manhattan_distance(_point_a, _point_b);
    auto max_offset = (dist + _delay_a._max_t - _delay_b._max_t) / 2;
    auto min_offset = (dist + _delay_b._min_t - _delay_a._min_t) / 2;
    auto offset = (type == BstDelayType::kMAX) ? max_offset : min_offset;

    return makePoint(offset);
  }

 private:
  point_type _point_a;
  BstDelay _delay_a;
  point_type _point_b;
  BstDelay _delay_b;

  DelayFuncType _func_type;
};

template <typename T>
class BstNode : public DmeNode<T> {
 public:
  BstNode() : DmeNode<T>(), _merge_region() { _edge_len = -1; }

  BstNode(const DmeNode<T> &dme_node) : DmeNode<T>(dme_node), _merge_region() {
    _edge_len = -1;
  }

  Segment &get_joining_segment() { return _join_seg; }
  double get_edge_length() const { return _edge_len; }
  Polygon &get_merge_region() { return _merge_region; }

  void set_joining_segment(const Segment &join_seg) { _join_seg = join_seg; }
  void set_edge_length(double edge_len) { _edge_len = edge_len; }
  void set_merge_region(const Polygon &merge_region) {
    _merge_region = merge_region;
  }
  void set_merge_region(const std::initializer_list<Point> &points) {
    _merge_region.set(points);
  }

  void copy_message(const BstNode<T> &that) {
    _merge_region = that._merge_region;
    _join_seg = that._join_seg;
    _edge_len = that._edge_len;
    _delays = that._delays;
  }

  // get the edge which the joining segment belongs in the polygon.
  Segment subordinateEdge(const Segment &join_seg) const {
    vector<Segment> segs = _merge_region.get_edges();
    for (auto &seg : segs) {
      if (gtl::contains(seg, join_seg) == true) {
        return seg;
      }
    }
    assert(false);
  }

  bool get_point_delay(BstDelay &delay, const Point &point) const {
    auto it = _delays.find(point);
    if (it == _delays.end()) {
      return false;
    }
    delay = it->second;
    return true;
  }

  bool get_point_delay(Pair<PointDelay> &point_delay,
                       const Segment &edge) const {
    BstDelay delay_l, delay_h;
    assert(get_point_delay(delay_l, edge.low()));
    assert(get_point_delay(delay_h, edge.high()));
    point_delay.first = PointDelay{edge.low(), delay_l};
    point_delay.second = PointDelay{edge.high(), delay_h};
    return true;
  }

  void add_point_delay(const Point &point, const BstDelay &delay) {
    if (_delays.count(point) == 0) {
      _delays[point] = delay;
    }
  }
  void update_point_delay(const Point &point, const BstDelay &delay) {
    _delays[point] = delay;
  }
  bool edge_len_determined() const { return _edge_len != -1; }

 private:
  Polygon _merge_region;  // merge region
  Segment _join_seg;      // joining segment
  double _edge_len;       // the edge length to the parent node
  // the delay time of vertexs and turn points of the merge region
  map<Point, BstDelay> _delays;
};

}  // namespace icts
