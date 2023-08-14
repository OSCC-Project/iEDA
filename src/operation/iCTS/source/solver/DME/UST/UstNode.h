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
#include "DmeNode.h"
#include "Params.h"
#include "Skew.h"
#include "pgl.h"

namespace icts {
typedef double PropagationTime;

class UstDelay {
 public:
  UstDelay() = default;

  UstDelay(const PropagationTime& left_t, const PropagationTime& right_t,
           const int& left_id, const int& right_id, const double& total_cap = 0)
      : _left_t(left_t),
        _right_t(right_t),
        _left_id(left_id),
        _right_id(right_id),
        _total_cap(total_cap) {}

  UstDelay(const UstDelay& other) {
    _left_t = other._left_t;
    _right_t = other._right_t;
    _left_id = other._left_id;
    _right_id = other._right_id;
    _total_cap = other._total_cap;
  }

  PropagationTime get_skew() const { return _left_t - _right_t; }
  PropagationTime get_left_delay() const { return _left_t; }
  int get_left_id() const { return _left_id; }
  PropagationTime get_right_delay() const { return _right_t; }
  int get_right_id() const { return _right_id; }
  double get_total_cap() const { return _total_cap; }

  bool operator==(const UstDelay& other) const {
    return _left_t == other._left_t && _left_id == other._left_id &&
           _right_t == other._right_t && _right_id == other._right_id &&
           _total_cap == other._total_cap;
  }

 private:
  PropagationTime _left_t = 0;   // the left delay time to sink node
  PropagationTime _right_t = 0;  // the right delay time to sink node
  int _left_id = -1;
  int _right_id = -1;
  double _total_cap = 0;
};

class UstDelayFunc {
 public:
  typedef int coord_type;
  typedef CtsPoint<coord_type> point_type;
  typedef CtsSegment<coord_type> segment_type;
  UstDelayFunc() = default;
  UstDelayFunc(const segment_type& js_a, const UstDelay& delay_a,
               const segment_type& js_b, const UstDelay& delay_b,
               const UstParams& params)
      : _js_a(js_a),
        _delay_a(delay_a),
        _js_b(js_b),
        _delay_b(delay_b),
        _params(params) {}

  UstDelayFunc(const UstDelayFunc&) = default;
  ~UstDelayFunc() = default;

  UstDelay delay(const point_type& point) const {
    return _params.get_delay_model() == DelayModel::kLinear
               ? linearDelay(point)
               : elmoreDelay(point);
  }

  PropagationTime linearFunc(const int& dist) const {
    // TBD
    return 0;
  }

  UstDelay linearDelay(const point_type& point) const {
    auto left_delay =
        pgl::manhattan_distance(point, _js_a) + _delay_a.get_left_delay();
    auto right_delay =
        pgl::manhattan_distance(point, _js_b) + _delay_b.get_right_delay();
    return UstDelay(left_delay, right_delay, _delay_a.get_left_id(),
                    _delay_b.get_right_id());
  }

  PropagationTime elmoreFunc(const PropagationTime& sub_t, const double& length,
                             const double& sub_cap) const {
    auto unit_cap = _params.get_unit_cap();
    auto unit_res = _params.get_unit_res();
    return sub_t + length * unit_res * (unit_cap * length / 2 + sub_cap);
  }

  UstDelay elmoreDelay(const point_type& point,
                       const coord_type& left_add_dist = 0,
                       const coord_type& right_add_dist = 0) const {
    auto sub_left_t = _delay_a.get_left_delay();
    auto sub_right_t = _delay_b.get_right_delay();
    auto sub_left_cap = _delay_a.get_total_cap();
    auto sub_right_cap = _delay_b.get_total_cap();
    auto left_length =
        static_cast<double>(pgl::manhattan_distance(point, _js_a) +
                            left_add_dist) /
        _params.get_db_unit();
    auto right_length =
        static_cast<double>(pgl::manhattan_distance(point, _js_b) +
                            right_add_dist) /
        _params.get_db_unit();
    auto left_delay = elmoreFunc(sub_left_t, left_length, sub_left_cap);
    auto right_delay = elmoreFunc(sub_right_t, right_length, sub_right_cap);
    auto unit_cap = _params.get_unit_cap();
    auto total_cap = (sub_left_cap + unit_cap * left_length) +
                     (sub_right_cap + unit_cap * right_length);

    return UstDelay(left_delay, right_delay, _delay_a.get_left_id(),
                    _delay_b.get_right_id(), total_cap);
  }
  coord_type detourLeft(const PropagationTime& left_bound) const {
    // elmore
    auto unit_res = _params.get_unit_res();
    auto unit_cap = _params.get_unit_cap();
    auto sub_left_t = _delay_a.get_left_delay();
    auto sub_right_t = _delay_b.get_right_delay();
    auto sub_left_cap = _delay_a.get_total_cap();
    auto snake_length =
        (std::sqrt(sub_left_cap * sub_left_cap +
                   2 * unit_cap * (left_bound + sub_right_t - sub_left_t) /
                       unit_res) -
         sub_left_cap) /
        unit_cap;
    return static_cast<coord_type>(
        std::ceil(snake_length * _params.get_db_unit()));
  }

  coord_type detourRight(const PropagationTime& right_bound) const {
    // elmore
    auto unit_res = _params.get_unit_res();
    auto unit_cap = _params.get_unit_cap();
    auto sub_left_t = _delay_a.get_left_delay();
    auto sub_right_t = _delay_b.get_right_delay();
    auto sub_right_cap = _delay_b.get_total_cap();
    auto snake_length =
        (std::sqrt(sub_right_cap * sub_right_cap +
                   2 * unit_cap * (sub_left_t - sub_right_t - right_bound) /
                       unit_res) -
         sub_right_cap) /
        unit_cap;
    return static_cast<coord_type>(
        std::ceil(snake_length * _params.get_db_unit()));
  }

  double elmoreEndpointLength(const PropagationTime& skew,
                              const segment_type& edge) {
    auto length =
        static_cast<double>(pgl::manhattan_distance(edge.low(), edge.high())) /
        _params.get_db_unit();
    auto l1 = static_cast<double>(pgl::manhattan_distance(_js_a, edge)) /
              _params.get_db_unit();
    auto l2 = static_cast<double>(pgl::manhattan_distance(_js_b, edge)) /
              _params.get_db_unit();
    auto t1 = _delay_a.get_left_delay();
    auto t2 = _delay_b.get_right_delay();
    auto c1 = _delay_a.get_total_cap();
    auto c2 = _delay_b.get_total_cap();
    auto unit_cap = _params.get_unit_cap();
    auto unit_res = _params.get_unit_res();
    auto part_1 = (skew - t1 + t2) / unit_res;
    auto part_2 = unit_cap / 2 * (length + l2 + l1) * (length + l2 - l1) +
                  c2 * l2 + c2 * length - c1 * l1;
    auto part_3 = unit_cap * (length + l1 + l2) + c1 + c2;
    auto x = (part_1 + part_2) / part_3;
    return x * _params.get_db_unit();
  }

  std::vector<point_type> feasibleSkewEndpoint(const segment_type& edge,
                                               const SkewRange& fsr) {
    std::vector<point_type> endpoints;
    if (pgl::manhattan_arc(edge)) {
      auto skew = delay(edge.low()).get_skew();
      if (fsr.first <= skew && skew <= fsr.second) {
        endpoints.emplace_back(edge.low());
        endpoints.emplace_back(edge.high());
      }
      return endpoints;
    }
    auto skew_lb = fsr.first;
    auto skew_ub = fsr.second;
    auto endpoint_l =
        static_cast<coord_type>(std::ceil(elmoreEndpointLength(skew_lb, edge)));
    auto endpoint_r = static_cast<coord_type>(
        std::floor(elmoreEndpointLength(skew_ub, edge)));
    if (endpoint_l > endpoint_r) {
      return endpoints;
    }
    auto dist = pgl::manhattan_distance(edge.low(), edge.high());
    auto edge_points = pgl::edge_to_points(edge);
    if (0 <= endpoint_l && endpoint_l <= dist) {
      endpoints.emplace_back(edge_points[endpoint_l]);
    }
    if (0 <= endpoint_r && endpoint_r <= dist) {
      endpoints.emplace_back(edge_points[endpoint_r]);
    }
    if (endpoint_l <= 0 && 0 <= endpoint_r) {
      endpoints.emplace_back(edge_points[0]);
    }
    if (endpoint_l <= dist && dist <= endpoint_r) {
      endpoints.emplace_back(edge_points[dist]);
    }
    return endpoints;
  }

 private:
  segment_type _js_a;
  UstDelay _delay_a;
  segment_type _js_b;
  UstDelay _delay_b;
  UstParams _params;
};

template <typename T>
class UstNode : public DmeNode<T> {
 public:
  UstNode() : DmeNode<T>(), _merge_region() { _edge_len = -1; }

  UstNode(const DmeNode<T>& dme_node) : DmeNode<T>(dme_node), _merge_region() {
    _edge_len = -1;
  }

  Segment& get_joining_segment() { return _join_seg; }
  double get_edge_length() const { return _edge_len; }
  Polygon& get_merge_region() { return _merge_region; }
  void set_joining_segment(const Segment& join_seg) { _join_seg = join_seg; }
  void set_edge_length(const double& edge_len) { _edge_len = edge_len; }
  void set_merge_region(const Polygon& merge_region) {
    _merge_region = merge_region;
  }
  void set_merge_region(const std::initializer_list<Point>& points) {
    _merge_region.set(points);
  }
  void set_delay_func(const UstDelayFunc& delay_func) {
    _delay_func = delay_func;
  }
  void setSubWirelength(const int& sub_wirelength) {
    DataTraits<T>::setSubWirelength(DmeNode<T>::_data, sub_wirelength);
  }
  void set_extra_wirelength(const int& extra_wirelength) {
    _extra_wirelength = extra_wirelength;
  }
  // get the edge which the joining segment belongs in the polygon.
  Segment subordinateEdge(const Segment& join_seg) const {
    std::vector<Segment> segs = _merge_region.get_edges();
    for (auto& seg : segs) {
      if (gtl::contains(seg, join_seg) == true) {
        return seg;
      }
    }
    assert(false);
  }

  bool find_point_delay(UstDelay& delay, const Point& point) const {
    auto it = _delays.find(point);
    if (it == _delays.end()) {
      return false;
    }
    delay = it->second;
    return true;
  }

  void get_point_delay(UstDelay& delay, const Point& point) const {
    if (!find_point_delay(delay, point)) {
      delay = _delay_func.delay(point);
    }
  }

  void add_point_delay(const Point& point, const UstDelay& delay) {
    if (_delays.count(point) == 0) {
      _delays[point] = delay;
    } else {
      update_point_delay(point, delay);
    }
  }

  void update_point_delay(const Point& point, const UstDelay& delay) {
    _delays[point] = delay;
  }

  bool edge_len_determined() const { return _edge_len != -1; }
  int get_extra_wirelength() const { return _extra_wirelength; }

 private:
  Polygon _merge_region;  // merge region
  Segment _join_seg;      // joining segment
  double _edge_len;       // the edge length to the parent node
  // the delay time of vertexs and turn points of the merge region
  std::map<Point, UstDelay> _delays;
  UstDelayFunc _delay_func;
  int _extra_wirelength = 0;
};
}  // namespace icts
