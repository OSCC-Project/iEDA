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
#include <cmath>

#include "DmeNode.h"
#include "Params.h"
#include "pgl.h"

namespace icts {

struct ZstDelay {
  double _time;
  double _cap;
};

class ZstDelayFunc {
 public:
  ZstDelayFunc() = default;
  ZstDelayFunc(const Segment &ms_a, const ZstDelay &delay_a,
               const Segment &ms_b, const ZstDelay &delay_b,
               const ZstParams &params)
      : _ms_a(ms_a),
        _delay_a(delay_a),
        _ms_b(ms_b),
        _delay_b(delay_b),
        _params(params) {}

  bool edge_length(std::pair<Coordinate, Coordinate> &edge_pair) const {
    return _params.get_delay_model() == DelayModel::kLinear
               ? linear_edge_length(edge_pair)
               : elmore_edge_length(edge_pair);
  }

  ZstDelay delay() const {
    std::pair<Coordinate, Coordinate> edge_pair;
    edge_length(edge_pair);
    auto edge_a = edge_pair.first;
    auto edge_b = edge_pair.second;
    auto time_a = _delay_a._time;
    auto time_b = _delay_b._time;

    double time = std::max(time_a + edge_a, time_b + edge_b);
    double cap = 0;
    if (_params.get_delay_model() == DelayModel::kElmore) {
      auto cap_a = _delay_a._cap;
      auto cap_b = _delay_b._cap;
      auto unit_res = _params.get_unit_res();
      auto unit_cap = _params.get_unit_cap();

      time = time_a + unit_res * edge_a * (unit_cap * edge_a / 2 + cap_a);
      cap = (cap_a + unit_cap * edge_a) + (cap_b + unit_cap * edge_b);
    }
    return ZstDelay{time, cap};
  }

 private:
  bool linear_edge_length(std::pair<Coordinate, Coordinate> &edge_pair) const {
    auto time_a = _delay_a._time;
    auto time_b = _delay_b._time;

    double edge_a, edge_b;
    bool have_min_delay = false;

    auto dist = merge_segment_distance();
    if (std::abs(time_a - time_b) <= dist) {
      edge_a = (dist + time_b - time_a) / 2;
      edge_b = dist - edge_a;
      have_min_delay = true;
    } else {
      if (time_a > time_b) {
        edge_a = 0;
        edge_b = time_a - time_b;
      } else {
        edge_a = time_b - time_a;
        edge_b = 0;
      }
    }
    edge_pair.first = static_cast<Coordinate>(std::round(edge_a));
    edge_pair.second = static_cast<Coordinate>(std::round(edge_b));
    return have_min_delay;
  }
  bool elmore_edge_length(std::pair<Coordinate, Coordinate> &edge_pair) const {
    auto time_a = _delay_a._time;
    auto time_b = _delay_b._time;
    auto cap_a = _delay_a._cap;
    auto cap_b = _delay_b._cap;
    auto unit_res = _params.get_unit_res();
    auto unit_cap = _params.get_unit_cap();

    double edge_a, edge_b;
    bool have_min_delay = false;

    auto dist = merge_segment_distance();
    auto x =
        (time_b - time_a + unit_res * dist * (cap_b + unit_cap * dist / 2)) /
        (unit_res * (cap_a + cap_b + unit_cap * dist));

    if (0 <= x && x <= dist) {
      edge_a = x;
      edge_b = dist - x;
      have_min_delay = true;
    } else {
      if (x < 0) {
        edge_a = 0;
        edge_b = winding_length(time_a, time_b, cap_b);
      } else {
        edge_a = winding_length(time_b, time_a, cap_a);
        edge_b = 0;
      }
    }
    edge_pair.first = static_cast<Coordinate>(std::round(edge_a));
    edge_pair.second = static_cast<Coordinate>(std::round(edge_b));
    return have_min_delay;
  }

  Coordinate winding_length(double time1, double time2, double cap2) const {
    auto unit_res = _params.get_unit_res();
    auto unit_cap = _params.get_unit_cap();

    auto tmp = std::pow(unit_res * cap2, 2) +
               2 * unit_res * unit_cap * (time1 - time2);
    return static_cast<Coordinate>((std::sqrt(tmp) - unit_res * cap2) /
                                   (unit_res * unit_cap));
  }

  Coordinate merge_segment_distance() const {
    return pgl::manhattan_distance(_ms_a, _ms_b);
  }

 private:
  Segment _ms_a;
  ZstDelay _delay_a;
  Segment _ms_b;
  ZstDelay _delay_b;

  ZstParams _params;
};

template <typename T>
class ZstNode : public DmeNode<T> {
 public:
  ZstNode() : _merge_segment(), _edge_len(0), _delay{0, 0} {}
  ZstNode(const DmeNode<T> &dme_node) : DmeNode<T>(dme_node), _merge_segment() {
    _edge_len = -1;
  }

  Segment get_merge_segment() const { return _merge_segment; }
  double get_edge_length() const { return _edge_len; }
  ZstDelay get_delay() const { return _delay; }

  void set_merge_segment(const Segment &merge_segment) {
    _merge_segment = merge_segment;
  }
  void set_edge_length(double len) { _edge_len = len; }
  void set_delay(const ZstDelay &delay) { _delay = delay; }

  void copy_message(const ZstNode<T> &that) {
    _merge_segment = that._merge_segment;
    _edge_len = that._edge_len;
    _delay = that._delay;
  }

 private:
  Segment _merge_segment;
  double _edge_len;
  ZstDelay _delay;
};

}  // namespace icts