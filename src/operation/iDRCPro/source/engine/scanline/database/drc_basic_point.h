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
#include <map>
#include <vector>

#include "scanline_neighbour.h"

namespace idrc {

enum class DrcDirection
{
  kNone,
  kUp,
  kDown,
  kLeft,
  kRight
};

class DrcBasicPoint
{
  enum PointState : uint64_t
  {
    kIsStartPoint = 1,
    kIsEndPoint = 2,
    kCheckedOverlap = 4,
    kCheckedHorizontal = 8,
    kCheckedVertical = 16,
    kMax
  };

 public:
  DrcBasicPoint(int x, int y, int id, bool is_endpoint = true, DrcBasicPoint* prev = nullptr, DrcBasicPoint* next = nullptr)
      : _x(x), _y(y), _id(id), _prev(prev), _next(next)
  {
    set_is_endpoint(is_endpoint);
  }

  ~DrcBasicPoint()
  {
    for (auto& neighbour : _neighbours) {
      delete neighbour.second;
      neighbour.second = nullptr;
    }
  }

  bool operator<(DrcBasicPoint& other)
  {
    if (this->get_x() > other.get_x()) {
      return false;
    } else if (this->get_x() < other.get_x()) {
      return true;
    } else {
      if (this->get_y() < other.get_y()) {
        return true;
      } else {
        return false;
      }
    }
  }

  // getter
  int get_x() { return _x; }
  int get_y() { return _y; }
  int get_id() { return _id; }

  DrcBasicPoint* get_prev() { return _prev; }
  DrcBasicPoint* get_next() { return _next; }

  ScanlineNeighbour* get_neighbour(DrcDirection direction)
  {
    if (_neighbours.find(direction) != _neighbours.end()) {
      return _neighbours[direction];
    }
    return nullptr;
  }

  // setter
  void set_x(int x) { _x = x; }
  void set_y(int y) { _y = y; }
  void set_id(int id) { _id = id; }

  void set_prev(DrcBasicPoint* prev) { _prev = prev; }
  void set_next(DrcBasicPoint* next) { _next = next; }

  void set_neighbour(DrcDirection direction, ScanlineNeighbour* neighbour)
  {
    if (_neighbours.find(direction) != _neighbours.end()) {
      delete _neighbours[direction];
    }
    _neighbours[direction] = neighbour;
  }

  void set_states(bool value, PointState state) { _states = value ? _states | state : _states & (~state); }
  bool get_states(PointState state) { return (_states & state) > 0; }
  void set_is_endpoint(bool is_endpoint) { set_states(is_endpoint, PointState::kIsEndPoint); }
  /// true = is a end point
  bool is_endpoint() { return get_states(PointState::kIsEndPoint); }
  void set_as_start() { set_states(true, PointState::kIsStartPoint); }
  /// true = the left lowest point for a polygon;
  bool is_start() { return get_states(PointState::kIsStartPoint); }
  void set_checked_overlap() { set_states(true, PointState::kCheckedOverlap); }
  /// true = has been checked for overlap condition
  bool is_overlap_checked() { return get_states(PointState::kCheckedOverlap); }
  void set_checked_vertical() { set_states(true, PointState::kCheckedVertical); }
  /// true = has been checked in vertical direction
  bool is_vertical_checked() { return get_states(PointState::kCheckedVertical); }
  void set_checked_horizontal() { set_states(true, PointState::kCheckedHorizontal); }
  /// true = has been checked in horizontal direction
  bool is_horizontal_checked() { return get_states(PointState::kCheckedHorizontal); }

  DrcBasicPoint* nextEndpoint()
  {
    DrcBasicPoint* next = _next;
    while (next && !next->is_endpoint()) {
      next = next->get_next();
    }
    return next;
  }

  DrcBasicPoint* prevEndpoint()
  {
    DrcBasicPoint* prev = _prev;
    while (prev && !prev->is_endpoint()) {
      prev = prev->get_prev();
    }
    return prev;
  }

  int distance(DrcBasicPoint* p) { return std::abs(_x - p->get_x()) + std::abs(_y - p->get_y()); }

  DrcDirection direction(DrcBasicPoint* p)
  {
    if (_x == p->get_x() && _y == p->get_y()) {
      return DrcDirection::kNone;
    }
    return _x == p->get_x() ? (_y > p->get_y() ? DrcDirection::kUp : DrcDirection::kDown)
                            : (_x > p->get_x() ? DrcDirection::kRight : DrcDirection::kLeft);
  }

 private:
  int _x;
  int _y;
  int _id;

  uint64_t _states = 0;

  DrcBasicPoint* _prev = nullptr;
  DrcBasicPoint* _next = nullptr;

  std::map<DrcDirection, ScanlineNeighbour*> _neighbours;
};

}  // namespace idrc
