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

#include <stdint.h>

#include <cmath>
#include <map>
#include <vector>

#include "idrc_data.h"

namespace idrc {

class DrcBasicPoint
{
 public:
  DrcBasicPoint(int x, int y, int net_id, int polygon_id, DrcBasicPoint* prev = nullptr, DrcBasicPoint* next = nullptr)
      : _x(x), _y(y), _net_id(net_id), _polygon_id(polygon_id), _prev(prev), _next(next)
  {
  }

  ~DrcBasicPoint() {}

  // getter
  int get_x() { return _x; }
  int get_y() { return _y; }
  int get_net_id() { return _net_id; }
  int get_polygon_id() { return _polygon_id; }

  DrcBasicPoint* get_prev() { return _prev; }
  DrcBasicPoint* get_next() { return _next; }

  // setter
  void set_prev(DrcBasicPoint* prev) { _prev = prev; }
  void set_next(DrcBasicPoint* next) { _next = next; }

  int distance(DrcBasicPoint* p) { return std::abs(_x - p->get_x()) + std::abs(_y - p->get_y()); }

  // p in witch direction of this point
  DrcDirection direction(DrcBasicPoint* p)
  {
    if (_x == p->get_x() && _y == p->get_y()) {
      return DrcDirection::kNone;
    }
    return _x == p->get_x() ? (_y > p->get_y() ? DrcDirection::kDown : DrcDirection::kUp)
                            : (_x > p->get_x() ? DrcDirection::kLeft : DrcDirection::kRight);
  }

 private:
  int _x;
  int _y;

  int _net_id;
  int _polygon_id;

  DrcBasicPoint* _prev = nullptr;
  DrcBasicPoint* _next = nullptr;
};

}  // namespace idrc
