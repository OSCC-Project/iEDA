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

#include "drc_basic_point.h"
#include "idrc_data.h"

namespace idrc {

class DrcBasicRect
{
 public:
  DrcBasicRect() {}
  DrcBasicRect(DrcCoordinate lb, DrcCoordinate ur) : _lb(lb), _ur(ur) {}
  DrcBasicRect(int lb_x, int lb_y, int ur_x, int ur_y) : _lb(DrcCoordinate(lb_x, lb_y)), _ur(DrcCoordinate(ur_x, ur_y)) {}
  ~DrcBasicRect() {}

  // getter
  DrcCoordinate get_lb() { return _lb; }
  DrcCoordinate get_ur() { return _ur; }

  // setter
  void set_lb(DrcCoordinate lb) { _lb = lb; }
  void set_ur(DrcCoordinate ur) { _ur = ur; }

  void set_lb_x(int x) { _lb.set_x(x); }
  void set_lb_y(int y) { _lb.set_y(y); }
  void set_ur_x(int x) { _ur.set_x(x); }
  void set_ur_y(int y) { _ur.set_y(y); }

  void addPoint(DrcCoordinate p)
  {
    _lb.set_x(std::min(_lb.get_x(), p.get_x()));
    _lb.set_y(std::min(_lb.get_y(), p.get_y()));
    _ur.set_x(std::max(_ur.get_x(), p.get_x()));
    _ur.set_y(std::max(_ur.get_y(), p.get_y()));
  }

 private:
  DrcCoordinate _lb;
  DrcCoordinate _ur;
};

}  // namespace idrc
