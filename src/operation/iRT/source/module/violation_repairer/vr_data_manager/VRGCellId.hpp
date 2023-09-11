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

#include "RTU.hpp"

namespace irt {

class VRGCellId
{
 public:
  VRGCellId() = default;
  VRGCellId(const irt_int x, const irt_int y)
  {
    _x = x;
    _y = y;
  }
  ~VRGCellId() = default;
  bool operator==(const VRGCellId& other) { return this->_x == other._x && this->_y == other._y; }
  bool operator!=(const VRGCellId& other) { return !((*this) == other); }
  // getter
  irt_int get_x() const { return _x; }
  irt_int get_y() const { return _y; }
  // setter
  void set_x(const irt_int x) { _x = x; }
  void set_y(const irt_int y) { _y = y; }
  // function

 private:
  irt_int _x = -1;
  irt_int _y = -1;
};

struct CmpVRGCellId
{
  bool operator()(const VRGCellId& a, const VRGCellId& b) const
  {
    if (a.get_x() != b.get_x()) {
      return a.get_x() < b.get_x();
    } else {
      return a.get_y() < b.get_y();
    }
  }
};

}  // namespace irt
