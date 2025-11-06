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

#include "RTHeader.hpp"

namespace irt {

class ERBoxId
{
 public:
  ERBoxId() = default;
  ERBoxId(const int32_t x, const int32_t y)
  {
    _x = x;
    _y = y;
  }
  ~ERBoxId() = default;
  bool operator==(const ERBoxId& other) { return this->_x == other._x && this->_y == other._y; }
  bool operator!=(const ERBoxId& other) { return !((*this) == other); }
  // getter
  int32_t get_x() const { return _x; }
  int32_t get_y() const { return _y; }
  // setter
  void set_x(const int32_t x) { _x = x; }
  void set_y(const int32_t y) { _y = y; }
  // function

 private:
  int32_t _x = -1;
  int32_t _y = -1;
};

struct CmpERBoxId
{
  bool operator()(const ERBoxId& a, const ERBoxId& b) const
  {
    if (a.get_x() != b.get_x()) {
      return a.get_x() < b.get_x();
    } else {
      return a.get_y() < b.get_y();
    }
  }
};

}  // namespace irt
