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

#include "DRCHeader.hpp"

namespace idrc {

class PlanarCoord
{
 public:
  PlanarCoord() = default;
  PlanarCoord(const int32_t x, const int32_t y)
  {
    _x = x;
    _y = y;
  }
  ~PlanarCoord() = default;
  bool operator==(const PlanarCoord& other) const { return (_x == other._x && _y == other._y); }
  bool operator!=(const PlanarCoord& other) const { return !((*this) == other); }
  // getter
  int32_t get_x() const { return _x; }
  int32_t get_y() const { return _y; }
  // setter
  void set_x(const int32_t x) { _x = x; }
  void set_y(const int32_t y) { _y = y; }
  void set_coord(const int32_t x, const int32_t y)
  {
    _x = x;
    _y = y;
  }
  void set_coord(const PlanarCoord& coord) { set_coord(coord.get_x(), coord.get_y()); }
  // function
 private:
  int32_t _x = -1;
  int32_t _y = -1;
};

struct CmpPlanarCoordByXASC
{
  bool operator()(const PlanarCoord& a, const PlanarCoord& b) const { return a.get_x() != b.get_x() ? a.get_x() < b.get_x() : a.get_y() < b.get_y(); }
};

struct CmpPlanarCoordByYASC
{
  bool operator()(const PlanarCoord& a, const PlanarCoord& b) const { return a.get_y() != b.get_y() ? a.get_y() < b.get_y() : a.get_x() < b.get_x(); }
};

}  // namespace idrc
