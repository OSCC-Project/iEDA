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

#include "PlanarCoord.hpp"

namespace irt {

class EXTPlanarCoord
{
 public:
  EXTPlanarCoord() = default;
  ~EXTPlanarCoord() = default;
  // getter
  PlanarCoord& get_grid_coord() { return _grid; }
  int32_t get_grid_x() const { return _grid.get_x(); }
  int32_t get_grid_y() const { return _grid.get_y(); }
  PlanarCoord& get_real_coord() { return _real; }
  int32_t get_real_x() const { return _real.get_x(); }
  int32_t get_real_y() const { return _real.get_y(); }
  // const getter
  const PlanarCoord& get_grid_coord() const { return _grid; }
  const PlanarCoord& get_real_coord() const { return _real; }
  // setter
  void set_grid_coord(const PlanarCoord& grid) { _grid = grid; }
  void set_grid_coord(const int32_t x, const int32_t y) { _grid.set_coord(x, y); }
  void set_real_coord(const PlanarCoord& real) { _real = real; }
  void set_real_coord(const int32_t x, const int32_t y) { _real.set_coord(x, y); }
  // function

 private:
  PlanarCoord _grid;
  PlanarCoord _real;
};

}  // namespace irt
