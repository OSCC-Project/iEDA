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

#include "LayerRect.hpp"
#include "RTU.hpp"
#include "RTUtil.hpp"

namespace irt {

class Guide : public LayerRect
{
 public:
  Guide() = default;
  Guide(const PlanarRect& shape, const irt_int layer_idx) : LayerRect(shape, layer_idx) {}
  Guide(const PlanarRect& shape, const irt_int layer_idx, const PlanarCoord& grid_coord)
      : LayerRect(shape, layer_idx), _grid_coord(grid_coord)
  {
  }
  ~Guide() = default;
  // getter
  PlanarCoord& get_grid_coord() { return _grid_coord; }
  // setter
  void set_grid_coord(const PlanarCoord& grid_coord) { _grid_coord = grid_coord; }
  void set_grid_coord(const irt_int x, const irt_int y)
  {
    _grid_coord.set_x(x);
    _grid_coord.set_y(y);
  }
  // function

 private:
  PlanarCoord _grid_coord;
};

}  // namespace irt
