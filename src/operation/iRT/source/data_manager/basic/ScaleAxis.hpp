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

#include "ScaleGrid.hpp"

namespace irt {

class ScaleAxis
{
 public:
  ScaleAxis() = default;
  ~ScaleAxis() = default;
  // getter
  std::vector<ScaleGrid>& get_x_grid_list() { return _x_grid_list; }
  std::vector<ScaleGrid>& get_y_grid_list() { return _y_grid_list; }
  // setter
  void set_x_grid_list(const std::vector<ScaleGrid>& x_grid_list) { _x_grid_list = x_grid_list; }
  void set_y_grid_list(const std::vector<ScaleGrid>& y_grid_list) { _y_grid_list = y_grid_list; }
  // function
 private:
  std::vector<ScaleGrid> _x_grid_list;
  std::vector<ScaleGrid> _y_grid_list;
};
}  // namespace irt
