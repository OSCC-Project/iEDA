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
/**
 * @file GridManager.cpp
 * @author Xinhao li
 * @brief
 * @version 0.1
 * @date 2024-07-15
 */

#include "GridManager.hh"

namespace ipnp {

PDNGridRegion::PDNGridRegion() : _shape(GridRegionShape::kRectangle)
{
}

PDNRectanGridRegion::PDNRectanGridRegion() : _x_left_bottom(10.0), _y_left_bottom(10.0), _x_right_top(40.0), _y_right_top(40.0)
{
}

/**
 * @brief Randomly initialize the number of region.
 * @attention chip width and height should be passed in from iDB.
 */
GridManager::GridManager()
  : _power_layers({ 5,7,8,9 }),
  _layer_count(_power_layers.size()),
  _ho_region_num(5),
  _ver_region_num(5),
  _core_width(100.0),
  _core_height(100.0)
{
  _template_libs.gen_template_libs();
  initialize_grid_data();
}

void GridManager::initialize_grid_data()
{
  // Initialize 3D grid data
  _grid_data.resize(_layer_count);
  for (int i = 0; i < _layer_count; ++i) {
    _grid_data[i].resize(_ho_region_num);
    for (int row = 0; row < _ho_region_num; ++row) {
      _grid_data[i][row].resize(_ver_region_num);
      for (int col = 0; col < _ver_region_num; ++col) {
        PDNRectanGridRegion& region = _grid_data[i][row][col];
        // Set default coordinates for each region
        double x_left = col * (_core_width / _ver_region_num);
        double y_bottom = row * (_core_height / _ho_region_num);
        region.set_x_left_bottom(x_left);
        region.set_y_left_bottom(y_bottom);
        region.set_x_right_top(x_left + _core_width / _ver_region_num);
        region.set_y_right_top(y_bottom + _core_height / _ho_region_num);
      }
    }
  }

  // Initialize 3D template data
  _template_data.resize(_layer_count);
  for (int i = 0; i < _layer_count; ++i) {
    _template_data[i].resize(_ho_region_num);
    for (int row = 0; row < _ho_region_num; ++row) {
      _template_data[i][row].resize(_ver_region_num, SingleTemplate());
    }
  }
}

}  // namespace ipnp