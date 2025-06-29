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
 * @author Jianrong Su
 * @brief
 * @version 1.0
 * @date 2025-06-23
 */

#include "GridManager.hh"
#include "PNPConfig.hh"

namespace ipnp {

PDNGridRegion::PDNGridRegion() : _shape(GridRegionShape::kRectangle)
{
}

PDNRectanGridRegion::PDNRectanGridRegion() 
: _x_left_bottom(), 
_y_left_bottom(), 
_x_right_top(),
_y_right_top()
{
}

void GridManager::init_GridManager_data()
{
  _template_libs.gen_template_libs();
  initialize_grid_data(_die_width, _die_height);
}

void GridManager::init_GridManager_data(const PNPConfig* config)
{
  if (config) {
    _template_libs.gen_template_libs_from_config(config);
  } else {
    _template_libs.gen_template_libs();
  }
  
  initialize_grid_data(_die_width, _die_height);
}

void GridManager::initialize_grid_data(int32_t width, int32_t height)
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
        double x_left = col * (width / _ver_region_num);
        double y_bottom = row * (height / _ho_region_num);
        region.set_x_left_bottom(x_left);
        region.set_y_left_bottom(y_bottom);
        region.set_x_right_top(x_left + width / _ver_region_num);
        region.set_y_right_top(y_bottom + height / _ho_region_num);
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