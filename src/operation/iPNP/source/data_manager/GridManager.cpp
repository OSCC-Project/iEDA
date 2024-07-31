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

SingleLayerGrid::SingleLayerGrid(StripeDirection direction, PowerType first_stripe_power_type, double width, double pg_offset, double space,
                                 double offset)
    : _direction(direction),
      _first_stripe_power_type(first_stripe_power_type),
      _width(width),
      _pg_offset(pg_offset),
      _space(space),
      _offset(offset)
{
}

/**
 * @brief Randomly initialize Templates when construct object.
 */
PDNGridTemplate::PDNGridTemplate() : _layers_occupied({1, 2, 6, 7, 8, 9})
{
  for (int layer : _layers_occupied) {
    SingleLayerGrid layer_grid = SingleLayerGrid();
    _grid_per_layer.insert({layer, layer_grid});
  }
}

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
GridManager::GridManager() : _ho_region_num(9), _ver_region_num(9), _chip_width(100.0), _chip_height(100.0)
{
  PDNRectanGridRegion pdn_rectan_grid_region = PDNRectanGridRegion();
  std::vector<std::vector<PDNRectanGridRegion>> grid_data(9, std::vector<PDNRectanGridRegion>(9, pdn_rectan_grid_region));
  _grid_data = grid_data;

  std::vector<std::vector<int>> template_data(9, std::vector<int>(9, 1));
  _template_data = template_data;

  PDNGridTemplate pdn_grid_template;
  std::vector<PDNGridTemplate> template_libs(3, pdn_grid_template);
  _template_libs = template_libs;
}

}  // namespace ipnp