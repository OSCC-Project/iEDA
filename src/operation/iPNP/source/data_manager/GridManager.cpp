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

/**
 * @brief Randomly initialize Templates when construct object.
 */
PDNGridTemplate::PDNGridTemplate() : _layers_occupied({1, 2, 6, 7, 8, 9})
{
  SingleLayerGrid layer1_grid = {StripeDirection::horizontal, 1.0, 1.0, 5.0};
  SingleLayerGrid layer2_grid = {StripeDirection::vertical, 1.0, 1.0, 5.0};
  SingleLayerGrid layer6_grid = {StripeDirection::horizontal, 1.0, 5.0, 5.0};
  SingleLayerGrid layer7_grid = {StripeDirection::vertical, 1.0, 5.0, 5.0};
  SingleLayerGrid layer8_grid = {StripeDirection::horizontal, 5.0, 10.0, 5.0};
  SingleLayerGrid layer9_grid = {StripeDirection::vertical, 5.0, 10.0, 5.0};

  _grid_per_layer.insert({1, layer1_grid});
  _grid_per_layer.insert({2, layer2_grid});
  _grid_per_layer.insert({6, layer6_grid});
  _grid_per_layer.insert({7, layer7_grid});
  _grid_per_layer.insert({8, layer8_grid});
  _grid_per_layer.insert({9, layer9_grid});
}

PDNGridRegion::PDNGridRegion() : _shape(GridRegionShape::rectangle)
{
}

PDNRectanGridRegion::PDNRectanGridRegion() : _height(15.0), _width(15.0)
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

  PDNGridTemplate pdn_grid_template = PDNGridTemplate();
  std::vector<PDNGridTemplate> template_libs(3, pdn_grid_template);
  _template_libs = template_libs;
}

}  // namespace ipnp