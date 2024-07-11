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

#include "GridManager.hh"

namespace ipnp {

PDNGridTemplate::PDNGridTemplate() : _layers_occupied({1, 2, 6, 7, 8, 9})
{
  GridPerLayer grid_per_layer1 = {"horizontal", 1.0, 1.0, 5.0};
  GridPerLayer grid_per_layer2 = {"vertical", 1.0, 1.0, 5.0};
  GridPerLayer grid_per_layer6 = {"vertical", 1.0, 5.0, 5.0};
  GridPerLayer grid_per_layer7 = {"vertical", 1.0, 5.0, 5.0};
  GridPerLayer grid_per_layer8 = {"vertical", 5.0, 10.0, 5.0};
  GridPerLayer grid_per_layer9 = {"vertical", 5.0, 10.0, 5.0};
  _layer_to_grid.insert({1, grid_per_layer1});
  _layer_to_grid.insert({2, grid_per_layer2});
  _layer_to_grid.insert({6, grid_per_layer6});
  _layer_to_grid.insert({7, grid_per_layer7});
  _layer_to_grid.insert({8, grid_per_layer8});
  _layer_to_grid.insert({9, grid_per_layer9});
}

PDNGridRegion::PDNGridRegion() : _type("rectangle")
{
}

PDNRectanGridRegion::PDNRectanGridRegion() : _height(15.0), _width(15.0)
{
}

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