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
#include <string>

#include "lm_layout.h"
#include "lm_net.h"
#include "lm_patch.h"

namespace ilm {

class LmLayoutDataManager
{
 public:
  LmLayoutDataManager() {}
  ~LmLayoutDataManager() = default;

  LmLayout& get_layout() { return _layout; }

  bool buildLayoutData(const std::string path);
  bool buildGraphData(const std::string path);

 private:
  LmLayout _layout;

  void init();
  void buildPatchs();

  std::map<int, LmNet> buildNetWires(bool b_graph);
  int buildCutLayer(int layer_id, LmPatchLayer& patch_layer, std::map<int, LmNet>& net_map);
  int buildRoutingLayer(int layer_id, LmPatchLayer& patch_layer, std::map<int, LmNet>& net_map);

  void buildSteinerWire(LmPatchLayer& patch_layer, std::map<int, LmNet>& net_map, std::vector<std::vector<bool>>& visited_matrix);
  void buildRegulerWire();

  void add_net_wire(std::map<int, LmNet>& net_map, int net_id, LmNetWire wire);
  int searchEndNode(LmNode& node_connected, LmLayerGrid& grid, std::map<int, LmNet>& net_map,
                    std::vector<std::vector<bool>>& visited_matrix);
  int search_node_in_direction(LmNode& node_connected, LmNodeDirection direction, LmLayerGrid& grid, std::map<int, LmNet>& net_map,
                               std::vector<std::vector<bool>>& visited_matrix);
  LmNode* travel_grid(LmNode* node_start, LmNodeDirection direction, LmLayerGrid& grid, std::vector<std::vector<bool>>& visited_matrix);
};

}  // namespace ilm