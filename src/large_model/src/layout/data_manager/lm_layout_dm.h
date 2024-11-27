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
  std::map<int, LmNet>& get_graph(std::string path = "") { return _layout.get_graph().get_net_map(); }

  bool buildLayoutData(const std::string path);
  bool buildGraphData(const std::string path);

 private:
  LmLayout _layout;

  void init();
  void buildPatchs();

  std::map<int, LmNet> buildNetWires(bool b_graph);
  int buildCutLayer(int layer_id, LmPatchLayer& patch_layer);
  int buildRoutingLayer(int layer_id, LmPatchLayer& patch_layer);

  void add_net_wire(int net_id, LmNetWire wire);
  int searchEndNode(LmNode& node_connected, LmLayerGrid& grid);
  int search_node_in_direction(LmNode& node_connected, LmNodeDirection direction, LmLayerGrid& grid);
  LmNode* travel_grid(LmNode* node_start, LmNodeDirection direction, LmLayerGrid& grid);
  LmNodeDirection get_corner_orthogonal_direction(LmNode* node, LmNodeDirection direction);
  LmNodeDirection get_opposite_direction(LmNodeDirection direction);
};

}  // namespace ilm