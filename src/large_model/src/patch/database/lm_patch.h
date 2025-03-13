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
/**
 * @project		large model
 * @date		06/11/2024
 * @version		0.1
 * @description
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <map>
#include <string>
#include <vector>

#include "lm_net.h"
#include "lm_node.h"

namespace ilm {

class LmPatchLayer
{
 public:
  LmPatchLayer() {}
  ~LmPatchLayer() {}

  // getter
  std::map<int, LmNet>& get_sub_nets() { return _sub_nets; }

  // setter

  // operator
  void addSubnet(int net_id, int64_t wire_id, LmNode* node1, LmNode* node2);
  void addSubnet(LmNet* sub_net, int64_t wire_id, LmNode* node1, LmNode* node2);
  LmNet* findNet(int net_id);

 public:
  int layer_id = -1;
  int rowIdMin = -1;
  int rowIdMax = -1;
  int colIdMin = -1;
  int colIdMax = -1;
  int wire_width = 0;
  int wire_len = 0;
  double wire_density = 0.0;
  double congestion = 0.0;

 private:
  std::map<int, LmNet> _sub_nets;  /// int : net id
};

class LmPatch
{
 public:
  LmPatch() {}
  ~LmPatch() {}

  // getter
  std::map<int, LmPatchLayer>& get_layer_map() { return _layer_map; }

  // setter

  // operator
  LmPatchLayer* findLayer(int layer_id);

 public:
  int patch_id = -1;
  int patch_id_row = -1;
  int patch_id_col = -1;

  int rowIdMin = -1;
  int rowIdMax = -1;
  int colIdMin = -1;
  int colIdMax = -1;

  int area = 0;
  double cell_density = -1;
  int pin_density = -1;
  double net_density = -1;
  int macro_margin = 0;
  double RUDY_congestion = -1;
  double EGR_congestion = -1;

  // timing, power, IR Drop map
  double timing_map = 0.0;
  double power_map = 0.0;
  double ir_drop_map = 0.0;

 private:
  std::map<int, LmPatchLayer> _layer_map;  /// int : layer id
};

}  // namespace ilm
