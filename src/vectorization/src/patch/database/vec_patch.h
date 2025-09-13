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
 * @project		vectorization
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

#include "vec_net.h"
#include "vec_node.h"

namespace ivec {

class VecPatchLayer
{
 public:
  VecPatchLayer() {}
  ~VecPatchLayer() {}

  // getter
  std::map<int, VecNet>& get_sub_nets() { return _sub_nets; }

  // setter

  // operator
  void addSubnet(int net_id, int64_t wire_id, VecNode* node1, VecNode* node2);
  void addSubnet(VecNet* sub_net, int64_t wire_id, VecNode* node1, VecNode* node2);
  VecNet* findNet(int net_id);

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
  std::map<int, VecNet> _sub_nets;  /// int : net id
};

class VecPatch
{
 public:
  VecPatch() {}
  ~VecPatch() {}

  // getter
  std::map<int, VecPatchLayer>& get_layer_map() { return _layer_map; }

  // setter

  // operator
  VecPatchLayer* findLayer(int layer_id);

 public:
  int patch_id = -1;
  int patch_id_row = -1;
  int patch_id_col = -1;

  int rowIdMin = -1;
  int rowIdMax = -1;
  int colIdMin = -1;
  int colIdMax = -1;

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
  std::map<int, VecPatchLayer> _layer_map;  /// int : layer id
};

}  // namespace ivec
