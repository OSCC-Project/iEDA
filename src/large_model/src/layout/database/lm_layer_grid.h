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

#include "lm_node.h"

namespace ilm {

class LmLayerGrid
{
 public:
  LmLayerGrid() {}
  ~LmLayerGrid();

  // getter
  std::vector<std::vector<LmNode*>>& get_node_matrix() { return _node_matrix; }
  LmNode* get_node(int row_id, int col_id, bool b_create = false);

  // setter

  // operator
  std::pair<int, int> buildNodeMatrix(int order);
  LmNode* findNode(int x, int y);

 public:
  int layer_order;

 private:
  std::vector<std::vector<LmNode*>> _node_matrix;
};

}  // namespace ilm
