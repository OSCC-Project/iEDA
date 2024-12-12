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
#include "lm_layer_grid.h"

#include "lm_grid_info.h"

namespace ilm {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
LmLayerGrid::~LmLayerGrid()
{
  for (int row_id = 0; row_id < gridInfoInst.node_row_num; ++row_id) {
    for (int col_id = 0; col_id < gridInfoInst.node_col_num; ++col_id) {
      if (_node_matrix[row_id][col_id] != nullptr) {
        delete _node_matrix[row_id][col_id];
        _node_matrix[row_id][col_id] = nullptr;
      }
    }
  }
}

std::pair<int, int> LmLayerGrid::buildNodeMatrix(int order)
{
  gridInfoInst.node_x_start = gridInfoInst.x_start % gridInfoInst.x_step;
  gridInfoInst.node_y_start = gridInfoInst.y_start % gridInfoInst.y_step;

  int row_num = (gridInfoInst.ury - gridInfoInst.node_y_start) / gridInfoInst.y_step;
  int col_num = (gridInfoInst.urx - gridInfoInst.node_x_start) / gridInfoInst.x_step;

  _node_matrix = std::vector<std::vector<LmNode*>>(row_num, std::vector<LmNode*>(col_num, nullptr));

  gridInfoInst.node_row_num = _node_matrix.size();
  gridInfoInst.node_col_num = _node_matrix[0].size();

  return std::make_pair(gridInfoInst.node_row_num, gridInfoInst.node_col_num);
}

LmNode* LmLayerGrid::get_node(int row_id, int col_id, bool b_create)
{
  if (_node_matrix[row_id][col_id] == nullptr && b_create) {
    _node_matrix[row_id][col_id] = new LmNode();
  }
  return _node_matrix[row_id][col_id];
}

LmNode* LmLayerGrid::findNode(int x, int y)
{
  auto [row_id, col_id] = gridInfoInst.findNodeID(x, y);

  return get_node(row_id, col_id);
}

}  // namespace ilm
