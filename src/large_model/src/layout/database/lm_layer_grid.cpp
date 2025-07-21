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
  //   for (int row_id = 0; row_id < gridInfoInst.node_row_num; ++row_id) {
  //     for (int col_id = 0; col_id < gridInfoInst.node_col_num; ++col_id) {
  //       if (_node_matrix[row_id][col_id] != nullptr) {
  //         delete _node_matrix[row_id][col_id];
  //         _node_matrix[row_id][col_id] = nullptr;
  //       }
  //     }
  //   }
    // 释放所有节点
  for (auto& pair : _node_map) {
      delete pair.second;
  }
}

std::pair<int, int> LmLayerGrid::buildNodeMatrix(int order)
{
  gridInfoInst.node_x_start = gridInfoInst.x_start % gridInfoInst.x_step;
  gridInfoInst.node_y_start = gridInfoInst.y_start % gridInfoInst.y_step;

  _row_num = (gridInfoInst.ury - gridInfoInst.node_y_start) / gridInfoInst.y_step;
  _col_num = (gridInfoInst.urx - gridInfoInst.node_x_start) / gridInfoInst.x_step;

  gridInfoInst.node_row_num = _row_num;
  gridInfoInst.node_col_num = _col_num;

  return std::make_pair(_row_num, _col_num);
}

LmNode* LmLayerGrid::get_node(int row_id, int col_id, bool b_create)
{  
  std::pair<int, int> key = {row_id, col_id};
  
  // 尝试查找节点
  auto it = _node_map.find(key);
  if (it != _node_map.end()) {
      return it->second;
  }
  
  // 如果不存在且不需要创建，返回nullptr
  if (!b_create) {
      return nullptr;
  }
  
  // 创建新节点
  LmNode* new_node = new LmNode();
  
  // 使用原子操作尝试插入
  // insert_or_assign 不适用于 TBB，使用 insert 代替
  auto result = _node_map.insert({key, new_node});
  
  if (!result.second) {
      // 插入失败，说明其他线程已经创建了节点
      delete new_node;
      return result.first->second;
  }
  
  // 插入成功，返回新节点
  return new_node;
}

LmNode* LmLayerGrid::findNode(int x, int y)
{
  auto [row_id, col_id] = gridInfoInst.findNodeID(x, y);

  return get_node(row_id, col_id);
}

std::vector<LmNode*> LmLayerGrid::get_all_nodes() 
{
  std::vector<LmNode*> nodes;
  nodes.reserve(_node_map.size());
  
  for (const auto& pair : _node_map) {
      nodes.push_back(pair.second);
  }
  
  return nodes;
}


}  // namespace ilm
