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

LmLayerGrid::LmLayerGrid() : _map_mutex(std::make_unique<std::shared_mutex>()) 
{
}

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
  // 检查坐标是否合法
  if (row_id < 0 || row_id >= _row_num || col_id < 0 || col_id >= _col_num) {
      return nullptr;
  }
  
  // 使用坐标对作为键
  std::pair<int, int> key = {row_id, col_id};
  
  if (b_create) {
      // 写操作需要独占锁
      std::unique_lock<std::shared_mutex> lock(*_map_mutex); 
      
      // 查找键
      auto it = _node_map.find(key);
      if (it == _node_map.end()) {
          // 键不存在，创建新节点并插入
          LmNode* new_node = new LmNode();
          _node_map[key] = new_node;
          return new_node;
      }
      
      return it->second;
  } else {
      // 读操作只需要共享锁
      std::shared_lock<std::shared_mutex> lock(*_map_mutex); 
      
      // 查找键
      auto it = _node_map.find(key);
      if (it != _node_map.end()) {
          return it->second;
      }
      
      return nullptr;
  }
}

LmNode* LmLayerGrid::findNode(int x, int y)
{
  auto [row_id, col_id] = gridInfoInst.findNodeID(x, y);

  return get_node(row_id, col_id);
}

std::vector<LmNode*> LmLayerGrid::get_all_nodes() 
{
    std::vector<LmNode*> nodes;
    
    // 使用共享锁保护读取操作
    std::shared_lock<std::shared_mutex> lock(*_map_mutex); 
    
    nodes.reserve(_node_map.size()); 
    for (const auto& [key, node] : _node_map) {
        nodes.push_back(node);
    }
    
    return nodes;
}


}  // namespace ilm
