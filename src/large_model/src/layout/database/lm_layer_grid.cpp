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
 * @file		patch.h
 * @date		06/11/2024
 * @version		0.1
 * @description
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "lm_layer_grid.h"

namespace ilm {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void LmLayerGrid::buildNodeMatrix()
{
  _info.node_x_start = _info.x_start % _info.x_step;
  _info.node_y_start = _info.y_start % _info.y_step;

  int row = 0;
  for (int y = _info.node_y_start; y < _info.ury; y = y + _info.y_step, ++row) {
    std::vector<LmNode> row_nodes;
    int col = 0;
    for (int x = _info.node_x_start; x < _info.urx; x = x + _info.x_step, ++col) {
      LmNode lm_node;

      lm_node.set_col_id(col);
      lm_node.set_row_id(row);
      lm_node.set_x(x);
      lm_node.set_y(y);

      row_nodes.push_back(lm_node);
    }

    _node_matrix.push_back(row_nodes);
  }

  _info.node_row_num = _node_matrix.size();
  _info.node_col_num = _node_matrix[0].size();
}

LmNode& LmLayerGrid::get_node(int row_id, int col_id)
{
  auto row_patchs = _node_matrix.at(row_id);
  return row_patchs.at(col_id);
}

LmNode& LmLayerGrid::findNode(int x, int y)
{
  auto [row_id, col_id] = findNodeID(x, y);

  return get_node(row_id, col_id);
}

std::pair<int, int> LmLayerGrid::findNodeID(int x, int y)
{
  int row_id = findNodeID(y, true);
  int col_id = findNodeID(x, false);

  return std::make_pair(row_id, col_id);
}

int LmLayerGrid::findNodeID(int value, bool b_row_id, SideType side_type)
{
  int node_id = -1;

  auto node_start = b_row_id ? _info.node_y_start : _info.node_x_start;
  auto step = b_row_id ? _info.y_step : _info.x_step;
  auto node_num = b_row_id ? _info.node_row_num : _info.node_col_num;

  if (value <= node_start) {
    node_id = 0;
  } else {
    auto remain = (value - node_start) % step;
    auto index = (value - node_start) / step;

    if (side_type == SideType::kLower) {
      node_id = index + 1;
    } else if (side_type == SideType::kHigher) {
      node_id = index;
    } else {
      /// find nearest node
      node_id = remain > step / 2 ? index + 1 : index;
    }

    node_id = node_id >= node_num ? node_num - 1 : node_id;
  }

  return node_id;
}

std::pair<int, int> LmLayerGrid::getNodeIDRange(int coord1, int coord2, bool b_row_id)
{
  auto index_1 = findNodeID(coord1, b_row_id, SideType::kLower);
  auto index_2 = findNodeID(coord2, b_row_id, SideType::kHigher);

  return std::make_pair(index_1, index_2);
}

std::tuple<int, int, int, int> LmLayerGrid::getNodeIdRange(int x1, int x2, int y1, int y2)
{
  /// row id
  auto row_id_1 = findNodeID(y1, true, SideType::kLower);
  auto row_id_2 = findNodeID(y2, true, SideType::kHigher);
  /// col id
  auto col_id_1 = findNodeID(x1, false, SideType::kLower);
  auto col_id_2 = findNodeID(x2, false, SideType::kHigher);

  return std::tuple<int, int, int, int>(row_id_1, row_id_2, col_id_1, col_id_2);
}

}  // namespace ilm
