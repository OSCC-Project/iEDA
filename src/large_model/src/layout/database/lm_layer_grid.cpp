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

namespace ilm {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::pair<int, int> LmLayerGrid::buildNodeMatrix(int order)
{
  _info.node_x_start = _info.x_start % _info.x_step;
  _info.node_y_start = _info.y_start % _info.y_step;

  int row_num = (_info.ury - _info.node_y_start) / _info.y_step;
  int col_num = (_info.urx - _info.node_x_start) / _info.x_step;

  _node_matrix = std::vector<std::vector<LmNode*>>(row_num, std::vector<LmNode*>(col_num, nullptr));

  // #pragma omp parallel for schedule(dynamic)
  //   for (int y = _info.node_y_start; y < _info.ury; y = y + _info.y_step) {
  //     std::vector<LmNode> row_nodes;
  //     int row = (y - _info.node_y_start) / _info.y_step;
  //     auto& lm_node_row = _node_matrix[row];
  //     int col = 0;
  //     for (int x = _info.node_x_start; x < _info.urx; x = x + _info.x_step, ++col) {
  //       LmNode& lm_node = lm_node_row[col];

  //       lm_node.set_col_id(col);
  //       lm_node.set_row_id(row);
  //       lm_node.set_x(x);
  //       lm_node.set_y(y);
  //       lm_node.set_layer_id(order);

  //       //   row_nodes.push_back(lm_node);
  //     }

  //     // _node_matrix.push_back(row_nodes);
  //   }

  _info.node_row_num = _node_matrix.size();
  _info.node_col_num = _node_matrix[0].size();

  return std::make_pair(_info.node_row_num, _info.node_col_num);
}

int LmLayerGrid::calculate_x(int col)
{
  return _info.node_x_start + _info.x_step * col;
}

int LmLayerGrid::calculate_y(int row)
{
  return _info.node_y_start + _info.y_step * row;
}

LmNode* LmLayerGrid::get_node(int row_id, int col_id, bool b_create)
{
  if (_node_matrix[row_id][col_id] == nullptr && b_create) {
    _node_matrix[row_id][col_id] = new LmNode();
  }
  return _node_matrix[row_id][col_id];
}

std::pair<int, int> LmLayerGrid::get_node_coodinate(int row_id, int col_id)
{
  int x = _info.node_y_start + row_id * _info.y_step;
  int y = _info.node_x_start + row_id * _info.x_step;

  return std::make_pair(x, y);
}

LmNode* LmLayerGrid::findNode(int x, int y)
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

    // if (side_type == SideType::kHigher) {
    //   node_id = remain == 0 ? index : index + 1;
    // } else {
    //   /// find nearest node
    //   //   node_id = remain > step / 2 ? index + 1 : index;
    //   node_id = index;
    // }

    node_id = index >= node_num ? node_num - 1 : index;
  }

  return node_id;
}

std::pair<int, int> LmLayerGrid::get_node_id_range(int coord1, int coord2, bool b_row_id)
{
  auto index_1 = findNodeID(coord1, b_row_id, SideType::kLower);
  auto index_2 = findNodeID(coord2, b_row_id, SideType::kHigher);

  return std::make_pair(index_1, index_2);
}

std::tuple<int, int, int, int> LmLayerGrid::get_node_id_range(int x1, int x2, int y1, int y2)
{
  /// 0 : no direction
  /// 1 : horizotal
  /// 2 : vertival
  int direction = (x2 - x1) == (y2 - y1) ? 0 : ((x2 - x1) > (y2 - y1) ? 1 : 2);

  /// row id
  int row_id_1 = findNodeID(y1, true, SideType::kLower);
  int row_id_2 = findNodeID(y2, true, SideType::kHigher);
  /// col id
  int col_id_1 = findNodeID(x1, false, SideType::kLower);
  int col_id_2 = findNodeID(x2, false, SideType::kHigher);

  //   switch (direction) {
  //     case 1:
  //       /// 1 : horizotal
  //       if (std::abs(y2 - y1) <= _info.y_step) {
  //         row_id_1 = findNodeID((y1 + y2) / 2, true, SideType::kLower);
  //         row_id_2 = row_id_1;
  //       } else {
  //         row_id_1 = findNodeID(y1, true, SideType::kLower);
  //         row_id_2 = findNodeID(y2, true, SideType::kHigher);
  //       }
  //       if (std::abs(x2 - x1) < _info.x_step) {
  //         col_id_1 = findNodeID((x1 + x2) / 2, true, SideType::kLower);
  //         col_id_2 = col_id_1;
  //       } else {
  //         col_id_1 = findNodeID(x1, false, SideType::kLower);
  //         col_id_2 = findNodeID(x2, false, SideType::kHigher);
  //       }

  //       break;
  //     case 2:
  //       /// 2 : vertival
  //       if (std::abs(y2 - y1) < _info.y_step) {
  //         row_id_1 = findNodeID((y1 + y2) / 2, true, SideType::kLower);
  //         row_id_2 = row_id_1;
  //       } else {
  //         row_id_1 = findNodeID(y1, true, SideType::kLower);
  //         row_id_2 = findNodeID(y2, true, SideType::kHigher);
  //       }
  //       if (std::abs(x2 - x1) <= _info.x_step) {
  //         col_id_1 = findNodeID((x1 + x2) / 2, true, SideType::kLower);
  //         col_id_2 = col_id_1;
  //       } else {
  //         col_id_1 = findNodeID(x1, false, SideType::kLower);
  //         col_id_2 = findNodeID(x2, false, SideType::kHigher);
  //       }

  //       break;
  //     default:
  //       if (std::abs(y2 - y1) <= _info.y_step) {
  //         row_id_1 = findNodeID((y1 + y2) / 2, true, SideType::kLower);
  //         row_id_2 = row_id_1;
  //       } else {
  //         row_id_1 = findNodeID(y1, true, SideType::kLower);
  //         row_id_2 = findNodeID(y2, true, SideType::kHigher);
  //       }
  //       if (std::abs(x2 - x1) <= _info.x_step) {
  //         col_id_1 = findNodeID((x1 + x2) / 2, true, SideType::kLower);
  //         col_id_2 = col_id_1;
  //       } else {
  //         col_id_1 = findNodeID(x1, false, SideType::kLower);
  //         col_id_2 = findNodeID(x2, false, SideType::kHigher);
  //       }

  //       break;
  //   }

  return std::tuple<int, int, int, int>(std::min(row_id_1, row_id_2), std::max(row_id_1, row_id_2), std::min(col_id_1, col_id_2),
                                        std::max(col_id_1, col_id_2));
}

}  // namespace ilm
