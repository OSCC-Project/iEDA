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
#include <map>
#include <string>
#include <vector>

#include "lm_node.h"

namespace ilm {

struct LmLayerGridInfo
{
  int layer_order;
  int llx;
  int lly;
  int urx;
  int ury;
  int x_start;
  int node_x_start;
  int x_step;
  int y_start;
  int node_y_start;
  int y_step;
  int node_row_num;  /// node number on rows
  int node_col_num;  /// node number on cols
};

enum class SideType
{
  kNone,
  kLower,
  kHigher,
  kMax
};

class LmLayerGrid
{
 public:
  LmLayerGrid() {}
  ~LmLayerGrid() = default;

  // getter
  LmLayerGridInfo& get_info() { return _info; }
  std::vector<std::vector<LmNode>>& get_node_matrix() { return _node_matrix; }
  LmNode& get_node(int row_id, int col_id);
  LmNode& findNode(int x, int y);
  std::pair<int, int> findNodeID(int x, int y);
  int findNodeID(int value, bool b_row_id, SideType side_type = SideType::kNone);
  std::pair<int, int> getNodeIDRange(int coord1, int coord2, bool b_row_id);
  std::tuple<int, int, int, int> getNodeIdRange(int x1, int x2, int y1, int y2);

  // setter

  // operator
  void buildNodeMatrix();

 private:
  LmLayerGridInfo _info;

  std::vector<std::vector<LmNode>> _node_matrix;
};

}  // namespace ilm
