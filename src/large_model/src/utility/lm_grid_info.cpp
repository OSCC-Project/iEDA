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
#include "lm_grid_info.h"

namespace ilm {

LmGridInfo* LmGridInfo::_info_inst = nullptr;
LmPatchInfo* LmPatchInfo::_info_inst = nullptr;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int LmGridInfo::calculate_x(int col)
{
  int x = node_x_start + col * x_step;
  return x > urx ? urx : x;
}

int LmGridInfo::calculate_y(int row)
{
  int y = node_y_start + row * y_step;
  return y > ury ? ury : y;
}

std::pair<int, int> LmGridInfo::get_node_coodinate(int row_id, int col_id)
{
  int x = node_y_start + row_id * y_step;
  int y = node_x_start + row_id * x_step;

  return std::make_pair(x, y);
}

std::pair<int, int> LmGridInfo::findNodeID(int x, int y)
{
  int row_id = findNodeID(y, true);
  int col_id = findNodeID(x, false);

  return std::make_pair(row_id, col_id);
}

int LmGridInfo::findNodeID(int value, bool b_row_id)
{
  int node_id = -1;

  auto node_start = b_row_id ? node_y_start : node_x_start;
  auto step = b_row_id ? y_step : x_step;
  auto node_num = b_row_id ? node_row_num : node_col_num;

  if (value <= node_start) {
    node_id = 0;
  } else {
    auto remain = (value - node_start) % step;
    auto index = (value - node_start) / step;

    node_id = index >= node_num ? node_num - 1 : index;
  }

  return node_id;
}

std::pair<int, int> LmGridInfo::get_node_id_range(int coord1, int coord2, bool b_row_id)
{
  auto index_1 = findNodeID(coord1, b_row_id);
  auto index_2 = findNodeID(coord2, b_row_id);

  return std::make_pair(index_1, index_2);
}

std::tuple<int, int, int, int> LmGridInfo::get_node_id_range(int x1, int x2, int y1, int y2)
{
  /// 0 : no direction
  /// 1 : horizotal
  /// 2 : vertival
  int direction = (x2 - x1) == (y2 - y1) ? 0 : ((x2 - x1) > (y2 - y1) ? 1 : 2);

  /// row id
  int row_id_1 = findNodeID(y1, true);
  int row_id_2 = findNodeID(y2, true);
  /// col id
  int col_id_1 = findNodeID(x1, false);
  int col_id_2 = findNodeID(x2, false);

  return std::tuple<int, int, int, int>(std::min(row_id_1, row_id_2), std::max(row_id_1, row_id_2), std::min(col_id_1, col_id_2),
                                        std::max(col_id_1, col_id_2));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
LmPatchInfo::LmPatchInfo()
{
  patch_row_start = gridInfoInst.y_start / gridInfoInst.y_step;
  patch_col_start = gridInfoInst.x_start / gridInfoInst.x_step;

  set_patch_num();
}

void LmPatchInfo::set_patch_num()
{
  {
    int bottom_num = patch_row_start % patch_row_step == 0 ? patch_row_start / patch_row_step : patch_row_start / patch_row_step + 1;
    int top_num = (gridInfoInst.node_row_num - patch_row_start) % patch_row_step == 0
                      ? (gridInfoInst.node_row_num - patch_row_start) / patch_row_step
                      : (gridInfoInst.node_row_num - patch_row_start) / patch_row_step + 1;
    patch_num_vertical = bottom_num + top_num;
  }

  {
    int left_num = patch_col_start % patch_col_step == 0 ? patch_col_start / patch_col_step : patch_col_start / patch_col_step + 1;
    int right_num = (gridInfoInst.node_col_num - patch_col_start) % patch_col_step == 0
                        ? (gridInfoInst.node_col_num - patch_col_start) / patch_col_step
                        : (gridInfoInst.node_col_num - patch_row_start) / patch_col_step + 1;
    patch_num_horizontal = left_num + right_num;
  }
}

}  // namespace ilm
