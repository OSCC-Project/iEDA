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
#include "vec_grid_info.h"

namespace ivec {

VecGridInfo* VecGridInfo::_info_inst = nullptr;
VecPatchInfo* VecPatchInfo::_info_inst = nullptr;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int64_t VecGridInfo::calculate_x(int64_t col)
{
  int64_t x = node_x_start + col * x_step;
  return x > urx ? urx : x;
}

int64_t VecGridInfo::calculate_y(int64_t row)
{
  int64_t y = node_y_start + row * y_step;
  return y > ury ? ury : y;
}

std::pair<int, int> VecGridInfo::get_node_coodinate(int row_id, int col_id)
{
  int x = node_x_start + col_id * x_step;
  int y = node_y_start + row_id * y_step;

  return std::make_pair(x, y);
}

std::pair<int, int> VecGridInfo::findNodeID(int x, int y)
{
  int row_id = findNodeID(y, true);
  int col_id = findNodeID(x, false);

  return std::make_pair(row_id, col_id);
}

int VecGridInfo::findNodeID(int value, bool b_row_id)
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

std::pair<int, int> VecGridInfo::get_node_id_range(int coord1, int coord2, bool b_row_id)
{
  auto index_1 = findNodeID(coord1, b_row_id);
  auto index_2 = findNodeID(coord2, b_row_id);

  return std::make_pair(index_1, index_2);
}

std::tuple<int, int, int, int> VecGridInfo::get_node_id_range(int x1, int x2, int y1, int y2)
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
VecPatchInfo::VecPatchInfo()
{
  int core_row_start = gridInfoInst.y_start / gridInfoInst.y_step;
  int core_col_start = gridInfoInst.x_start / gridInfoInst.x_step;

  patch_row_start = core_row_start % patch_row_step;
  patch_col_start = core_col_start % patch_col_step;

  set_patch_num();
}

void VecPatchInfo::set_patch_num()
{
  {
    patch_num_vertical = patch_row_start == 0 ? 0 : 1;

    int row_num = (gridInfoInst.node_row_num - patch_row_start) % patch_row_step == 0
                      ? (gridInfoInst.node_row_num - patch_row_start) / patch_row_step
                      : (gridInfoInst.node_row_num - patch_row_start) / patch_row_step + 1;

    patch_num_vertical += row_num;
  }

  {
    patch_num_horizontal = patch_col_start == 0 ? 0 : 1;

    int right_num = (gridInfoInst.node_col_num - patch_col_start) % patch_col_step == 0
                        ? (gridInfoInst.node_col_num - patch_col_start) / patch_col_step
                        : (gridInfoInst.node_col_num - patch_col_start) / patch_col_step + 1;
    patch_num_horizontal += right_num;
  }
}

int VecPatchInfo::get_patch_id(int node_row, int node_col)
{
  return patch_num_horizontal * get_patch_row_id(node_row) + get_patch_col_id(node_col);
}

int VecPatchInfo::get_patch_row_id(int node_row)
{
  int patch_row_id = 0;
  if (node_row < patch_row_start) {
    patch_row_id = 0;
  } else {
    patch_row_id = patch_row_start == 0 ? 0 : 1;
    patch_row_id += ((node_row - patch_row_start) / patch_row_step);
  }

  return patch_row_id;
}

int VecPatchInfo::get_patch_col_id(int node_col)
{
  int patch_col_id = 0;
  {
    if (node_col < patch_col_start) {
      patch_col_id = 0;
    } else {
      patch_col_id = patch_col_start == 0 ? 0 : 1;
      patch_col_id += ((node_col - patch_col_start) / patch_col_step);
    }
  }

  return patch_col_id;
}

std::pair<int, int> VecPatchInfo::get_node_range(int index, bool b_horizontal)
{
  int min_id, max_id;
  if (b_horizontal) {
    if (patch_col_start == 0) {
      min_id = index * patch_col_step;
      max_id = (index + 1) * patch_col_step;
      max_id = max_id > gridInfoInst.node_col_num ? gridInfoInst.node_col_num : max_id;
    } else {
      if (index == 0) {
        min_id = 0;
        max_id = patch_col_start;
      } else if (index == patch_num_horizontal - 1) {
        min_id = patch_col_start + (index - 1) * patch_col_step;
        max_id = gridInfoInst.node_col_num;
      } else {
        min_id = patch_col_start + (index - 1) * patch_col_step;
        max_id = patch_col_start + index * patch_col_step;
      }
    }
  } else {
    if (patch_row_start == 0) {
      min_id = index * patch_row_step;
      max_id = (index + 1) * patch_row_step;
      max_id = max_id > gridInfoInst.node_row_num ? gridInfoInst.node_row_num : max_id;
    } else {
      if (index == 0) {
        min_id = 0;
        max_id = patch_row_start;
      } else if (index == patch_num_vertical - 1) {
        min_id = patch_row_start + (index - 1) * patch_row_step;
        max_id = gridInfoInst.node_row_num;
      } else {
        min_id = patch_row_start + (index - 1) * patch_row_step;
        max_id = patch_row_start + index * patch_row_step;
      }
    }
  }

  return std::make_pair(min_id, max_id);
}

}  // namespace ivec
