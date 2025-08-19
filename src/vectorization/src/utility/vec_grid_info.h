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

namespace ivec {

#define gridInfoInst VecGridInfo::getInst()
#define patchInfoInst VecPatchInfo::getInst()

class VecGridInfo
{
 public:
  static VecGridInfo& getInst()
  {
    if (_info_inst == nullptr) {
      _info_inst = new VecGridInfo();
    }
    return *_info_inst;
  }

  static void destroyInst();

  // getter
  std::pair<int, int> get_node_id_range(int coord1, int coord2, bool b_row_id);
  std::tuple<int, int, int, int> get_node_id_range(int x1, int x2, int y1, int y2);
  std::pair<int, int> get_node_coodinate(int row_id, int col_id);
  bool is_out_of_range(int row_id, int col_id)
  {
    return (row_id >= 0 && row_id < node_row_num) && (col_id >= 0 && col_id < node_col_num) ? false : true;
  }

  // setter

  // operator
  std::pair<int, int> findNodeID(int x, int y);
  int findNodeID(int value, bool b_row_id);

  int64_t calculate_x(int64_t col);
  int64_t calculate_y(int64_t row);

 public:
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

 private:
  static VecGridInfo* _info_inst;

  VecGridInfo() {}
  ~VecGridInfo() {}
};

class VecPatchInfo
{
 public:
  static VecPatchInfo& getInst()
  {
    if (_info_inst == nullptr) {
      _info_inst = new VecPatchInfo();
    }
    return *_info_inst;
  }

  static VecPatchInfo& getInst(int row_step, int col_step)
  {
    if (_info_inst == nullptr) {
      _info_inst = new VecPatchInfo(row_step, col_step);
    } else {
      _info_inst->set_patch_size(row_step, col_step);
    }
    return *_info_inst;
  }

  static void destroyInst();

  /// getter
  int get_patch_id(int node_row, int node_col);
  int get_patch_row_id(int node_row);
  int get_patch_col_id(int node_col);
  std::pair<int, int> get_node_range(int index, bool b_horizontal);

  // setter
  void set_patch_size(int row_size, int col_size)
  {
    patch_row_step = row_size;
    patch_col_step = col_size;

    set_patch_num();
  }

  // operator

 public:
  int patch_row_start = 0;
  int patch_row_step = 9 * 2 * 10;  /// default 9T/180T
  int patch_col_start = 0;
  int patch_col_step = 9 * 2 * 10;  /// default 9T/180T
  int patch_num_vertical = 0;       /// indicate how many patchs in vertical direciton
  int patch_num_horizontal = 0;     /// indicate how many patchs in horizontal direciton

 private:
  static VecPatchInfo* _info_inst;

  VecPatchInfo();
  VecPatchInfo(int row_step, int col_step) : 
      patch_row_start(0), patch_row_step(row_step), 
      patch_col_start(0), patch_col_step(col_step),
      patch_num_vertical(0), patch_num_horizontal(0)
  {
    set_patch_num();
  }
  ~VecPatchInfo() {}

  void set_patch_num();
};

}  // namespace ivec
