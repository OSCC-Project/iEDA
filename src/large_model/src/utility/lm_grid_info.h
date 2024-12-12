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

namespace ilm {

#define gridInfoInst LmGridInfo::getInst()

class LmGridInfo
{
 public:
  static LmGridInfo& getInst()
  {
    if (_info_inst == nullptr) {
      _info_inst = new LmGridInfo();
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

  int calculate_x(int col);
  int calculate_y(int row);

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
  static LmGridInfo* _info_inst;

  LmGridInfo() {}
  ~LmGridInfo() {}
};

}  // namespace ilm
