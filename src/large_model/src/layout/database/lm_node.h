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

namespace ilm {

enum class LmNodeTYpe
{
  None,
  lm_pdn,
  lm_wire,
  lm_via,
  lm_pin,
  lm_io,
  lm_max
};

class LmNode
{
 public:
  LmNode() = default;
  ~LmNode() = default;

  // getter

  // setter
  void set_x(int x) { _x = x; }
  void set_y(int y) { _y = y; }
  void set_row_id(int row_id) { _row_id = row_id; }
  void set_col_id(int col_id) { _col_id = col_id; }
  void set_node_type(LmNodeTYpe node_type) { _node_type = node_type; }

  // operator

 private:
  int _x;
  int _y;
  int _row_id;  // node order of layer rows
  int _col_id;  // node order of layer cols
  LmNodeTYpe _node_type = LmNodeTYpe::None;
};

}  // namespace ilm
