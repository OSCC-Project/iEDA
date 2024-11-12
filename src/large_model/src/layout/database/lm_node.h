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
  kNone,
  lm_pdn,
  lm_net,
  lm_pin,
  lm_io,
  lm_obs,
  kMax
};

enum class LmNodeStatus
{
  kNone,
  lm_connected,   /// connect points, including via, end point of wire, connected points pon pins
  lm_connecting,  /// points between end points
  lm_fix,         /// objects that no need to connect such as obs, pdn eg.
  kMax
};

enum class LmNodeDirection : uint8_t
{
  kNone = 0,
  lm_left = 1,
  lm_right = 2,
  lm_up = 4,
  lm_down = 8,
  lm_top = 16,
  lm_bottom = 32,
  kMax
};

class LmNodeData
{
 public:
  LmNodeData() = default;
  ~LmNodeData() = default;

  // getter
  LmNodeTYpe get_type() { return _type; }
  LmNodeStatus get_status() { return _status; }
  LmNodeDirection get_direction() { return _direction; }

  // setter
  void set_type(LmNodeTYpe type) { _type = type; }
  void set_status(LmNodeStatus status) { _status = status; }
  void set_net_id(int32_t id) { _net_id = id; }
  void set_direction(LmNodeDirection direction);

  // operator

 private:
  int32_t _net_id = -1;
  LmNodeTYpe _type = LmNodeTYpe::kNone;
  LmNodeStatus _status = LmNodeStatus::kNone;
  LmNodeDirection _direction = LmNodeDirection::kNone;
};

class LmNode
{
 public:
  LmNode() = default;
  ~LmNode() = default;

  // getter
  int get_x() { return _x; }
  int get_y() { return _y; }
  int get_row_id() { return _row_id; }
  int get_col_id() { return _col_id; }

  LmNodeData& get_node_data() { return _node_data; }

  bool isSteinerPoint();
  bool isVia();

  // setter
  void set_x(int x) { _x = x; }
  void set_y(int y) { _y = y; }
  void set_row_id(int row_id) { _row_id = row_id; }
  void set_col_id(int col_id) { _col_id = col_id; }

  // operator

 private:
  int _x;
  int _y;
  int _row_id;  // node order of layer rows
  int _col_id;  // node order of layer cols
  LmNodeData _node_data;
};

}  // namespace ilm
