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

enum class LmNodeTYpe : uint8_t
{
  kNone = 0,
  lm_pdn = 1,
  lm_net = 2,
  lm_pin = 4,
  lm_io = 8,
  lm_obs = 16,
  kMax
};

enum class LmNodeConnectType : uint8_t
{
  kNone = 0,
  lm_wire = 1,
  lm_delta = 2,
  lm_via = 4,
  lm_enclosure = 8,
  kMax
};

enum class LmNodeStatus : uint8_t
{
  kNone = 0,
  lm_connected = 1,   /// connected points, including via, end point of wire, connected points on pins, use to describe wire end points pair
  lm_connecting = 2,  /// points that are not end point but as a connecting point in a wire
  lm_end_point = 4,   /// end points for wire
  lm_fix = 8,         /// objects that no need to connect such as obs, pdn eg.
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
  lm_middle = 64,
  kMax
};

class LmNodeData
{
 public:
  LmNodeData() = default;
  ~LmNodeData() = default;

  // getter
  int32_t get_net_id() { return _net_id; }
  LmNodeTYpe get_type() { return _type; }
  LmNodeConnectType get_connect_type() { return _connect_type; }
  bool is_connect_type(LmNodeConnectType type);
  bool is_wire() { return is_connect_type(LmNodeConnectType::lm_wire); }
  bool is_delta() { return is_connect_type(LmNodeConnectType::lm_delta); }
  bool is_via() { return is_connect_type(LmNodeConnectType::lm_via); }
  bool is_enclosure() { return is_connect_type(LmNodeConnectType::lm_enclosure); }
  LmNodeStatus get_status() { return _status; }
  bool is_status(LmNodeStatus type);
  bool is_connected() { return is_status(LmNodeStatus::lm_connected); }
  bool is_connecting() { return is_status(LmNodeStatus::lm_connecting); }
  bool is_end_point() { return is_status(LmNodeStatus::lm_end_point); }
  bool is_fix() { return is_status(LmNodeStatus::lm_fix); }
  LmNodeDirection get_direction() { return _direction; }
  bool is_direction(LmNodeDirection direction);
  bool is_direction_visited(LmNodeDirection direction);
  bool is_visited();

  bool is_net() { return is_type(LmNodeTYpe::lm_net); }
  bool is_pdn() { return is_type(LmNodeTYpe::lm_pdn); }
  bool is_pin() { return is_type(LmNodeTYpe::lm_pin); }
  bool is_io() { return is_type(LmNodeTYpe::lm_io); }
  bool is_obs() { return is_type(LmNodeTYpe::lm_obs); }

  // setter
  void set_net_id(int32_t id) { _net_id = id; }
  void set_type(LmNodeTYpe type);
  void set_connect_type(LmNodeConnectType type);
  void set_status(LmNodeStatus status, bool b_cancel = false);
  void set_direction(LmNodeDirection direction);
  void set_direction_visited(LmNodeDirection direction);
  void set_visited();

  // operator

 private:
  int32_t _net_id = -1;
  LmNodeTYpe _type = LmNodeTYpe::kNone;  /// multiple type in one node
  LmNodeConnectType _connect_type = LmNodeConnectType::kNone;
  LmNodeStatus _status = LmNodeStatus::kNone;
  LmNodeDirection _direction = LmNodeDirection::kNone;  /// multiple direction in one node
  LmNodeDirection _visited = LmNodeDirection::kNone;    /// a temp data, describe visited status while travel the grid

  bool is_type(LmNodeTYpe type);
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
  int32_t get_layer_id() { return _layer_id; }

  LmNodeData& get_node_data() { return _node_data; }

  bool is_steiner_point();
  bool is_corner();
  bool is_via();
  bool is_end_point();

  // setter
  void set_x(int x) { _x = x; }
  void set_y(int y) { _y = y; }
  void set_row_id(int row_id) { _row_id = row_id; }
  void set_col_id(int col_id) { _col_id = col_id; }
  void set_layer_id(int32_t layer_id) { _layer_id = layer_id; }

  // operator

 private:
  int _x;
  int _y;
  int _row_id;  // node order of layer rows
  int _col_id;  // node order of layer cols
  int32_t _layer_id = -1;
  LmNodeData _node_data;
};

}  // namespace ilm
