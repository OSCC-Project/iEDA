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
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace ivec {

enum class VecNodeTYpe : uint8_t
{
  kNone = 0,
  vec_pdn = 1,
  vec_net = 2,
  vec_pin = 4,
  vec_io = 8,
  vec_obs = 16,
  vec_instance = 32,
  kMax
};

enum class VecNodeConnectType : uint8_t
{
  kNone = 0,
  vec_wire = 1,
  vec_delta = 2,
  vec_via = 4,
  vec_enclosure = 8,
  kMax
};

struct VecNodeFeature
{
  std::set<int> drc_ids;
};

class VecNodeData
{
 public:
  VecNodeData() {}
  ~VecNodeData() {}

  // getter
  int32_t get_net_id() { return _net_id; }
  int32_t get_pdn_id() { return _pdn_id; }
  int32_t get_pin_id() { return _pin_id; }
  int32_t get_instance_id() { return _inst_id; }
  int32_t get_via_id() { return _via_id; }
  VecNodeTYpe get_type() { return _type; }
  VecNodeConnectType get_connect_type() { return _connect_type; }
  VecNodeFeature& get_feature() { return _feature; }
  bool is_connect_type(VecNodeConnectType type);
  bool is_wire() { return is_connect_type(VecNodeConnectType::vec_wire); }
  bool is_delta() { return is_connect_type(VecNodeConnectType::vec_delta); }
  bool is_via() { return is_connect_type(VecNodeConnectType::vec_via); }
  bool is_enclosure() { return is_connect_type(VecNodeConnectType::vec_enclosure); }

  bool is_type(VecNodeTYpe type);
  bool is_net() { return is_type(VecNodeTYpe::vec_net); }
  bool is_pdn() { return is_type(VecNodeTYpe::vec_pdn); }
  bool is_pin() { return is_type(VecNodeTYpe::vec_pin); }
  bool is_io() { return is_type(VecNodeTYpe::vec_io); }
  bool is_obs() { return is_type(VecNodeTYpe::vec_obs); }

  // setter
  void set_net_id(int32_t id) { _net_id = id; }
  void set_pdn_id(int32_t id) { _pdn_id = id; }
  void set_pin_id(int32_t id);
  void set_instance_id(int32_t id) { _inst_id = id; }
  void set_via_id(int32_t id) { _via_id = id; }
  void set_type(VecNodeTYpe type);
  void set_connect_type(VecNodeConnectType type);

  // operator

 private:
  int32_t _net_id = -1;
  int32_t _pdn_id = -1;
  int32_t _pin_id = -1;
  int32_t _inst_id = -1;
  int32_t _via_id = -1;
  VecNodeTYpe _type = VecNodeTYpe::kNone;  /// multiple type in one node
  VecNodeConnectType _connect_type = VecNodeConnectType::kNone;
  VecNodeFeature _feature;
};

class VecNode
{
 public:
  VecNode() {}
  ~VecNode() {}

  // getter
  uint64_t get_node_id();
  int64_t get_x();
  int64_t get_y();
  int get_row_id() { return _row_id; }
  int get_col_id() { return _col_id; }
  int16_t get_layer_id() { return _layer_id; }
  int32_t get_realx() { return _real_x; }
  int32_t get_realy() { return _real_y; }

  VecNodeData* get_node_data(int net_id = -1, bool b_create = false);

  // setter
  void set_row_id(int row_id) { _row_id = row_id; }
  void set_col_id(int col_id) { _col_id = col_id; }
  void set_layer_id(int16_t layer_id) { _layer_id = layer_id; }
  void set_real_coordinate(int32_t real_x, int32_t real_y)
  {
    _real_x = real_x;
    _real_y = real_y;
  }

  // operator

 private:
  int _row_id = -1;  // node order of layer rows
  int _col_id = -1;  // node order of layer cols
  int16_t _layer_id = -1;
  VecNodeData* _node_data = nullptr;
  int32_t _real_x = -1;
  int32_t _real_y = -1;
  //   std::map<int, VecNodeData> _data_map;
};

}  // namespace ivec
