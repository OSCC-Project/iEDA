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

#include "lm_node.h"
#include "lm_shape.h"

namespace ilm {

class LmNetWire
{
 public:
  LmNetWire(LmNode* node1 = nullptr, LmNode* node2 = nullptr) { _node_pair = std::make_pair(node1, node2); };
  ~LmNetWire() = default;

  // getter
  std::pair<LmNode*, LmNode*>& get_connected_nodes() { return _node_pair; }
  std::vector<std::pair<LmNode*, LmNode*>>& get_paths() { return _paths; }

  // setter
  void set_start(LmNode* node) { _node_pair.first = node; }
  void set_end(LmNode* node) { _node_pair.second = node; }

  void add_path(LmNode* node1, LmNode* node2) { _paths.push_back(std::make_pair(node1, node2)); }
  void insert_paths(std::vector<std::pair<LmNode*, LmNode*>> paths) { _paths.insert(_paths.end(), paths.begin(), paths.end()); }

  // operator

 private:
  std::pair<LmNode*, LmNode*> _node_pair;

  std::vector<std::pair<LmNode*, LmNode*>> _paths;
};

class LmPin
{
 public:
  LmPin(int pin_id) { _pin_id = pin_id; };
  ~LmPin() = default;

  // getter
  int get_pin_id() { return _pin_id; }
  std::map<int, LmLayerShape>& get_shape_map() { return _shape_map; }
  LmLayerShape* get_layer_shape(int layer_id);

  // setter

  // operator
  void addShape(int layer, LmLayerShape shape) { _shape_map.insert(std::make_pair(layer, shape)); }

 private:
  int _pin_id = -1;

  std::map<int, LmLayerShape> _shape_map;
};

class LmNet
{
 public:
  LmNet(int net_id) : _net_id(net_id) {}
  ~LmNet() = default;

  // getter
  int get_net_id() { return _net_id; }
  std::vector<LmNetWire>& get_wires() { return _wires; }
  std::vector<int>& get_pin_ids() { return _pin_ids; }
  std::map<int, LmPin>& get_pin_map() { return _pin_map; }
  LmPin* get_pin(int pin_id);

  // setter
  void set_net_id(int net_id) { _net_id = net_id; }

  // operator
  void setWires(std::vector<LmNetWire> wires) { _wires = wires; }
  void addWire(LmNetWire wire);
  void addPinId(int id) { _pin_ids.push_back(id); }
  void addPin(LmPin lm_pin) { _pin_map.insert(std::make_pair(lm_pin.get_pin_id(), lm_pin)); }

 private:
  int _net_id = -1;
  std::vector<LmNetWire> _wires;
  std::vector<int> _pin_ids;
  std::map<int, LmPin> _pin_map;
};

class LmGraph
{
 public:
  LmGraph() {}
  ~LmGraph() = default;

  // getter
  std::map<int, LmNet>& get_net_map() { return _net_map; }
  LmNet* get_net(int net_id);

  // setter
  LmNet* addNet(int net_id);
  void add_net_wire(int net_id, LmNetWire wire);

  // operator

 private:
  std::map<int, LmNet> _net_map;
};

}  // namespace ilm
