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

class LmNetWire
{
 public:
  LmNetWire(LmNode* node1, LmNode* node2) { _node_pair = std::make_pair(node1, node2); };
  ~LmNetWire() = default;

  // getter
  std::pair<LmNode*, LmNode*>& get_connected_nodes() { return _node_pair; }

  // setter

  // operator

 private:
  std::pair<LmNode*, LmNode*> _node_pair;
};

class LmNet
{
 public:
  LmNet(int net_id) : _net_id(net_id) {}
  ~LmNet() = default;

  // getter
  int get_net_id() { return _net_id; }
  std::vector<LmNetWire> get_wires() { return _wires; }

  // setter
  void set_net_id(int net_id) { _net_id = net_id; }

  // operator
  void addWire(LmNetWire wire) { _wires.push_back(wire); }
  void addWire(LmNode* node1, LmNode* node2) { _wires.push_back(LmNetWire(node1, node2)); }

 private:
  int _net_id = -1;
  std::vector<LmNetWire> _wires;
};

}  // namespace ilm
