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
/*
 * @Author: S.J Chen
 * @Date: 2022-03-08 12:18:27
 * @LastEditTime: 2022-04-12 10:56:25
 * @LastEditors: S.J Chen
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/util/topology_manager/TopologyManager.cc
 * Contact : https://github.com/sjchanson
 */

#include "TopologyManager.hh"

#include <cmath>
#include <queue>
#include <tuple>

#include "Log.hh"

namespace ipl {
#define TOPO_SMALL_GAP 10
#define TOPO_LARGE_GAP 1000

std::vector<Node*> NetWork::get_node_list() const
{
  std::vector<Node*> node_list;
  node_list.reserve(_receiver_list.size() + 1);
  if (_transmitter) {
    node_list.push_back(_transmitter);
  }
  node_list.insert(node_list.end(), _receiver_list.begin(), _receiver_list.end());
  return node_list;
}

bool NetWork::isIgnoreNetwork()
{
  return (_net_weight - 0.0f) < 1e-9;
}

Rectangle<int32_t> NetWork::obtainNetWorkShape()
{
  int32_t lower_x = INT32_MAX;
  int32_t lower_y = INT32_MAX;
  int32_t upper_x = INT32_MIN;
  int32_t upper_y = INT32_MIN;

  for (auto* node : this->get_node_list()) {
    Point<int32_t> node_loc = node->get_location();

    node_loc.get_x() < lower_x ? lower_x = node_loc.get_x() : lower_x;
    node_loc.get_y() < lower_y ? lower_y = node_loc.get_y() : lower_y;
    node_loc.get_x() > upper_x ? upper_x = node_loc.get_x() : upper_x;
    node_loc.get_y() > upper_y ? upper_y = node_loc.get_y() : upper_y;
  }

  return Rectangle<int32_t>(lower_x, lower_y, upper_x, upper_y);
}

std::vector<Node*> Group::obtainInputNodes()
{
  std::vector<Node*> inputs;
  for (auto* node : _node_list) {
    if (node->get_node_type() == NODE_TYPE::kInput) {
      inputs.push_back(node);
    }
  }
  return inputs;
}

std::vector<Node*> Group::obtainOutputNodes()
{
  std::vector<Node*> outputs;
  for (auto* node : _node_list) {
    if (node->get_node_type() == NODE_TYPE::kOutput) {
      outputs.push_back(node);
    }
  }
  return outputs;
}

void TopologyManager::add_node(Node* node)
{
  _node_list.push_back(node);
  node->set_node_id(_nodes_range);
  _nodes_range += 1;
}

void TopologyManager::add_network(NetWork* network)
{
  _network_list.push_back(network);
  network->set_network_id(_networks_range);
  _networks_range += 1;
}

void TopologyManager::add_group(Group* group)
{
  _group_list.push_back(group);
  group->set_group_id(_groups_range);
  _groups_range += 1;
}

Node* TopologyManager::findNodeById(int32_t node_id)
{
  if (node_id < 0 || node_id >= _nodes_range) {
    return nullptr;
  } else {
    return _node_list[node_id];
  }
}

NetWork* TopologyManager::findNetworkById(int32_t network_id)
{
  if (network_id < 0 || network_id >= _networks_range) {
    return nullptr;
  } else {
    return _network_list[network_id];
  }
}

Group* TopologyManager::findGroupById(int32_t group_id)
{
  if (group_id < 0 || group_id >= _groups_range) {
    return nullptr;
  } else {
    return _group_list[group_id];
  }
}

void TopologyManager::updateTopoId(Node* node)
{
  int32_t lower = -std::numeric_limits<int32_t>::infinity();
  bool has_lower = false;
  for (auto* predecessor_arc : node->get_input_arc_list()) {
    auto* predecessor = predecessor_arc->get_from_node();
    lower = std::max(lower, predecessor->get_topo_id());
    has_lower = true;
  }

  int32_t upper = +std::numeric_limits<int32_t>::infinity();
  bool has_upper = false;
  for (auto* successor_arc : node->get_output_arc_list()) {
    auto* successor = successor_arc->get_to_node();
    upper = std::min(upper, successor->get_topo_id());
    has_upper = true;
  }

  if (!has_lower && !has_upper) {
    node->set_topo_id(0);
  } else if (has_lower && !has_upper) {
    node->set_topo_id(lower + TOPO_SMALL_GAP);
  } else if (!has_lower && has_upper) {
    node->set_topo_id(upper - TOPO_SMALL_GAP);
  } else {
    if (lower < upper && (upper - lower >= 2)) {
      node->set_topo_id((lower + upper) / 2);
    } else {
      int32_t small_gap = TOPO_SMALL_GAP;
      int32_t large_gap = TOPO_LARGE_GAP;

      int32_t left_0 = upper;
      int32_t left_1 = lower + small_gap + 1;
      int32_t right = left_1 + large_gap;
      int32_t w_0 = (right - left_0);
      int32_t w_1 = (right - left_1);

      node->set_topo_id((lower + left_1) / 2);

      std::queue<std::tuple<Node*, int32_t>> open;
      for (auto* successor_arc : node->get_output_arc_list()) {
        auto* successor = successor_arc->get_to_node();
        if (successor->get_topo_id() < right) {
          open.push(std::make_tuple(successor, successor->get_topo_id()));
        }
      }

      while (!open.empty()) {
        auto* current = std::get<0>(open.front());
        int32_t generator_order = std::get<1>(open.front());
        open.pop();

        if (current == node) {
          // loop detected.
          LOG_WARNING << "Loop detected.";
          continue;
        }

        if (current->get_topo_id() > generator_order) {
          // no need to continue propagating...
          continue;
        }

        int32_t order = static_cast<int32_t>(std::floor(float((current->get_topo_id() - upper) * w_1) / float(w_0)) + left_1);
        if (order <= generator_order) {
          order = generator_order + small_gap;
        }

        current->set_topo_id(order);

        for (auto* successor_arc : node->get_output_arc_list()) {
          auto* successor = successor_arc->get_to_node();
          if (successor->get_topo_id() <= order) {
            open.push(std::make_tuple(successor, order));
          }
        }
      }
    }
  }
}

void TopologyManager::updateALLNodeTopoId()
{
  for (auto* node : _node_list) {
    updateTopoId(node);
  }
}

}  // namespace ipl