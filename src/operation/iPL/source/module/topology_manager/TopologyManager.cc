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

namespace ipl {

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

bool NetWork::isIgnoreNetwork(){
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

void TopologyManager::add_node(std::string name, Node* node)
{
  _node_list.push_back(node);
  _node_map.emplace(name, node);
}

void TopologyManager::add_network(std::string name, NetWork* network)
{
  _network_list.push_back(network);
  _network_map.emplace(name, network);
}

void TopologyManager::add_group(std::string name, Group* group)
{
  _group_list.push_back(group);
  _group_map.emplace(name, group);
}

Node* TopologyManager::findNode(std::string name)
{
  auto node_it = _node_map.find(name);
  if (node_it == _node_map.end()) {
    return nullptr;
  } else {
    return node_it->second;
  }
}

NetWork* TopologyManager::findNetwork(std::string name)
{
  auto network_it = _network_map.find(name);
  if (network_it == _network_map.end()) {
    return nullptr;
  } else {
    return network_it->second;
  }
}

Group* TopologyManager::findGroup(std::string name)
{
  auto group_it = _group_map.find(name);
  if (group_it == _group_map.end()) {
    return nullptr;
  } else {
    return group_it->second;
  }
}

}  // namespace ipl