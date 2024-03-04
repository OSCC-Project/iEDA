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
 * @Date: 2022-04-11 12:06:03
 * @LastEditTime: 2022-12-14 14:49:15
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/evaluator/wirelength/SteinerWirelength.cc
 * Contact : https://github.com/sjchanson
 */

#include "SteinerWirelength.hh"

#include <map>
#include <queue>
#include <set>

namespace ipl {

void SteinerWirelength::updateAllNetWorkPointPair()
{
  if (_point_pair_map.empty()) {
    initAllNetWorkPointPair();
  } else {
    updatePartOfNetWorkPointPair(_topology_manager->get_network_list());
  }
}

void SteinerWirelength::updateNetWorkPointPair(NetWork* network){
  std::vector<std::pair<Point<int32_t>, Point<int32_t>>> point_pair;
  this->obtainNetWorkPointPair(network, point_pair);

  auto iter = _point_pair_map.find(network);
  if (iter == _point_pair_map.end()) {
    _point_pair_map.emplace(network, point_pair);
  } else {
    iter->second = point_pair;
  }
}

void SteinerWirelength::updatePartOfNetWorkPointPair(const std::vector<NetWork*>& network_list)
{
  for (auto* network : network_list) {
    std::vector<std::pair<Point<int32_t>, Point<int32_t>>> point_pair;
    obtainNetWorkPointPair(network, point_pair);

    auto iter = _point_pair_map.find(network);
    if (iter == _point_pair_map.end()) {
      _point_pair_map.emplace(network, point_pair);
    } else {
      iter->second = point_pair;
    }
  }
}

void SteinerWirelength::initAllNetWorkPointPair()
{
  for (auto* network : _topology_manager->get_network_list()) {
    std::vector<std::pair<Point<int32_t>, Point<int32_t>>> point_pair;

    obtainNetWorkPointPair(network, point_pair);
    _point_pair_map.emplace(network, point_pair);
  }
}

void SteinerWirelength::obtainNetWorkPointPair(NetWork* network, std::vector<std::pair<Point<int32_t>, Point<int32_t>>>& point_pair)
{
  const auto& node_list = network->get_node_list();

  size_t node_num = node_list.size();
  if (node_num <= 1) {
    // TODO.
  } else if (node_num == 2) {
    // point_pair.push_back(std::make_pair(node_list.at(0)->get_location(), node_list.at(1)->get_location()));
    auto point_1 = node_list.at(0)->get_location();
    auto point_2 = node_list.at(1)->get_location();
    // Deal with the oblique line.
    if ((point_1.get_x() != point_2.get_x()) && (point_1.get_y() != point_2.get_y())) {
      Point<int32_t> point_3(point_1.get_x(), point_2.get_y());
      point_pair.push_back(std::make_pair(point_1, point_3));
      point_pair.push_back(std::make_pair(point_2, point_3));
    } else {
      point_pair.push_back(std::make_pair(point_1, point_2));
    }

  } else {
    std::set<Point<int32_t>, PointCMP> coord_set;
    // deal with the repeating location's node.
    for (auto* node : node_list) {
      Point<int32_t> node_loc = std::move(node->get_location());
      coord_set.emplace(node_loc);
    }
    std::vector<Point<int32_t>> point_vec;
    point_vec.assign(coord_set.begin(), coord_set.end());
    obtainFlutePointPair(point_vec, point_pair);
  }
}

void SteinerWirelength::obtainFlutePointPair(std::vector<Point<int32_t>>& point_vec,
                                             std::vector<std::pair<Point<int32_t>, Point<int32_t>>>& point_pair)
{
  Flute::Tree flute_tree;
  size_t coord_num = point_vec.size();
  Flute::DTYPE* x = (Flute::DTYPE*) malloc(sizeof(Flute::DTYPE) * (coord_num));
  Flute::DTYPE* y = (Flute::DTYPE*) malloc(sizeof(Flute::DTYPE) * (coord_num));

  for (size_t i = 0; i < point_vec.size(); ++i) {
    x[i] = static_cast<Flute::DTYPE>(point_vec[i].get_x());
    y[i] = static_cast<Flute::DTYPE>(point_vec[i].get_y());
  }

  flute_tree = Flute::flute(coord_num, x, y, FLUTE_ACCURACY);
  free(x);
  free(y);

  int branch_num = 2 * flute_tree.deg - 2;
  point_pair.reserve(branch_num);

  for (int j = 0; j < branch_num; ++j) {
    int n = flute_tree.branch[j].n;
    if (j == n) {
      continue;
    }
    Point<int32_t> point_1(flute_tree.branch[j].x, flute_tree.branch[j].y);
    Point<int32_t> point_2(flute_tree.branch[n].x, flute_tree.branch[n].y);

    // dual with the repetitive point pair.
    if (point_1 == point_2) {
      continue;
    }

    // dual with the oblique line.
    if ((point_1.get_x() != point_2.get_x()) && (point_1.get_y() != point_2.get_y())) {
      Point<int32_t> point_3(point_1.get_x(), point_2.get_y());
      point_pair.push_back(std::make_pair(point_1, point_3));
      point_pair.push_back(std::make_pair(point_2, point_3));
      continue;
    }

    point_pair.push_back(std::make_pair(point_1, point_2));
  }
}

const std::vector<std::pair<Point<int32_t>, Point<int32_t>>>& SteinerWirelength::obtainPointPairList(NetWork* network)
{
  auto iter = _point_pair_map.find(network);
  if (iter == _point_pair_map.end()) {
    LOG_ERROR << "NetWork : " << network->get_name() << " has not been initialized!";
    exit(1);
  }

  return iter->second;
}

MultiTree* SteinerWirelength::obtainMultiTree(NetWork* network)
{
  // obtain the point pair list
  auto iter = _point_pair_map.find(network);
  if (iter == _point_pair_map.end()) {
    LOG_ERROR << "NetWork : " << network->get_name() << " has not been initialized!";
    exit(1);
  }
  auto& point_pair_list = iter->second;

  // looking for the overlap node.
  std::map<Point<int32_t>, Node*, PointCMP> point_to_node;
  std::map<Node*, Node*> node_to_overlap_node;
  for (auto* node : network->get_node_list()) {
    const auto& node_loc = node->get_location();
    auto loc_iter = point_to_node.find(node_loc);
    if (loc_iter != point_to_node.end()) {
      auto* driven_node = loc_iter->second;
      node_to_overlap_node.emplace(driven_node, node);
    } else {
      point_to_node.emplace(node_loc, node);
    }
  }

  // construct the point multitree.
  auto* first_node = network->get_node_list().at(0);
  MultiTree* multi_tree = obtainPointMultiTree(first_node->get_location(), point_pair_list);
  multi_tree->set_network(network);

  // add node in tree node.
  std::queue<TreeNode*> tree_node_queue;
  tree_node_queue.push(multi_tree->get_root());
  while (!tree_node_queue.empty()) {
    auto* tree_node = tree_node_queue.front();
    auto iter_1 = point_to_node.find(tree_node->get_point());
    if (iter_1 != point_to_node.end()) {
      tree_node->set_node(iter_1->second);

      // for overlap node.
      auto iter_2 = node_to_overlap_node.find(iter_1->second);
      if (iter_2 != node_to_overlap_node.end()) {
        TreeNode* sink_tree_node = new TreeNode(iter_2->second->get_location());
        sink_tree_node->set_node(iter_2->second);
        sink_tree_node->set_parent(tree_node);
        tree_node->add_child(sink_tree_node);
      }
    }
    tree_node_queue.pop();
    for (auto* child_node : tree_node->get_child_list()) {
      if (child_node->get_node()) {
        continue;  // for child node already has tree node
      }

      tree_node_queue.push(child_node);
    }
  }

  return multi_tree;
}

MultiTree* SteinerWirelength::obtainPointMultiTree(Point<int32_t> root_point,
                                                   std::vector<std::pair<Point<int32_t>, Point<int32_t>>>& point_pair_list)
{
  std::set<Point<int32_t>, PointCMP> visited_point;
  std::map<Point<int32_t>, TreeNode*, PointCMP> point_to_tree_node;
  std::queue<Point<int32_t>> point_queue;
  point_queue.push(root_point);
  visited_point.emplace(root_point);

  TreeNode* root_node = new TreeNode(root_point);
  point_to_tree_node.emplace(root_point, root_node);

  while (!point_queue.empty()) {
    auto driven_point = point_queue.front();
    TreeNode* driven_node = point_to_tree_node[driven_point];

    std::vector<Point<int32_t>> sink_point_list;
    for (auto pair : point_pair_list) {
      if (pair.first == driven_point) {
        sink_point_list.push_back(pair.second);
      }
      if (pair.second == driven_point) {
        sink_point_list.push_back(pair.first);
      }
    }

    for (auto sink_point : sink_point_list) {
      if (visited_point.find(sink_point) == visited_point.end()) {
        point_queue.push(sink_point);
        TreeNode* sink_node = new TreeNode(sink_point);
        sink_node->set_parent(driven_node);
        driven_node->add_child(sink_node);
        point_to_tree_node.emplace(sink_point, sink_node);
        visited_point.emplace(sink_point);
      }
    }
    point_queue.pop();
  }

  MultiTree* multi_tree = new MultiTree(root_node);
  return multi_tree;
}

int64_t SteinerWirelength::obtainTotalWirelength()
{
  int64_t total_wirelength = 0;
  for (auto map_pair : _point_pair_map) {
    for (auto point_pair : map_pair.second) {
      total_wirelength += (std::abs(point_pair.first.get_x() - point_pair.second.get_x())
                           + std::abs(point_pair.first.get_y() - point_pair.second.get_y()));
    }
  }

  return total_wirelength;
}

int64_t SteinerWirelength::obtainNetWirelength(int32_t net_id)
{
  int64_t wirelength = 0;
  auto* network = _topology_manager->findNetworkById(net_id);
  LOG_ERROR_IF(!network) << "NetWork Index : " << net_id << " is not existed!";

  auto iter = _point_pair_map.find(network);
  if (iter != _point_pair_map.end()) {
    for (auto point_pair : iter->second) {
      wirelength += (std::abs(point_pair.first.get_x() - point_pair.second.get_x())
                     + std::abs(point_pair.first.get_y() - point_pair.second.get_y()));
    }
  } else {
    LOG_ERROR << "NetWork Index : " << net_id << " has not been initialized!";
  }

  return wirelength;
}

int64_t SteinerWirelength::obtainPartOfNetWirelength(int32_t net_id, int32_t sink_pin_id)
{
  int64_t netlength = -1;
  // TODO.
  return netlength;
}

}  // namespace ipl