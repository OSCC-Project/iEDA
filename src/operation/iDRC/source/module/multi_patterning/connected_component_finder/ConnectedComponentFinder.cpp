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
#include "ConnectedComponentFinder.h"
namespace idrc {

ConnectedComponentFinder::~ConnectedComponentFinder()
{
  for (DrcConflictGraph* sub_graph : _connected_component_list) {
    if (sub_graph != nullptr) {
      delete sub_graph;
      sub_graph = nullptr;
    }
  }
}

bool ConnectedComponentFinder::isInStack(DrcConflictNode* conflict_node)
{
  return _node_in_stack.find(conflict_node) != _node_in_stack.end();
}

void ConnectedComponentFinder::Tarjan(DrcConflictNode* node)
{
  _dfn[node] = _low[node] = ++_index;
  _temp_stack.push_back(node);
  _node_in_stack.insert(node);
  for (auto conflict_node : node->get_conflict_node_list()) {
    if (_dfn.find(conflict_node) == _dfn.end()) {
      conflict_node->set_parent_node(node);
      Tarjan(conflict_node);
      _low[node] = std::min(_low[node], _low[conflict_node]);
      conflict_node->erase_parent_node();
    } else if (isInStack(conflict_node)) {
      if (conflict_node != node->get_parent_node()) {
        _low[node] = std::min(_low[node], _dfn[conflict_node]);
      }
    }
  }

  if (_low[node] == _dfn[node]) {
    storeConnectedComponent(node);
  }
}

void ConnectedComponentFinder::storeConnectedComponent(DrcConflictNode* node)
{
  std::vector<DrcConflictNode*> record_node_list;
  DrcConflictNode* record_node;
  do {
    record_node = _temp_stack.back();
    record_node_list.push_back(record_node);
    _temp_stack.pop_back();
    _node_in_stack.erase(node);

  } while (record_node != node);

  if (record_node_list.size() >= 3) {
    DrcConflictGraph* connected_component = new DrcConflictGraph();
    subGraphPruning(record_node_list);
    connected_component->set_conflict_graph(record_node_list);
    _connected_component_list.push_back(connected_component);
  }
}

std::vector<DrcConflictGraph*>& ConnectedComponentFinder::getAllConnectedComponentInGraph(DrcConflictGraph* conflict_graph)
{
  init();
  for (auto node : conflict_graph->get_conflict_node_list()) {
    if (_dfn.find(node) == _dfn.end()) {
      Tarjan(node);
    }
  }
  return _connected_component_list;
}

void ConnectedComponentFinder::init()
{
  _index = 0;
  _dfn.clear();
  _low.clear();
  _node_in_stack.clear();
  _temp_stack.clear();
  _connected_component_list.clear();
}

void ConnectedComponentFinder::subGraphPruning(std::vector<DrcConflictNode*>& sub_graph)
{
  std::set<DrcConflictNode*> temp_node_set(sub_graph.begin(), sub_graph.end());
  for (auto& node : sub_graph) {
    std::vector<DrcConflictNode*> temp_node_vector;

    for (auto& conflict_node : node->get_conflict_node_list()) {
      if (temp_node_set.find(conflict_node) != temp_node_set.end()) {
        temp_node_vector.push_back(conflict_node);
      }
    }
    node->set_conflict_node_list(temp_node_vector);
  }
}

}  // namespace idrc