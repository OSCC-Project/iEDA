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
#include "OddCycleFinder.h"

namespace idrc {

std::vector<std::vector<DrcConflictNode*>>& OddCycleFinder::findAllOddCycles(std::vector<DrcConflictGraph*> connected_component_list)
{
  for (auto sub_graph : connected_component_list) {
    findOddCyclesInConnectedComponent(sub_graph);
  }
  return _odd_cycle_list;
}

bool OddCycleFinder::isIgnoredNode(DrcConflictNode* node)
{
  return _ignored_node_list.find(node) != _ignored_node_list.end();
}

bool OddCycleFinder::isBlock(DrcConflictNode* node)
{
  return _blocked_set.find(node) != _blocked_set.end();
}

bool OddCycleFinder::isStoredCycle(const std::vector<DrcConflictNode*>& path)
{
  std::vector<DrcConflictNode*> temp_path = path;
  std::reverse(temp_path.begin(), temp_path.end());
  for (auto& cycle : _odd_cycle_list) {
    if (cycle == temp_path) {
      return true;
    }
  }
  return false;
}

void OddCycleFinder::findOddCyclesInConnectedComponent(DrcConflictGraph* connected_component)
{
  _blocked_set.clear();
  _blocked_map.clear();
  _ignored_node_list.clear();

  _origin_subgraph_node_list.clear();
  std::vector<DrcConflictNode*> subgraph_node_list = connected_component->get_conflict_node_list();
  _origin_subgraph_node_list.insert(subgraph_node_list.begin(), subgraph_node_list.end());
  for (DrcConflictNode* conflict_node : connected_component->get_conflict_node_list()) {
    findOddCyclesInConnectedComponent(conflict_node, conflict_node);
    _ignored_node_list.insert(conflict_node);
  }
}

void OddCycleFinder::storeOddCycle(DrcConflictNode* start_node)
{
  int cycle_edge_num = _temp_stack.size();
  if (cycle_edge_num % 2 == 0) {
    std::vector<DrcConflictNode*> path;
    path = _temp_stack;
    path.push_back(start_node);
    if (!isStoredCycle(path)) {
      _odd_cycle_list.push_back(path);
    }
  }
}

void OddCycleFinder::unlock(DrcConflictNode* node)
{
  _blocked_set.erase(node);
  for (auto map_node : _blocked_map[node]) {
    if (isBlock(map_node)) {
      unlock(map_node);
    }
  }
  _blocked_map[node].clear();
  _blocked_map.erase(node);
}

void OddCycleFinder::storeBlockMap(DrcConflictNode* current_node)
{
  for (auto confilct_node : current_node->get_conflict_node_list()) {
    if (!isIgnoredNode(confilct_node)) {
      _blocked_map[confilct_node].push_back(current_node);
    }
  }
}

bool OddCycleFinder::findOddCyclesInConnectedComponent(DrcConflictNode* start_node, DrcConflictNode* current_node)
{
  bool findCycle = false;
  _temp_stack.push_back(current_node);
  _blocked_set.insert(current_node);

  for (auto conflict_node : current_node->get_conflict_node_list()) {
    if (isIgnoredNode(conflict_node)) {
      continue;
    }

    if (conflict_node == start_node) {
      findCycle = true;
      storeOddCycle(start_node);
    } else if (!isBlock(conflict_node) && _origin_subgraph_node_list.find(conflict_node) != _origin_subgraph_node_list.end()) {
      bool getCycle = findOddCyclesInConnectedComponent(start_node, conflict_node);
      findCycle = findCycle || getCycle;
    }
  }

  if (findCycle) {
    unlock(current_node);
  } else {
    storeBlockMap(current_node);
  }

  _temp_stack.pop_back();
  return findCycle;
}
}  // namespace idrc