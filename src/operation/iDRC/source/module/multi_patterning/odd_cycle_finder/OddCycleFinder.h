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
#ifndef IDRC_SRC_MODULE_ODD_CYCLE_FINDER_H_
#define IDRC_SRC_MODULE_ODD_CYCLE_FINDER_H_
#include "DrcConflictGraph.h"

namespace idrc {
class OddCycleFinder
{
 public:
  OddCycleFinder() {}
  ~OddCycleFinder() {}
  std::vector<std::vector<DrcConflictNode*>>& findAllOddCycles(std::vector<DrcConflictGraph*> connected_component_list);

 private:
  std::set<DrcConflictNode*> _ignored_node_list;
  std::vector<DrcConflictNode*> _temp_stack;
  std::set<DrcConflictNode*> _blocked_set;
  std::map<DrcConflictNode*, std::vector<DrcConflictNode*>> _blocked_map;
  std::set<DrcConflictNode*> _origin_subgraph_node_list;
  std::vector<std::vector<DrcConflictNode*>> _odd_cycle_list;

  bool isIgnoredNode(DrcConflictNode* node);
  bool isBlock(DrcConflictNode* node);
  bool isStoredCycle(const std::vector<DrcConflictNode*>& path);
  void storeOddCycle(DrcConflictNode* start_node);
  void storeBlockMap(DrcConflictNode* current_node);
  void findOddCyclesInConnectedComponent(DrcConflictGraph* connected_component);
  bool findOddCyclesInConnectedComponent(DrcConflictNode* start_node, DrcConflictNode* current_node);
  void unlock(DrcConflictNode* node);
};
}  // namespace idrc

#endif