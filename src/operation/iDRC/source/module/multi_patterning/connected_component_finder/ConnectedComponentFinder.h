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
#ifndef IDRC_SRC_MODULE_CONNECTED_COMPONENT_H_
#define IDRC_SRC_MODULE_CONNECTED_COMPONENT_H_
#include "DrcConflictGraph.h"

namespace idrc {
class ConnectedComponentFinder
{
 public:
  ConnectedComponentFinder() {}
  ~ConnectedComponentFinder();

  std::vector<DrcConflictGraph*>& getAllConnectedComponentInGraph(DrcConflictGraph* conflict_graph);

 private:
  int _index = 0;
  std::map<DrcConflictNode*, int> _dfn;
  std::map<DrcConflictNode*, int> _low;
  std::set<DrcConflictNode*> _node_in_stack;
  std::vector<DrcConflictNode*> _temp_stack;
  std::vector<DrcConflictGraph*> _connected_component_list;

  void init();
  bool isInStack(DrcConflictNode* conflict_node);
  void Tarjan(DrcConflictNode* node);
  void storeConnectedComponent(DrcConflictNode* node);
  void subGraphPruning(std::vector<DrcConflictNode*>& sub_graph);
};
}  // namespace idrc

#endif
