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
#ifndef IDRC_SRC_DB_CONFLICTGRAPH_H_
#define IDRC_SRC_DB_CONFLICTGRAPH_H_

#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <vector>

#include "DrcConflictNode.h"
#include "DrcRect.h"

namespace idrc {
class DrcConflictGraph
{
 public:
  DrcConflictGraph() {}
  ~DrcConflictGraph();

  // setter
  void addNode(DrcConflictNode* conflict_node) { _conflict_graph.push_back(conflict_node); }
  void set_conflict_graph(const std::vector<DrcConflictNode*>& conflict_node_list) { _conflict_graph = conflict_node_list; }
  // getter
  std::vector<DrcConflictNode*>& get_conflict_node_list() { return _conflict_graph; }
  int get_node_num() { return _conflict_graph.size(); }
  // function
  // void addEdge(DrcRect* rect1, DrcRect* rect2);
  DrcConflictNode* getConflictNode(DrcRect* conflict_rect);
  void clearRectToNodeRecord() { _rect_to_node.clear(); }
  void clearAllNode();
  // init by polygon
  void initGraph(const std::map<DrcPolygon*, std::set<DrcPolygon*>>& conflict_map);
  DrcConflictNode* getConflictNode(std::map<DrcPolygon*, DrcConflictNode*>& polygon_to_node, DrcPolygon* conflict_polygon);

 private:
  std::map<DrcRect*, DrcConflictNode*> _rect_to_node;
  std::vector<DrcConflictNode*> _conflict_graph;
};
}  // namespace idrc

#endif