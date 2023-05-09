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
#include "DrcConflictGraph.h"

namespace idrc {

DrcConflictGraph::~DrcConflictGraph()
{
  clearAllNode();
}

void DrcConflictGraph::clearAllNode()
{
  for (auto node : _conflict_graph) {
    if (node != nullptr) {
      delete node;
      node = nullptr;
    }
  }
  _conflict_graph.clear();
}

// void DrcConflictGraph::addEdge(DrcRect* spot_rect, DrcRect* conflict_rect)
// {
//   DrcConflictNode* spot_node = getConflictNode(spot_rect);
//   DrcConflictNode* conflict_node = getConflictNode(conflict_rect);
//   spot_node->addConflictNode(conflict_node);
//   conflict_node->addConflictNode(spot_node);
// }

DrcConflictNode* DrcConflictGraph::getConflictNode(DrcRect* conflict_rect)
{
  DrcConflictNode* conflict_node = nullptr;
  if (_rect_to_node.find(conflict_rect) == _rect_to_node.end()) {
    conflict_node = new DrcConflictNode(conflict_rect);
    _rect_to_node[conflict_rect] = conflict_node;
    _conflict_graph.push_back(conflict_node);
    conflict_node->set_node_id(_conflict_graph.size());
  } else {
    conflict_node = _rect_to_node[conflict_rect];
  }
  return conflict_node;
}
// init by drc_polygon
void DrcConflictGraph::initGraph(const std::map<DrcPolygon*, std::set<DrcPolygon*>>& conflict_map)
{
  std::map<DrcPolygon*, DrcConflictNode*> polygon_to_node;

  for (auto& [spot_polygon, conflict_polygon_list] : conflict_map) {
    DrcConflictNode* spot_node = getConflictNode(polygon_to_node, spot_polygon);

    for (auto conflict_polygon : conflict_polygon_list) {
      DrcConflictNode* conflict_node = getConflictNode(polygon_to_node, conflict_polygon);
      spot_node->addConflictNode(conflict_node);
      // conflict_node->addConflictNode(spot_node);
    }
  }
}
DrcConflictNode* DrcConflictGraph::getConflictNode(std::map<DrcPolygon*, DrcConflictNode*>& polygon_to_node, DrcPolygon* conflict_polygon)
{
  DrcConflictNode* conflict_node = nullptr;
  if (polygon_to_node.find(conflict_polygon) == polygon_to_node.end()) {
    conflict_node = new DrcConflictNode(conflict_polygon);
    polygon_to_node[conflict_polygon] = conflict_node;
    _conflict_graph.push_back(conflict_node);
    conflict_node->set_node_id(_conflict_graph.size());
  } else {
    conflict_node = polygon_to_node[conflict_polygon];
  }
  return conflict_node;
}
}  // namespace idrc