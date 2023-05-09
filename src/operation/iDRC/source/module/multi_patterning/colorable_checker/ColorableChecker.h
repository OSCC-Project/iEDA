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
#ifndef IDRC_SRC_MODULE_COLORABLE_CHECKER_H_
#define IDRC_SRC_MODULE_COLORABLE_CHECKER_H_
#include <limits.h>

#include "DrcConflictGraph.h"

namespace idrc {

class ColorableChecker
{
 public:
  ColorableChecker() {}
  ~ColorableChecker() {}

  std::vector<DrcConflictNode*>& colorable_check(const std::vector<DrcConflictGraph*>& sub_graph_list);
  void set_optional_color_num(int optional_color_num) { _optional_color_num = optional_color_num; }

 private:
  int _optional_color_num = 0;
  int _graph_node_num = 0;
  int _colorable_node_num = 0;
  int _fewest_uncolorable_node_num = INT_MAX;

  int _uncolorable_node_num = 0;
  bool _record_uncolorable_num = false;
  std::set<DrcConflictNode*> _visited;
  std::vector<DrcConflictNode*> _origin_subgraph_node_list;
  // std::map<DrcConflictNode*, int> _node_to_color;
  std::vector<DrcConflictNode*> _temp_uncolorable_node_list;
  std::vector<DrcConflictNode*> _uncolorable_node_list;

  // void colorable_check(DrcConflictGraph* sub_graph);
  bool judgeIsColorable(DrcConflictNode* node, int color);
  bool dfs(DrcConflictNode* node);
  void storeUncolorableNode();
  void addRecordsOfUncolorableNode();
  ////////////////////////////////////////////////////
  void colorable_check_new(DrcConflictGraph* sub_graph);
  bool DFS(int i);
};
}  // namespace idrc

#endif
