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
#ifndef IDRC_SRC_MODULE_MULTI_PATTERNING_H_
#define IDRC_SRC_MODULE_MULTI_PATTERNING_H_
#include <fstream>

#include "BoostType.h"
#include "ColorableChecker.h"
#include "ConnectedComponentFinder.h"
#include "DrcPolygon.h"
#include "OddCycleFinder.h"
namespace idrc {
class MultiPatterning
{
 public:
  MultiPatterning() {}
  explicit MultiPatterning(DrcConflictGraph* conflict_graph) { _conflict_graph = conflict_graph; }
  ~MultiPatterning() {}

  // setter
  void set_conflict_graph(DrcConflictGraph* graph) { _conflict_graph = graph; }

  // std::vector<std::vector<DrcConflictNode*>> checkDoublePatterning();
  std::vector<DrcConflictNode*> checkTriplePatterning();
  std::vector<DrcConflictNode*> checkMultiPatterning(int check_colorable_num);

 private:
  ConnectedComponentFinder _connected_component_finder;
  OddCycleFinder _odd_cycle_finder;
  ColorableChecker _colorable_checker;
  DrcConflictGraph* _conflict_graph = nullptr;

  void repotResult(std::vector<DrcConflictNode*>& uncolorable_node_list, int optional_color_num);
};
}  // namespace idrc

#endif
