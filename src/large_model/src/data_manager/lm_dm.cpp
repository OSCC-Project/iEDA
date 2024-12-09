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

#include "lm_dm.h"

#include "Log.hh"
#include "lm_file.h"
#include "lm_graph_check.hh"
#include "lm_wire_pattern.hh"

namespace ilm {

bool LmDataManager::buildLayoutData()
{
  return layout_dm.buildLayoutData();
}

bool LmDataManager::buildGraphData()
{
  LmGraphDataManager graph_dm(&layout_dm.get_layout());
  bool b_success = graph_dm.buildGraphData();

  return b_success;
}

bool LmDataManager::buildPatternData()
{
  LmWirePatternGenerator wire_pattern_gen;
  wire_pattern_gen.genPatterns();
  wire_pattern_gen.patternSummary("/home/liweiguo/temp/file/pattern_summary.csv");
  return true;
}

bool LmDataManager::checkData()
{
  auto& graph = layout_dm.get_graph();

  if (graph.size() > 0) {
    // connectiviy check
    LmLayoutChecker checker;
    // checker.checkPinConnection(graph);
    return checker.checkLayout(graph);
  }

  return false;
}

std::map<int, LmNet> LmDataManager::getGraph(std::string path)
{
  return layout_dm.get_graph();
}

void LmDataManager::saveData(const std::string dir)
{
  LmLayoutFileIO file_io(dir, &layout_dm.get_layout());
  file_io.saveJson();
}

}  // namespace ilm