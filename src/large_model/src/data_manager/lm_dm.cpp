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

namespace ilm {

bool LmDataManager::buildLayoutData(const std::string path)
{
  LmLayoutDataManager layout_dm;
  return layout_dm.buildLayoutData();
}

bool LmDataManager::buildGraphData(const std::string dir)
{
  LmLayoutDataManager layout_dm;
  bool b_success = layout_dm.buildGraphData();

  auto& graph = layout_dm.get_graph();

  if (graph.size() > 0) {
    // connectiviy check
    LmLayoutChecker checker;
    LOG_ERROR_IF(!checker.checkLayout(graph)) << "Graph is not connected";
    /// save to path
    LmLayoutFileIO file_io(dir);
    file_io.saveJson(graph);
  }

  //   LmGraphDataManager dm;
  //   return dm.buildGraphData(path);

  return b_success;
}

std::map<int, LmNet> LmDataManager::getGraph(std::string path)
{
  LmGraphDataManager layout_dm;
  return layout_dm.get_graph(path);
}

}  // namespace ilm