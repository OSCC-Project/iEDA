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

#include "vec_dm.h"

#include "Log.hh"
#include "vec_file.h"
#include "vec_graph_check.hh"
#include "vec_graph_dm.h"
#include "vec_patch_dm.h"
#include "vec_wire_pattern.hh"

namespace ivec {

bool VecDataManager::buildLayoutData()
{
  return layout_dm.buildLayoutData();
}

bool VecDataManager::buildGraphData()
{
  VecGraphDataManager graph_dm(&layout_dm.get_layout());
  bool b_success = graph_dm.buildGraphData();

  return b_success;
}

bool VecDataManager::buildPatternData()
{
  VecWirePatternGenerator wire_pattern_gen;
  wire_pattern_gen.genPatterns();
  wire_pattern_gen.patternSummary("/home/liweiguo/temp/file/pattern_summary.csv");
  return true;
}

bool VecDataManager::buildPatchData(const std::string dir)
{
  patch_dm = new VecPatchDataManager(&layout_dm.get_layout());
  return patch_dm->buildPatchData();
}

bool VecDataManager::buildPatchData(const std::string dir, int patch_row_step, int patch_col_step)
{
  patch_dm = new VecPatchDataManager(&layout_dm.get_layout());
  return patch_dm->buildPatchData(patch_row_step, patch_col_step);
}

bool VecDataManager::checkData()
{
  auto& graph = layout_dm.get_graph();

  if (graph.size() > 0) {
    // connectiviy check
    VecLayoutChecker checker;
    // checker.checkPinConnection(graph);
    return checker.checkLayout(graph);
  }

  return false;
}

std::map<int, VecNet> VecDataManager::getGraph(std::string path)
{
  return layout_dm.get_graph();
}

void VecDataManager::saveData(const std::string dir)
{
  VecLayoutFileIO file_io(dir, &layout_dm.get_layout(), &patch_dm->get_patch_grid());
  file_io.saveJson();
}

bool VecDataManager::readNetsToIDB(std::string dir)
{
  VecLayoutFileIO file_io(dir, &layout_dm.get_layout());
  return file_io.readJsonNets();
}

}  // namespace ivec