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

#include "vectorization.h"

#include "Log.hh"
#include "MemoryMonitor.hh"
#include "idm.h"
#include "init_sta.hh"
#include "vec_feature.h"

namespace ivec {
Vectorization::Vectorization()
{
  initLog();
}

void Vectorization::initLog(std::string log_path)
{
  char config[] = "vectorization";
  char* argv[] = {config};

  if (log_path == "") {
    log_path = dmInst->get_config().get_output_path() + "/log/";
  }

  ieda::Log::init(argv, log_path);
}

bool Vectorization::buildLayoutData(const std::string path)
{
  bool b_success = _data_manager.buildLayoutData();

  return b_success;
}

bool Vectorization::buildGraphData(const std::string path)
{
  bool b_success = _data_manager.buildGraphData();

  _data_manager.checkData();

  _data_manager.saveData(path);

  return b_success;
}

bool Vectorization::buildGraphDataWithoutSave(const std::string path)
{
  bool b_success = _data_manager.buildGraphData();
  return b_success;
}

std::map<int, VecNet> Vectorization::getGraph(std::string path)
{
  return _data_manager.getGraph(path);
}

void Vectorization::buildFeature(const std::string dir, int patch_row_step, int patch_col_step)
{
  {
    /// build layout data
    MemoryMonitor monitor("buildLayoutData", "./memory_usage.log");
    _data_manager.buildLayoutData();
  }

  {
    /// build graph
    MemoryMonitor monitor("buildGraphData", "./memory_usage.log");
    _data_manager.buildGraphData();
  }

  {
    /// build patch data
    MemoryMonitor monitor("buildPatchData", "./memory_usage.log");
    // buildPatchData(dir);
    buildPatchData(dir, patch_row_step, patch_col_step);  // default patch size
  }
  // build pattern
  // _data_manager.buildPatternData();

  /// check data
  bool check_ok = _data_manager.checkData();

  /// build feature
  generateFeature(dir);

  /// save
  _data_manager.saveData(dir);
}

void Vectorization::generateFeature(const std::string dir)
{
  auto* patch_grid = _data_manager.patch_dm == nullptr ? nullptr : &_data_manager.patch_dm->get_patch_grid();
  VecFeature feature(&_data_manager.layout_dm.get_layout(), patch_grid, dir);
  {
    MemoryMonitor monitor("buildFeatureTiming", "./memory_usage.log");
    feature.buildFeatureTiming();
  }
  {
    MemoryMonitor monitor("buildFeatureDrc", "./memory_usage.log");
    feature.buildFeatureDrc();
  }
  {
    MemoryMonitor monitor("buildFeatureStatis", "./memory_usage.log");
    feature.buildFeatureStatis();
  }
}

/// for run vectorization sta api.
bool Vectorization::runVecSTA(const std::string dir)
{
  auto* eval_tp = ieval::InitSTA::getInst();  // evaluate timing and power.

  eval_tp->runVecSTA(&_data_manager.layout_dm.get_layout(), dir);

  return true;
}

bool Vectorization::buildPatchData(const std::string dir)
{
  return _data_manager.buildPatchData(dir);
}

bool Vectorization::buildPatchData(const std::string dir, int patch_row_step, int patch_col_step)
{
  return _data_manager.buildPatchData(dir, patch_row_step, patch_col_step);
}

bool Vectorization::readNetsToIDB(const std::string dir)
{
  bool b_success = _data_manager.buildLayoutData();
  if (b_success) {
    b_success = _data_manager.readNetsToIDB(dir);
  }

  return b_success;
}

bool Vectorization::readNetsPatternToIDB(const std::string path)
{
  bool b_success = _data_manager.buildLayoutData();
  if (b_success) {
    b_success = _data_manager.readNetsPatternToIDB(path);
  }

  return b_success;
}

}  // namespace ivec