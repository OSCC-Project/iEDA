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

#include "large_model.h"

#include "Log.hh"
#include "MemoryMonitor.hh"
#include "idm.h"
#include "init_sta.hh"
#include "lm_feature.h"

namespace ilm {
LargeModel::LargeModel()
{
  initLog();
}

void LargeModel::initLog(std::string log_path)
{
  char config[] = "large model";
  char* argv[] = {config};

  if (log_path == "") {
    log_path = dmInst->get_config().get_output_path() + "/log/";
  }

  ieda::Log::init(argv, log_path);
}

bool LargeModel::buildLayoutData(const std::string path)
{
  bool b_success = _data_manager.buildLayoutData();

  return b_success;
}

bool LargeModel::buildGraphData(const std::string path)
{
  bool b_success = _data_manager.buildGraphData();

  _data_manager.checkData();

  _data_manager.saveData(path);

  return b_success;
}

bool LargeModel::buildGraphDataWithoutSave(const std::string path)
{
  bool b_success = _data_manager.buildGraphData();
  return b_success;
}

std::map<int, LmNet> LargeModel::getGraph(std::string path)
{
  return _data_manager.getGraph(path);
}

void LargeModel::buildFeature(const std::string dir)
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
    buildPatchData(dir);
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

void LargeModel::generateFeature(const std::string dir)
{
  auto* patch_grid = _data_manager.patch_dm == nullptr ? nullptr : &_data_manager.patch_dm->get_patch_grid();
  LmFeature feature(&_data_manager.layout_dm.get_layout(), patch_grid, dir);
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

/// for run large model sta api.
bool LargeModel::runLmSTA(const std::string dir)
{
  auto* eval_tp = ieval::InitSTA::getInst();  // evaluate timing and power.

  eval_tp->runLmSTA(&_data_manager.layout_dm.get_layout(), dir);

  return true;
}

bool LargeModel::buildPatchData(const std::string dir)
{
  return _data_manager.buildPatchData(dir);
}

}  // namespace ilm