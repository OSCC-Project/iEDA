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
/**
 * @file IREval.cpp
 * @author Jianrong Su
 * @brief
 * @version 1.0
 * @date 2025-06-23
 */

#include "IREval.hh"

#include "PNPIdbWrapper.hh"
#include "idm.h"
#include "log/Log.hh"

namespace ipnp {

void IREval::runIREval()
{
  _power_engine = ipower::PowerEngine::getOrCreatePowerEngine();

  std::string power_net_name = "VDD";

  _power_engine->resetIRAnalysisData();
  _power_engine->buildPGNetWireTopo();
  _power_engine->runIRAnalysis(power_net_name);

  _power_engine->reportIRAnalysis();

  _coord_ir_map = _power_engine->displayIRDropMap();
}

void IREval::initIREval()
{
  LOG_INFO << "Start initialize IREval";

  _timing_engine = TimingEngine::getOrCreateTimingEngine();
  _timing_engine->set_num_threads(48);

  // Set working directory
  std::string design_work_space = "./";
  if (!pnpConfig->get_timing_design_workspace().empty()) {
    design_work_space = pnpConfig->get_timing_design_workspace();
  }
  _timing_engine->set_design_work_space(design_work_space.c_str());

  // Get library files from database manager instead of config
  _timing_engine->readLiberty(dmInst->get_config().get_lib_paths());

  _timing_engine->get_ista()->set_analysis_mode(ista::AnalysisMode::kMaxMin);
  _timing_engine->get_ista()->set_n_worst_path_per_clock(1);

  // Read DEF design
  _timing_engine->setDefDesignBuilder(dmInst->get_idb_builder());

  // Read SDC file
  _timing_engine->readSdc(dmInst->get_config().get_sdc_path().c_str());

  _timing_engine->buildGraph();
  _timing_engine->get_ista()->updateTiming();
  _timing_engine->reportTiming();

  _ista = Sta::getOrCreateSta();
  _ipower = ipower::Power::getOrCreatePower(&(_ista->get_graph()));

  _ipower->runCompleteFlow();

  LOG_INFO << "End initialize IREval";
}

double IREval::getMaxIRDrop() const
{
  if (_coord_ir_map.empty()) {
    return 0.0;
  }
  double max_ir_drop = -std::numeric_limits<double>::max();
  for (auto it = _coord_ir_map.begin(); it != _coord_ir_map.end(); ++it) {
    if (it->second > max_ir_drop) {
      max_ir_drop = it->second;
    }
  }
  return max_ir_drop;
}

double IREval::getMinIRDrop() const
{
  if (_coord_ir_map.empty()) {
    return 0.0;
  }
  double min_ir_drop = std::numeric_limits<double>::max();
  for (auto it = _coord_ir_map.begin(); it != _coord_ir_map.end(); ++it) {
    if (it->second < min_ir_drop) {
      min_ir_drop = it->second;
    }
  }
  return min_ir_drop;
}

double IREval::getAvgIRDrop() const
{
  if (_coord_ir_map.empty()) {
    return 0.0;
  }
  double sum_ir_drop = 0.0;
  int count = 0;
  for (auto it = _coord_ir_map.begin(); it != _coord_ir_map.end(); ++it) {
    sum_ir_drop += it->second;
    count++;
  }
  return sum_ir_drop / count;
}

}  // namespace ipnp