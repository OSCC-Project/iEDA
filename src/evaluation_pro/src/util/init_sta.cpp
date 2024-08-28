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
 * @file init_sta.cpp
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-08-25
 */

#include "init_sta.h"

#include "TimingEngine.hh"
#include "idm.h"

namespace ieval {
#define STA_INST ista::TimingEngine::getOrCreateTimingEngine()
#define PW_INST ipower::PowerEngine::getOrCreatePowerEngine()
InitSTA::~InitSTA()
{
  PW_INST->destroyPowerEngine();
  STA_INST->destroyTimingEngine();
}

void InitSTA::runSTA()
{

}

void InitSTA::embeddingSTA()
{
  initStaEngine();
  // TODO: build rc tree for each net

  // report timing
  STA_INST->updateTiming();

  // report power
  initPowerEngine();

  // get timing and power
  getInfoFromSTA();
  getInfoFromPW();
}

void InitSTA::initStaEngine()
{
  STA_INST->set_design_work_space(_work_dir.c_str());
  STA_INST->set_num_threads(_num_threads);
  STA_INST->get_ista()->set_n_worst_path_per_clock(_n_worst);

  if (STA_INST->isBuildGraph()) {
    return;
  }
  STA_INST->readLiberty(dmInst->get_config().get_lib_paths());
  auto sta_db_adapter = std::make_unique<ista::TimingIDBAdapter>(STA_INST->get_ista());
  sta_db_adapter->set_idb(dmInst->get_idb_builder());
  sta_db_adapter->convertDBToTimingNetlist();
  STA_INST->set_db_adapter(std::move(sta_db_adapter));
  STA_INST->readSdc(dmInst->get_config().get_sdc_path().c_str());
  STA_INST->initRcTree();
  STA_INST->buildGraph();
  STA_INST->updateTiming();
}

void InitSTA::initPowerEngine()
{
  if (PW_INST->isBuildGraph()) {
    return;
  }
  PW_INST->get_power()->initPowerGraphData();
  PW_INST->get_power()->initToggleSPData();
  PW_INST->get_power()->updatePower();
}

void InitSTA::getInfoFromSTA()
{
  auto clk_list = STA_INST->getClockList();
  std::ranges::for_each(clk_list, [&](ista::StaClock* clk) {
    auto clk_name = clk->get_clock_name();
    auto setup_tns = STA_INST->getTNS(clk_name, AnalysisMode::kMax);
    auto setup_wns = STA_INST->getWNS(clk_name, AnalysisMode::kMax);
    auto suggest_freq = 1000.0 / (clk->getPeriodNs() - setup_wns);
    _timing[clk_name]["TNS"] = setup_tns;
    _timing[clk_name]["WNS"] = setup_wns;
    _timing[clk_name]["Freq(MHz)"] = suggest_freq;
  });
}

void InitSTA::getInfoFromPW()
{
  double static_power = 0;
  for (const auto& data : PW_INST->get_power()->get_leakage_powers()) {
    static_power += data->get_leakage_power();
  }
  double dynamic_power = 0;
  for (const auto& data : PW_INST->get_power()->get_internal_powers()) {
    dynamic_power += data->get_internal_power();
  }
  for (const auto& data : PW_INST->get_power()->get_switch_powers()) {
    dynamic_power += data->get_switch_power();
  }
  _power["static_power"] = static_power;
  _power["dynamic_power"] = dynamic_power;
}

}  // namespace ieval