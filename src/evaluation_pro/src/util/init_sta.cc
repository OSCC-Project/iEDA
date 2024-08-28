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
 * @file init_sta.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-08-25
 */

#include "init_sta.hh"

#include "RTInterface.hpp"
#include "api/PowerEngine.hh"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "feature_irt.h"
#include "idm.h"
namespace ieval {
#define STA_INST (ista::TimingEngine::getOrCreateTimingEngine())
#define RT_INST (irt::RTInterface::getInst())
#define PW_INST (ipower::PowerEngine::getOrCreatePowerEngine())
InitSTA::~InitSTA()
{
  PW_INST->destroyPowerEngine();
  STA_INST->destroyTimingEngine();
}

void InitSTA::runSTA()
{
  if (_routing_type == RoutingType::kEGR || _routing_type == RoutingType::kDR) {
    callRT();
    return;
  }
  embeddingSTA();
}

double InitSTA::evalNetPower(const std::string& net_name) const
{
  for (const auto& data : PW_INST->get_power()->get_switch_powers()) {
    auto* net = dynamic_cast<ista::Net*>(data->get_design_obj());
    if (net->get_name() != net_name) {
      continue;
    }
    return data->get_switch_power();
  }
  return 0;
}

std::map<std::string, double> InitSTA::evalAllNetPower() const
{
  std::map<std::string, double> power_map;
  for (const auto& data : PW_INST->get_power()->get_switch_powers()) {
    auto* net = dynamic_cast<ista::Net*>(data->get_design_obj());
    power_map[net->get_name()] = data->get_switch_power();
  }
  return power_map;
}

void InitSTA::callRT()
{
  LOG_FATAL_IF(_routing_type != RoutingType::kEGR && _routing_type != RoutingType::kDR) << "Unsupported routing type";
  std::map<std::string, std::any> config_map;
  config_map.insert({"-enable_timing", 1});
  RT_INST.initRT(config_map);

  if (_routing_type == RoutingType::kEGR) {
    RT_INST.runEGR();
  } else if (_routing_type == RoutingType::kDR) {
    RT_INST.runRT();
  }

  getInfoFromRT();
}

void InitSTA::getInfoFromRT()
{
  LOG_FATAL_IF(_routing_type != RoutingType::kEGR && _routing_type != RoutingType::kDR) << "Unsupported routing type";
  auto summary = RT_INST.outputSummary();
  auto clocks_timing
      = _routing_type == RoutingType::kEGR ? summary.ir_summary.clocks_timing : summary.iter_dr_summary_map.rbegin()->second.clocks_timing;
  auto power_info
      = _routing_type == RoutingType::kEGR ? summary.ir_summary.power_info : summary.iter_dr_summary_map.rbegin()->second.power_info;
  for (auto clock_timing : clocks_timing) {
    auto clk_name = clock_timing.clock_name;
    _timing[clk_name]["TNS"] = clock_timing.setup_tns;
    _timing[clk_name]["WNS"] = clock_timing.setup_wns;
    _timing[clk_name]["Freq(MHz)"] = clock_timing.suggest_freq;
  }
  _power["static_power"] = power_info.static_power;
  _power["dynamic_power"] = power_info.dynamic_power;
}

void InitSTA::embeddingSTA()
{
  initStaEngine();
  buildRCTree();
  initPowerEngine();

  // get timing and power
  getInfoFromSTA();
  getInfoFromPW();
}

void InitSTA::initStaEngine()
{
  if (STA_INST->isBuildGraph()) {
    return;
  }
  STA_INST->readLiberty(dmInst->get_config().get_lib_paths());
  auto sta_db_adapter = std::make_unique<ista::TimingIDBAdapter>(STA_INST->get_ista());
  sta_db_adapter->set_idb(dmInst->get_idb_builder());
  sta_db_adapter->convertDBToTimingNetlist();
  STA_INST->set_db_adapter(std::move(sta_db_adapter));
  STA_INST->readSdc(dmInst->get_config().get_sdc_path().c_str());
  STA_INST->buildGraph();
  STA_INST->initRcTree();
}

void InitSTA::buildRCTree()
{
  LOG_FATAL_IF(_routing_type != RoutingType::kWLM && _routing_type != RoutingType::kHPWL && _routing_type != RoutingType::kFLUTE)
      << "Unsupported routing type";
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