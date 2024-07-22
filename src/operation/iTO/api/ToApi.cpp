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
#include "ToApi.hpp"

#include "ToConfig.h"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "builder.h"
#include "data_manager.h"
#include "feature_ito.h"
#include "iTO.h"
#include "idm.h"
#include "timing_engine.h"

namespace ito {

ToApi* ToApi::_instance = nullptr;

ToApi::ToApi()
{
}

ToApi::~ToApi()
{
}

void ToApi::destroyInst()
{
  if (_instance != nullptr) {
    delete _instance;
    _instance = nullptr;
  }
}

void ToApi::init(const std::string& ITO_CONFIG_PATH)
{
  if (_ito == nullptr) {
    _ito = new ito::iTO(ITO_CONFIG_PATH);
  }
}

void ToApi::initEngine()
{
  timingEngine->initEngine();
}

void ToApi::runTO()
{
  _ito->runTO();
}

void ToApi::optimizeDrv()
{
  _ito->optimizeDrv();
}

void ToApi::optimizeDrvSpecialNet(const char* net_name)
{
  _ito->optimizeDrvSpecialNet(net_name);
}

void ToApi::optimizeSetup()
{
  _ito->optimizeSetup();
}

void ToApi::performBuffering(const char* net_name)
{
  _ito->performBuffering(net_name);
}

void ToApi::optimizeHold()
{
  _ito->optimizeHold();
}

void ToApi::saveDef(string saved_def_path)
{
  if (saved_def_path.empty()) {
    saved_def_path = toConfig->get_output_def_file();
  }
  dmInst->saveDef(saved_def_path);
}

void ToApi::resetConfigLibs(std::vector<std::string>& paths)
{
  toConfig->set_lib_files(paths);
}

void ToApi::resetConfigSdc(std::string& path)
{
  toConfig->set_sdc_file(path);
}

void ToApi::reportTiming()
{
  timingEngine->get_sta_engine()->reportTiming();
}

ieda_feature::TimingOptSummary ToApi::outputSummary()
{
  ieda_feature::TimingOptSummary to_summary;

  std::map<std::string, ieda_feature::TONetTimingCmp> summary_map;

  // origin data，tns，wns，freq
  auto to_eval_data = timingEngine->eval_data();
  for (auto eval_data : to_eval_data) {
    ieda_feature::TONetTiming net_timing;
    std::string net_name = eval_data.name;
    net_timing.tns = eval_data.initial_tns;
    net_timing.wns = eval_data.initial_wns;
    net_timing.suggest_freq = eval_data.initial_freq;

    ieda_feature::TONetTimingCmp net_cmp;
    memset(&net_cmp, 0, sizeof(ieda_feature::TONetTimingCmp));
    net_cmp.origin = net_timing;
    summary_map[net_name] = net_cmp;
  }

  // after optimize timing
  auto clk_list = timingEngine->get_sta_engine()->getClockList();

  std::ranges::for_each(clk_list, [&](ista::StaClock* clk) {
    auto clk_name = clk->get_clock_name();
    auto drv_tns = timingEngine->get_sta_engine()->getTNS(clk_name, AnalysisMode::kMax);
    auto drv_wns = timingEngine->get_sta_engine()->getWNS(clk_name, AnalysisMode::kMax);
    auto suggest_freq = 1000.0 / (clk->getPeriodNs() - drv_wns);

    ieda_feature::TONetTiming net_timing;
    std::string net_name = clk_name;
    net_timing.tns = drv_tns;
    net_timing.wns = drv_wns;
    net_timing.suggest_freq = suggest_freq;

    summary_map[net_name].opt = net_timing;
  });

  for (auto [net_name, net_timings] : summary_map) {
    net_timings.net_name = net_name;

    net_timings.delta.tns = net_timings.opt.tns - net_timings.origin.tns;
    net_timings.delta.wns = net_timings.opt.wns - net_timings.origin.wns;
    net_timings.delta.suggest_freq = net_timings.opt.suggest_freq - net_timings.origin.suggest_freq;

    to_summary.net_timings.push_back(net_timings);
  }

  return to_summary;
}

}  // namespace ito
