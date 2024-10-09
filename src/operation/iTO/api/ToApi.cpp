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

  std::map<std::string, ieda_feature::TOClockTimingCmp> summary_map;

  // origin data，tns，wns，freq
  auto to_eval_data = timingEngine->eval_data();
  for (auto eval_data : to_eval_data) {
    ieda_feature::TOClockTiming clock_timing;
    std::string clock_name = eval_data.name;
    clock_timing.tns = eval_data.initial_tns;
    clock_timing.wns = eval_data.initial_wns;
    clock_timing.suggest_freq = eval_data.initial_freq;

    ieda_feature::TOClockTimingCmp clock_cmp;
    memset(&clock_cmp, 0, sizeof(ieda_feature::TOClockTimingCmp));
    clock_cmp.origin = clock_timing;
    summary_map[clock_name] = clock_cmp;
  }

  // after optimize timing
  auto clk_list = timingEngine->get_sta_engine()->getClockList();

  std::ranges::for_each(clk_list, [&](ista::StaClock* clk) {
    auto clk_name = clk->get_clock_name();
    auto drv_tns = timingEngine->get_sta_engine()->getTNS(clk_name, AnalysisMode::kMax);
    auto drv_wns = timingEngine->get_sta_engine()->getWNS(clk_name, AnalysisMode::kMax);
    auto suggest_freq = 1000.0 / (clk->getPeriodNs() - drv_wns);

    ieda_feature::TOClockTiming clock_timing;
    std::string clock_name = clk_name;
    clock_timing.tns = drv_tns;
    clock_timing.wns = drv_wns;
    clock_timing.suggest_freq = suggest_freq;

    summary_map[clock_name].opt = clock_timing;
  });

  for (auto [clock_name, clock_timings] : summary_map) {
    clock_timings.clock_name = clock_name;

    clock_timings.delta.tns = clock_timings.opt.tns - clock_timings.origin.tns;
    clock_timings.delta.wns = clock_timings.opt.wns - clock_timings.origin.wns;
    clock_timings.delta.suggest_freq = clock_timings.opt.suggest_freq - clock_timings.origin.suggest_freq;

    to_summary.clock_timings.push_back(clock_timings);
  }

  return to_summary;
}

}  // namespace ito
