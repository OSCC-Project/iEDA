// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#include "api/Power.hh"
#include "api/TimingEngine.hh"
#include "gtest/gtest.h"
#include "log/Log.hh"
#include "usage/usage.hh"

using namespace ipower;
using namespace ieda;
using ieda::Stats;

namespace {
class PowerTest : public testing::Test {
  void SetUp() final {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() final { Log::end(); }
};

TEST_F(PowerTest, example1) {
  Stats stats;

  Log::setVerboseLogLevel("Pwr*", 1);

  auto* timing_engine = TimingEngine::getOrCreateTimingEngine();
  timing_engine->set_num_threads(48);
  const char* design_work_space = "/home/shaozheqing/power";
  timing_engine->set_design_work_space(design_work_space);

  std::vector<const char*> lib_files{
      "/home/taosimin/iEDA/src/operation/iSTA/source/data/example1/"
      "example1_slow.lib"};
  timing_engine->readLiberty(lib_files);

  timing_engine->get_ista()->set_analysis_mode(ista::AnalysisMode::kMaxMin);
  timing_engine->get_ista()->set_n_worst_path_per_clock(1);

  timing_engine->get_ista()->set_top_module_name("top");

  timing_engine->readDesign(
      "/home/taosimin/iEDA/src/operation/iSTA/source/data/example1/"
      "example1.v");

  timing_engine->readSdc(
      "/home/taosimin/iEDA/src/operation/iSTA/source/data/example1/"
      "example1.sdc");

  timing_engine->readSpef(
      "/home/taosimin/iEDA/src/operation/iSTA/source/data/example1/"
      "example1.spef");

  timing_engine->buildGraph();

  timing_engine->updateTiming();
  timing_engine->reportTiming();

  Power ipower(&(timing_engine->get_ista()->get_graph()));
  // set fastest clock for default toggle.
  auto* fastest_clock = timing_engine->get_ista()->getFastestClock();
  PwrClock pwr_fastest_clock(fastest_clock->get_clock_name(),
                             fastest_clock->getPeriodNs());
  // get sta clocks
  auto clocks = timing_engine->get_ista()->getClocks();

  ipower.setupClock(std::move(pwr_fastest_clock), std::move(clocks));

  // build power graph.
  ipower.buildGraph();

  // build seq graph
  ipower.buildSeqGraph();

  // update power.
  ipower.updatePower();

  // report power.
  ipower.reportSummaryPower("report.txt", PwrAnalysisMode::kAveraged);
}

}  // namespace