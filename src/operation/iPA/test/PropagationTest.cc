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
#include <signal.h>

#include <cstdlib>
#include <fstream>

#include "api/Power.hh"
#include "api/TimingEngine.hh"
#include "gtest/gtest.h"
#include "include/PwrConfig.hh"
#include "log/Log.hh"
#include "sta/StaApplySdc.hh"
#include "sta/StaBuildPropTag.hh"
#include "sta/StaClockPropagation.hh"
#include "sta/StaConstPropagation.hh"
#include "sta/StaDataPropagation.hh"
#include "sta/StaDelayPropagation.hh"
#include "sta/StaSlewPropagation.hh"
#include "usage/usage.hh"

using namespace ipower;
using namespace ieda;
using ieda::Stats;
namespace {
class PropagationTest : public testing::Test {
  void SetUp() final {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() final { Log::end(); }
};

TEST_F(PropagationTest, seq_graph) {
  Log::setVerboseLogLevel("Pwr*", 1);

  Stats stats;

  auto* timing_engine = TimingEngine::getOrCreateTimingEngine();
  timing_engine->set_num_threads(48);
  const char* design_work_space = "/home/taosimin/power";
  timing_engine->set_design_work_space(design_work_space);

  std::vector<const char*> lib_files = {};

  timing_engine->readLiberty(lib_files);

  timing_engine->get_ista()->set_analysis_mode(ista::AnalysisMode::kMaxMin);
  timing_engine->get_ista()->set_n_worst_path_per_clock(1);

  timing_engine->get_ista()->set_top_module_name("asic_top");
  timing_engine->readDesign("");

  timing_engine->readSdc("");

  timing_engine->readSpef("");

  timing_engine->buildGraph();

  StaGraph& the_graph = timing_engine->get_ista()->get_graph();

  Vector<std::function<unsigned(StaGraph*)>> funcs = {
      StaApplySdc(StaApplySdc::PropType::kApplySdcPreProp),
      StaConstPropagation(),
      StaClockPropagation(StaClockPropagation::PropType::kIdealClockProp),
      StaSlewPropagation(),
      StaDelayPropagation(),
      StaClockPropagation(StaClockPropagation::PropType::kNormalClockProp),
      StaApplySdc(StaApplySdc::PropType::kApplySdcPostClockProp),
      StaBuildPropTag(StaPropagationTag::TagType::kProp),
      StaDataPropagation(StaDataPropagation::PropType::kFwdProp)};

  for (auto& func : funcs) {
    the_graph.exec(func);
  }

  Power ipower(&(timing_engine->get_ista()->get_graph()));
  // set fastest clock for default toggle.
  auto* fastest_clock = timing_engine->get_ista()->getFastestClock();
  ipower.setFastestClock(fastest_clock->get_clock_name(),
                         fastest_clock->getPeriodNs());

  // get sta clocks
  auto clocks = timing_engine->get_ista()->getClocks();
  ipower.setStaClocks(std::move(clocks));

  // build power graph.
  ipower.buildGraph();

  // build seq graph
  ipower.buildSeqGraph();

  // check pipeline loop.
  ipower.checkPipelineLoop();

  // build seq graphviz
  // ipower.dumpSeqGraphViz();

  // levelize seq graph.
  ipower.levelizeSeqGraph();

  // propagate const.
  ipower.propagateConst();

  // dump graph information.
  // ipower.dumpGraph();

  ipower.propagateToggleSP();
  // timing_engine->updateTiming();

  ipower.propagateClock();

  // calc leakage power
  ipower.calcLeakagePower();

  // calc internal power
  ipower.calcInternalPower();

  // calc switch power.
  ipower.calcSwitchPower();

  ipower.analyzeGroupPower();

  ipower.reportSummaryPower("report.txt", PwrAnalysisMode::kAveraged);

  double memory_delta = stats.memoryDelta();
  LOG_INFO << "memory usage " << memory_delta << "MB";

  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "time elapsed " << time_delta << "s";
}
}  // namespace