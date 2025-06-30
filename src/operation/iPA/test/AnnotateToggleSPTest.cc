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
#include "ops/build_graph/PwrBuildGraph.hh"
#include "usage/usage.hh"

using namespace ipower;
using namespace ieda;
using ieda::Stats;

namespace {
class AnnotateToggleSPTest : public testing::Test {
  void SetUp() final {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() final { Log::end(); }
};

TEST_F(AnnotateToggleSPTest, calc_tc_sp) {
  Stats stats;

  auto* timing_engine = TimingEngine::getOrCreateTimingEngine();
  timing_engine->set_num_threads(48);
  const char* design_work_space = "";
  timing_engine->set_design_work_space(design_work_space);

  std::vector<const char*> lib_files = {};

  timing_engine->readLiberty(lib_files);

  timing_engine->get_ista()->set_analysis_mode(ista::AnalysisMode::kMaxMin);
  timing_engine->get_ista()->set_n_worst_path_per_clock(1);

  timing_engine->get_ista()->set_top_module_name("asic_top");
  timing_engine->readDesign("");

  timing_engine->readSdc("");

  timing_engine->buildGraph();

  Power ipower(&(timing_engine->get_ista()->get_graph()));
  ipower.buildGraph();

  ipower.readRustVCD("/home/shaozheqing/benchmark/asic_top.vcd", "u0_asic_top");
  ipower.annotateToggleSP();

  // timing_engine->updateTiming();

  double memory_delta = stats.memoryDelta();
  LOG_INFO << "memory usage " << memory_delta << "MB";

  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "time elapsed " << time_delta << "s";
}
}  // namespace