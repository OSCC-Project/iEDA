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
#include "api/PowerEngine.hh"
#include "api/TimingEngine.hh"
#include "gtest/gtest.h"
#include "log/Log.hh"
#include "usage/usage.hh"

using namespace ipower;
using namespace ieda;
using ieda::Stats;

namespace {
class PowerEngineTest : public testing::Test {
  void SetUp() final {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() final { Log::end(); }
};

TEST_F(PowerEngineTest, example1) {
  Stats stats;

  Log::setVerboseLogLevel("Pwr*", 1);

  auto* timing_engine = TimingEngine::getOrCreateTimingEngine();
  timing_engine->set_num_threads(48);
  const char* design_work_space = "/home/taosimin/power";
  timing_engine->set_design_work_space(design_work_space);

  std::vector<const char*> lib_files{
      "/home/taosimin/nangate45/lib/NangateOpenCellLibrary_typical.lib"};
  timing_engine->readLiberty(lib_files);

  timing_engine->get_ista()->set_analysis_mode(ista::AnalysisMode::kMaxMin);
  timing_engine->get_ista()->set_n_worst_path_per_clock(1);

  timing_engine->get_ista()->set_top_module_name("top");

  timing_engine->readDesign(
      "/home/taosimin/nangate45/design/example/example1.v");

  timing_engine->readSdc(
      "/home/taosimin/nangate45/design/example/example1.sdc");

  auto* power_engine = PowerEngine::getOrCreatePowerEngine();
  power_engine->creatDataflow();

  auto connection_map = power_engine->buildConnectionMap(
      {{"r1", "u1"}, {"r2", "u2"}, {"r3"}, {"in1"}, {"in2"}, {"out"}}, {}, 2);

  for (auto [src_cluster_id, snk_clusters] : connection_map) {
    for (auto snk_cluster : snk_clusters) {
      std::string stages_str = std::accumulate(
          snk_cluster._stages_each_hop.begin(),
          snk_cluster._stages_each_hop.end(), std::string(),
          [](const auto a, const auto b) { return a + " " + std::to_string(b); });
      LOG_INFO << "src cluster id " << src_cluster_id << " -> "
               << "snk cluster id " << snk_cluster._dst_cluster_id << " stages "
               << stages_str << " hop " << snk_cluster._hop;
    }
  }
}

}  // namespace

