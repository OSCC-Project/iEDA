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

#include "api/TimingEngine.hh"
#include "gtest/gtest.h"
#include "log/Log.hh"
using namespace ista;
using ieda::Log;
using ieda::Stats;

namespace {

class DelayGPUTest : public testing::Test {
  void SetUp() {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() { Log::end(); }
};

TEST_F(DelayGPUTest, example1) {
  Stats stats;

  auto* timing_engine = TimingEngine::getOrCreateTimingEngine();
  timing_engine->set_num_threads(48);
  const char* design_work_space = "/home/longshuaiying/cuda_delay";
  timing_engine->set_design_work_space(design_work_space);

  std::vector<const char*> lib_files{
      "/home/taosimin/nangate45/lib/NangateOpenCellLibrary_typical.lib"};
  timing_engine->readLiberty(lib_files);

  timing_engine->get_ista()->set_analysis_mode(ista::AnalysisMode::kMaxMin);
  timing_engine->get_ista()->set_n_worst_path_per_clock(1);

  timing_engine->get_ista()->set_top_module_name("top");

  timing_engine->readDesign(
      "/home/taosimin/nangate45/design/example/example1.v");
  timing_engine->linkDesign("top");

  timing_engine->readSdc(
      "/home/taosimin/nangate45/design/example/example1.sdc");

  timing_engine->readSpef(
      "/home/taosimin/nangate45/design/example/example1.spef");

  LOG_INFO << "netlist instance num : "
           << timing_engine->get_netlist()->getInstanceNum();
  LOG_INFO << "netlist net num : " << timing_engine->get_netlist()->getNetNum();
}

TEST_F(DelayGPUTest, skywater130) {
  Stats stats;

  auto* timing_engine = TimingEngine::getOrCreateTimingEngine();
  timing_engine->set_num_threads(48);
  const char* design_work_space = "/home/longshuaiying/cuda_delay";
  timing_engine->set_design_work_space(design_work_space);

  std::vector<const char*> lib_files{
      "/home/taosimin/skywater130/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"};
  timing_engine->readLiberty(lib_files);

  timing_engine->get_ista()->set_analysis_mode(ista::AnalysisMode::kMaxMin);
  timing_engine->get_ista()->set_n_worst_path_per_clock(1);

  timing_engine->get_ista()->set_top_module_name("aes_cipher_top");

  timing_engine->readDesign(
      "/home/taosimin/skywater130/design/aes_cipher_top.v");
  timing_engine->linkDesign("aes_cipher_top");

  timing_engine->readSdc(
      "/home/taosimin/skywater130/design/aes_cipher_top.sdc");

  timing_engine->readSpef(
      "/home/taosimin/skywater130/spef/aes_cipher_top.spef");

  LOG_INFO << "netlist instance num : "
           << timing_engine->get_netlist()->getInstanceNum();
  LOG_INFO << "netlist net num : " << timing_engine->get_netlist()->getNetNum();
}

TEST_F(DelayGPUTest, T28) {
  Stats stats;

  auto* timing_engine = TimingEngine::getOrCreateTimingEngine();
  timing_engine->set_num_threads(48);
  const char* design_work_space = "/home/longshuaiying/cuda_delay";
  timing_engine->set_design_work_space(design_work_space);

  std::vector<const char*> lib_files{
      "/home/taosimin/T28/ccslib/tcbn28hpcplusbwp30p140hvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/tcbn28hpcplusbwp30p140lvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "tcbn28hpcplusbwp30p140mblvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/tcbn28hpcplusbwp30p140mbssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "tcbn28hpcplusbwp30p140opphvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "tcbn28hpcplusbwp30p140opplvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/tcbn28hpcplusbwp30p140oppssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "tcbn28hpcplusbwp30p140oppuhvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "tcbn28hpcplusbwp30p140oppulvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/tcbn28hpcplusbwp30p140ssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "tcbn28hpcplusbwp30p140uhvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "tcbn28hpcplusbwp30p140ulvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/tcbn28hpcplusbwp35p140hvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/tcbn28hpcplusbwp35p140lvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "tcbn28hpcplusbwp35p140mbhvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "tcbn28hpcplusbwp35p140mblvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/tcbn28hpcplusbwp35p140mbssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "tcbn28hpcplusbwp35p140opphvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "tcbn28hpcplusbwp35p140opplvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/tcbn28hpcplusbwp35p140oppssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "tcbn28hpcplusbwp35p140oppuhvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "tcbn28hpcplusbwp35p140oppulvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/tcbn28hpcplusbwp35p140ssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "tcbn28hpcplusbwp35p140uhvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "tcbn28hpcplusbwp35p140ulvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "tcbn28hpcplusbwp40p140ehvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/tcbn28hpcplusbwp40p140hvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/tcbn28hpcplusbwp40p140lvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "tcbn28hpcplusbwp40p140mbhvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/tcbn28hpcplusbwp40p140mbssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "tcbn28hpcplusbwp40p140oppehvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "tcbn28hpcplusbwp40p140opphvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "tcbn28hpcplusbwp40p140opplvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/tcbn28hpcplusbwp40p140oppssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "tcbn28hpcplusbwp40p140oppuhvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/tcbn28hpcplusbwp40p140ssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "tcbn28hpcplusbwp40p140uhvtssg0p81v125c_ccs.lib",
      "/home/taosimin/T28/ccslib/"
      "ts5n28hpcplvta256x32m4fw_130a_ssg0p81v125c.lib",
      "/home/taosimin/T28/ccslib/"
      "ts5n28hpcplvta64x128m2fw_130a_ssg0p81v125c.lib",
      "/home/taosimin/T28/ccslib/tphn28hpcpgv18ssg0p81v1p62v125c.lib",
      "/home/taosimin/T28/ccslib/PLLTS28HPMLAINT_SS_0P81_125C.lib"};
  timing_engine->readLiberty(lib_files);

  timing_engine->get_ista()->set_analysis_mode(ista::AnalysisMode::kMaxMin);
  timing_engine->get_ista()->set_n_worst_path_per_clock(1);

  timing_engine->get_ista()->set_top_module_name("asic_top");

  timing_engine->readDesign("/home/taosimin/T28/tapout/asic_top_1220.v");
  timing_engine->linkDesign("asic_top");

  timing_engine->readSdc("/home/taosimin/T28/ieda_1204/asic_top_SYN_MAX.sdc");

  timing_engine->readSpef("/home/taosimin/T28/spef/asic_top.rcworst.125c.spef");
  LOG_INFO << "netlist instance num : "
           << timing_engine->get_netlist()->getInstanceNum();
  LOG_INFO << "netlist net num : " << timing_engine->get_netlist()->getNetNum();
}

}  // namespace