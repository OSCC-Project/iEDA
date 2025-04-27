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
#include "api/TimingIDBAdapter.hh"
#include "gtest/gtest.h"
#include "iIR/source/module/power-netlist/PGNetlist.hh"
#include "log/Log.hh"
#include "usage/usage.hh"

using namespace ipower;
using namespace ieda;
using ieda::Stats;
using namespace iir;

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
      "/home/taosimin/nangate45/lib/NangateOpenCellLibrary_typical.lib"};
  timing_engine->readLiberty(lib_files);

  timing_engine->get_ista()->set_analysis_mode(ista::AnalysisMode::kMaxMin);
  timing_engine->get_ista()->set_n_worst_path_per_clock(1);

  timing_engine->get_ista()->set_top_module_name("top");

  timing_engine->readDesign(
      "/home/taosimin/nangate45/design/example/example1.v");

  timing_engine->readSdc(
      "/home/taosimin/nangate45/design/example/example1.sdc");

  timing_engine->readSpef(
      "/home/taosimin/nangate45/design/example/example1.spef");

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
  ipower.reportInstancePower("report_instance.txt", PwrAnalysisMode::kAveraged);
  ipower.reportInstancePowerCSV("report_instance.csv");
}

TEST_F(PowerTest, runIR) {
  Log::setVerboseLogLevel("Pwr*", 1);

  auto* timing_engine = TimingEngine::getOrCreateTimingEngine();
  timing_engine->set_num_threads(48);
  const char* design_work_space = "/home/taosimin/ir_example/aes/rpt";
  timing_engine->set_design_work_space(design_work_space);

  std::vector<const char*> lib_files{
      "/home/taosimin/T28/lib/tcbn28hpcplusbwp40p140ssg0p81v125c.lib",
      "/home/taosimin/T28/lib/tcbn28hpcplusbwp40p140hvtssg0p81v125c.lib",
      "/home/taosimin/T28/lib/tcbn28hpcplusbwp35p140ssg0p81v125c.lib",
      "/home/taosimin/T28/lib/tcbn28hpcplusbwp35p140lvtssg0p81v125c.lib",
      "/home/taosimin/T28/lib/tcbn28hpcplusbwp40p140lvtssg0p81v125c.lib",
      "/home/taosimin/T28/lib/tcbn28hpcplusbwp30p140lvtssg0p81v125c.lib",
      "/home/taosimin/T28/lib/tcbn28hpcplusbwp30p140ssg0p81v125c.lib"};
  timing_engine->readLiberty(lib_files);

  timing_engine->get_ista()->set_analysis_mode(ista::AnalysisMode::kMaxMin);

  std::vector<std::string> lef_files{
      "/home/taosimin/T28/tlef/tsmcn28_9lm6X2ZUTRDL.tlef",
      "/home/taosimin/T28/lef/PLLTS28HPMLAINT.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140opplvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140lvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140uhvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140hvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140oppuhvt.lef",
      "/home/taosimin/T28/lef/ts5n28hpcplvta256x32m4fw_130a.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140cg.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140oppuhvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140mbhvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140ulvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140uhvt.lef",
      "/home/taosimin/T28/lef/ts5n28hpcplvta64x100m2fw_130a.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140hvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140oppulvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140mb.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140cgcwhvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140lvt.lef",
      "/home/taosimin/T28/lef/tpbn28v_9lm.lef",
      "/home/taosimin/T28/lef/ts5n28hpcplvta64x128m2f_130a.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140uhvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140mblvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140cgcw.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140mbhvt.lef",
      "/home/taosimin/T28/lef/tpbn28v.lef",
      "/home/taosimin/T28/lef/ts5n28hpcplvta64x128m2fw_130a.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140lvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140ulvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140opphvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140cgehvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140mb.lef",
      "/home/taosimin/T28/lef/tphn28hpcpgv18_9lm.lef",
      "/home/taosimin/T28/lef/ts5n28hpcplvta64x88m2fw_130a.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140mb.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140cghvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140opp.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140cghvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140oppehvt.lef",
      "/home/taosimin/T28/lef/ts1n28hpcplvtb2048x48m8sw_180a.lef",
      "/home/taosimin/T28/lef/ts5n28hpcplvta64x92m2fw_130a.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140mblvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140cg.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140opplvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140cg.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140opphvt.lef",
      "/home/taosimin/T28/lef/ts1n28hpcplvtb512x128m4sw_180a.lef",
      "/home/taosimin/T28/lef/ts5n28hpcplvta64x96m2fw_130a.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140opphvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140hvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140oppuhvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140cguhvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140opp.lef",
      "/home/taosimin/T28/lef/ts1n28hpcplvtb512x64m4sw_180a.lef",
      "/home/taosimin/T28/lef/ts6n28hpcplvta2048x32m8sw_130a.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140opp.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140oppulvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140ehvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140opplvt.lef",
      "/home/taosimin/T28/lef/ts1n28hpcplvtb8192x64m8sw_180a.lef"};

  std::string def_file = "/home/taosimin/ir_example/aes/aes.def";
  timing_engine->readDefDesign(def_file, lef_files);

  timing_engine->readSdc("/home/taosimin/ir_example/aes/aes.sdc");

  timing_engine->readSpef("/home/taosimin/ir_example/aes/aes.spef");

  timing_engine->buildGraph();

  timing_engine->get_ista()->updateTiming();
  timing_engine->reportTiming();

  Sta* ista = Sta::getOrCreateSta();
  Power* ipower = Power::getOrCreatePower(&(ista->get_graph()));

  ipower->runCompleteFlow();
  
  std::string power_net_name = "VDD";
  ipower->runIRAnalysis(power_net_name);
}

TEST_F(PowerTest, estimateIR) {
  // Log::setVerboseLogLevel("Pwr*", 1);

  auto* timing_engine = TimingEngine::getOrCreateTimingEngine();
  timing_engine->set_num_threads(48);
  const char* design_work_space = "/home/taosimin/ir_example/aes/rpt";
  timing_engine->set_design_work_space(design_work_space);

  std::vector<const char*> lib_files{
      "/home/taosimin/T28/lib/tcbn28hpcplusbwp40p140ssg0p81v125c.lib",
      "/home/taosimin/T28/lib/tcbn28hpcplusbwp40p140hvtssg0p81v125c.lib",
      "/home/taosimin/T28/lib/tcbn28hpcplusbwp35p140ssg0p81v125c.lib",
      "/home/taosimin/T28/lib/tcbn28hpcplusbwp35p140lvtssg0p81v125c.lib",
      "/home/taosimin/T28/lib/tcbn28hpcplusbwp40p140lvtssg0p81v125c.lib",
      "/home/taosimin/T28/lib/tcbn28hpcplusbwp30p140lvtssg0p81v125c.lib",
      "/home/taosimin/T28/lib/tcbn28hpcplusbwp30p140ssg0p81v125c.lib"};
  timing_engine->readLiberty(lib_files);

  timing_engine->get_ista()->set_analysis_mode(ista::AnalysisMode::kMaxMin);
  timing_engine->get_ista()->set_n_worst_path_per_clock(1);

  std::vector<std::string> lef_files{
      "/home/taosimin/T28/tlef/tsmcn28_9lm6X2ZUTRDL.tlef",
      "/home/taosimin/T28/lef/PLLTS28HPMLAINT.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140opplvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140lvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140uhvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140hvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140oppuhvt.lef",
      "/home/taosimin/T28/lef/ts5n28hpcplvta256x32m4fw_130a.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140cg.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140oppuhvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140mbhvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140ulvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140uhvt.lef",
      "/home/taosimin/T28/lef/ts5n28hpcplvta64x100m2fw_130a.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140hvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140oppulvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140mb.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140cgcwhvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140lvt.lef",
      "/home/taosimin/T28/lef/tpbn28v_9lm.lef",
      "/home/taosimin/T28/lef/ts5n28hpcplvta64x128m2f_130a.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140uhvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140mblvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140cgcw.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140mbhvt.lef",
      "/home/taosimin/T28/lef/tpbn28v.lef",
      "/home/taosimin/T28/lef/ts5n28hpcplvta64x128m2fw_130a.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140lvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140ulvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140opphvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140cgehvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140mb.lef",
      "/home/taosimin/T28/lef/tphn28hpcpgv18_9lm.lef",
      "/home/taosimin/T28/lef/ts5n28hpcplvta64x88m2fw_130a.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140mb.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140cghvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140opp.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140cghvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140oppehvt.lef",
      "/home/taosimin/T28/lef/ts1n28hpcplvtb2048x48m8sw_180a.lef",
      "/home/taosimin/T28/lef/ts5n28hpcplvta64x92m2fw_130a.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140mblvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140cg.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140opplvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140cg.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140opphvt.lef",
      "/home/taosimin/T28/lef/ts1n28hpcplvtb512x128m4sw_180a.lef",
      "/home/taosimin/T28/lef/ts5n28hpcplvta64x96m2fw_130a.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140opphvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140hvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140oppuhvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140cguhvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140opp.lef",
      "/home/taosimin/T28/lef/ts1n28hpcplvtb512x64m4sw_180a.lef",
      "/home/taosimin/T28/lef/ts6n28hpcplvta2048x32m8sw_130a.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140opp.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140oppulvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140ehvt.lef",
      "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140opplvt.lef",
      "/home/taosimin/T28/lef/ts1n28hpcplvtb8192x64m8sw_180a.lef"};

  std::string def_file = "/home/taosimin/ir_example/aes/aes.def";
  timing_engine->readDefDesign(def_file, lef_files);

  timing_engine->readSdc("/home/taosimin/ir_example/aes/aes.sdc");

  timing_engine->readSpef("/home/taosimin/ir_example/aes/aes.spef");

  timing_engine->buildGraph();

  timing_engine->get_ista()->updateTiming();
  timing_engine->reportTiming();

  Sta* ista = Sta::getOrCreateSta();
  Power* ipower = Power::getOrCreatePower(&(ista->get_graph()));

  ipower->runCompleteFlow();

  PowerEngine* power_engine = PowerEngine::getOrCreatePowerEngine();

  std::string power_net_name = "VDD";

  // estimate rc from topo.
  // power_engine->buildPGNetWireTopo();

  // or read pg spef to calc rc.
  const char* pg_spef_file_path = "/home/taosimin/ir_example/aes/aes_vdd_vss.spef";
  power_engine->readPGSpef(pg_spef_file_path);

  power_engine->runIRAnalysis(power_net_name);

  // LOG_INFO << "rerun IR analysis";

  // rerun IR test.
  // power_engine->resetIRAnalysisData();
  // power_engine->buildPGNetWireTopo();
  // power_engine->runIRAnalysis(power_net_name);

  // for display power map and IR drop map.
  // power_engine->displayPowerMap();
  // power_engine->displayIRDropMap();

  power_engine->reportIRAnalysis();
}

}  // namespace