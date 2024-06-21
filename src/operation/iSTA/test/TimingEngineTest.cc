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

using namespace ista;

namespace {

class TimingEngineTest : public testing::Test {
  void SetUp() {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() { Log::end(); }
};

TEST_F(TimingEngineTest, my_test) {
  bool have_sub_module;
  do {
    have_sub_module = false;
    for (int i = 0; i <= 5; i++) {
      //   have_sub_module = true;
      LOG_INFO << i << "\n";
      break;
    }
  } while (have_sub_module);
}

TEST_F(TimingEngineTest, resizer) {
  TimingEngine* timing_engine = TimingEngine::getOrCreateTimingEngine();
  if (timing_engine) {
    LOG_INFO << "Start STA"
             << "\n";
    timing_engine->set_num_threads(1);

    const char* design_work_space =
        "/home/longshuaiying/iEDA/src/iSTA/example/sizer/";
    timing_engine->set_design_work_space(design_work_space);

    std::vector<const char*> lib_files = {"NangateOpenCellLibrary_fast.lib"};

    const char* verilog_file = "sizer.v";
    const char* sdc_file = "sizer.sdc";
    const char* spef_file = "sizer.spef";

    DelayCalcMethod kmethod = DelayCalcMethod::kElmore;
    ModeTransPair mode_trans = {AnalysisMode::kMax, TransType::kRise};

    const double wl = 1.2;
    const double ucap = 3;

    timing_engine->readLiberty(lib_files);
    timing_engine->readDesign(verilog_file);
    timing_engine->readSdc(sdc_file);

    timing_engine->buildGraph();
    timing_engine->resetGraphData();
    timing_engine->resetPathData();

    timing_engine->setNetDelay(wl, ucap, "net_1", "inst_3:A2", mode_trans);
    timing_engine->buildRCTree(spef_file, kmethod);
    timing_engine->updateTiming();
    timing_engine->reportTiming();

    // report the timing//power/area
    double inst_delay =
        timing_engine->getInstDelay("inst_3", "inst_3:A2", "inst_3:ZN",
                                    AnalysisMode::kMax, TransType::kRise);

    // timing_engine->setNetDelay(wl, ucap, "net_1", "inst_3:A2", mode_trans);

    double net_delay = timing_engine->getNetDelay(
        "net_1", "inst_3:A2", AnalysisMode::kMax, TransType::kRise);
    double slew = timing_engine->getSlew("inst_3:A2", AnalysisMode::kMax,
                                         TransType::kRise);
    auto arrive_time =
        timing_engine->getAT("inst_3:A2", AnalysisMode::kMax, TransType::kFall);
    auto req_time =
        timing_engine->getRT("inst_3:A2", AnalysisMode::kMax, TransType::kFall);
    auto slack = timing_engine->getSlack("inst_3:A2", AnalysisMode::kMax,
                                         TransType::kRise);
    StaVertex* worst_vertex = nullptr;
    std::optional<double> worst_slack;
    timing_engine->getWorstSlack(AnalysisMode::kMax, TransType::kRise,
                                 worst_vertex, worst_slack);

    double WNS = timing_engine->getWNS("clk", AnalysisMode::kMax);
    double TNS = timing_engine->getTNS("clk", AnalysisMode::kMax);
    double network_latency = timing_engine->getClockNetworkLatency(
        "inst_7:CK", AnalysisMode::kMax, TransType::kRise);
    // double skew= timing_engine->reportClockSkew("inst_6:CK","inst_x:CK",
    // AnalysisMode::kMax, TransType::kRise);

    StaVertex* vertex = timing_engine->findVertex("inst_6:D");
    StaSeqPathData* seq_path_data = timing_engine->getWorstSeqData(
        vertex, AnalysisMode::kMax, TransType::kRise);
    int path_slack = seq_path_data->getSlack();

    unsigned is_clock = timing_engine->isClock("inst_6:CK");
    unsigned is_load = timing_engine->isLoad("inst_5:A1");

    double check_slew = 0;
    std::optional<double> slew_limit;
    double slew_slack = 0;
    timing_engine->validateSlew("inst_3:A2", AnalysisMode::kMax,
                                TransType::kRise, check_slew, slew_limit,
                                slew_slack);
    double check_fanout = 0;
    std::optional<double> fanout_limit;
    double fanout_slack = 0;
    timing_engine->validateFanout("inst_3:A2", AnalysisMode::kMax, check_fanout,
                                  fanout_limit, fanout_slack);

    auto& all_libs = timing_engine->getAllLib();
    std::vector<LibLibrary*> equiv_libs;
    for (auto& lib : all_libs) {
      equiv_libs.push_back(lib.get());
    }

    timing_engine->makeClassifiedCells(equiv_libs);
    auto* liberty_cell = timing_engine->findLibertyCell("NAND2_X1");
    auto* equiv_cells = timing_engine->classifyCells(liberty_cell);
    auto num_equiv_cells = equiv_cells->size();
    std::map<std::pair<StaVertex*, StaVertex*>, double> twosinks2worstslack =
        timing_engine->getWorstSlackBetweenTwoSinks(AnalysisMode::kMin);
    for (auto* equiv_cell : *equiv_cells) {
      LOG_INFO << equiv_cell->get_cell_name();
    }

    LOG_INFO << "\nBefore sizing\n"
             << "instance delay         : " << inst_delay << "     ns" << '\n'
             << "net delay              : " << net_delay << "    ns" << '\n'
             << "slew                   : " << slew << "     ns" << '\n'
             << "arrive time            : " << *arrive_time << "     ns" << '\n'
             << "required time          : " << *req_time << "     ns" << '\n'
             << "slack                  : " << *slack << "    ns" << '\n'
             << "worst slack            : " << *worst_slack << "    ns" << '\n'
             << "WNS                    : " << WNS << "    ns" << '\n'
             << "TNS                    : " << TNS << "    ns" << '\n'
             << "network latency        : " << network_latency << "    ns"
             << '\n'
             << "path slack             : " << path_slack << "   fs" << '\n'
             << "is clock               : " << is_clock << '\n'
             << "is load                : " << is_load << '\n'
             << "check slew             : " << check_slew << "     ns" << '\n'
             << "slew limit             : " << *slew_limit << "     ns" << '\n'
             << "slew slack             : " << slew_slack << "    ns" << '\n'
             << "check fanout           : " << check_fanout << '\n'
             << "fanout limit           : " << *fanout_limit << '\n'
             << "fanout slack           : " << fanout_slack << '\n'
             << "num equiv cells        : " << num_equiv_cells << '\n';

    // upsize inst_3 from X2 to X4
    timing_engine->repowerInstance("inst_3", "NAND2_X4");

    timing_engine->buildRCTree(spef_file, kmethod);
    timing_engine->updateTiming();
    timing_engine->reportTiming();

    inst_delay =
        timing_engine->getInstDelay("inst_3", "inst_3:A2", "inst_3:ZN",
                                    AnalysisMode::kMax, TransType::kRise);
    net_delay = timing_engine->getNetDelay(
        "net_1", "inst_3:A2", AnalysisMode::kMax, TransType::kRise);
    slew = timing_engine->getSlew("inst_3:A2", AnalysisMode::kMax,
                                  TransType::kRise);
    arrive_time =
        timing_engine->getAT("inst_3:A2", AnalysisMode::kMax, TransType::kFall);
    req_time =
        timing_engine->getRT("inst_3:A2", AnalysisMode::kMax, TransType::kFall);
    slack = timing_engine->getSlack("inst_3:A2", AnalysisMode::kMax,
                                    TransType::kRise);
    WNS = timing_engine->getWNS("clk", AnalysisMode::kMax);
    TNS = timing_engine->getTNS("clk", AnalysisMode::kMax);
    network_latency = timing_engine->getClockNetworkLatency(
        "inst_7:CK", AnalysisMode::kMax, TransType::kRise);

    seq_path_data = timing_engine->getWorstSeqData(
        vertex, AnalysisMode::kMax, TransType::kRise);
    path_slack = seq_path_data->getSlack();
    is_clock = timing_engine->isClock("inst_6:CK");

    LOG_INFO << "\nupsize inst_3 from X2 to X4\n"
             << "instance delay         : " << inst_delay << "      ns" << '\n'
             << "net delay              : " << net_delay << "    ns" << '\n'
             << "slew                   : " << slew << "     ns" << '\n'
             << "arrive time            : " << *arrive_time << "     ns" << '\n'
             << "required time          : " << *req_time << "     ns" << '\n'
             << "slack                  : " << *slack << "    ns" << '\n'
             << "WNS                    : " << WNS << "    ns" << '\n'
             << "TNS                    : " << TNS << "    ns" << '\n'
             << "network latency        : " << network_latency << "    ns"
             << '\n'
             << "path slack             : " << path_slack << "   fs" << '\n'
             << "is clock               : " << is_clock << '\n';
  }
}

TEST_F(TimingEngineTest, move_Instance) {
  TimingEngine* timing_engine = TimingEngine::getOrCreateTimingEngine();
  timing_engine->set_num_threads(1);

  const char* design_work_space =
      "/home/taosimin/iEDA-main/iEDA/src/iSTA/example/sizer/";
  timing_engine->set_design_work_space(design_work_space);

  std::vector<const char*> lib_files = {
      "/home/taosimin/iEDA-main/iEDA/src/iSTA/example/sizer/"
      "NangateOpenCellLibrary_fast.lib"};

  const char* verilog_file =
      "/home/taosimin/iEDA-main/iEDA/src/iSTA/example/sizer/sizer.v";
  const char* sdc_file =
      "/home/taosimin/iEDA-main/iEDA/src/iSTA/example/sizer/sizer.sdc";
  const char* spef_file =
      "/home/taosimin/iEDA-main/iEDA/src/iSTA/example/sizer/sizer.spef";

  DelayCalcMethod kmethod = DelayCalcMethod::kElmore;

  timing_engine->readLiberty(lib_files);
  timing_engine->readDesign(verilog_file);
  timing_engine->readSdc(sdc_file);

  timing_engine->buildGraph();
  timing_engine->resetGraphData();
  timing_engine->resetPathData();

  timing_engine->buildRCTree(spef_file, kmethod);

  // non-incremental updateTiming for comparing result.
  timing_engine->updateTiming();
  timing_engine->reportTiming();

  Netlist* design_netlist = timing_engine->get_netlist();

  Net* net_1 = design_netlist->findNet("net_1");
  Net* nx3 = design_netlist->findNet("nx3");
  Net* nx6 = design_netlist->findNet("nx6");

  auto* pin = design_netlist->findPin("inst_2:A2", false, false).front();
  auto* node1 = timing_engine->makeOrFindRCTreeNode(pin);
  timing_engine->incrCap(node1, 0.003);

  timing_engine->updateRCTreeInfo(net_1);
  timing_engine->updateRCTreeInfo(nx3);
  timing_engine->updateRCTreeInfo(nx6);

  timing_engine->moveInstance("inst_0", 1);

  timing_engine->incrUpdateTiming();
  timing_engine->reportTiming();
}

TEST_F(TimingEngineTest, equiv_lib) {
  TimingEngine* timing_engine = TimingEngine::getOrCreateTimingEngine();
  timing_engine->set_num_threads(1);

  std::vector<const char*> lib_files = {
      "/home/taosimin/nangate45/lib/NangateOpenCellLibrary_typical.lib"};

  timing_engine->readLiberty(lib_files);

  std::vector<LibLibrary*> equiv_libs;
  auto& all_libs = timing_engine->getAllLib();
  for (auto& lib : all_libs) {
    equiv_libs.push_back(lib.get());
  }

  timing_engine->makeClassifiedCells(equiv_libs);
}

}  // namespace
