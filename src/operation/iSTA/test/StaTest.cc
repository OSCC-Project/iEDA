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
#include <iostream>

#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "gtest/gtest.h"
#include "liberty/Lib.hh"
#include "log/Log.hh"
#include "netlist/Netlist.hh"
#include "sdc-cmd/Cmd.hh"
#include "sta/Sta.hh"
#include "sta/StaAnalyze.hh"
#include "sta/StaApplySdc.hh"
#include "sta/StaBuildGraph.hh"
#include "sta/StaBuildRCTree.hh"
#include "sta/StaClockPropagation.hh"
#include "sta/StaDataPropagation.hh"
#include "sta/StaDelayPropagation.hh"
#include "sta/StaDump.hh"
#include "sta/StaGraph.hh"
#include "sta/StaSlewPropagation.hh"
#include "tcl/ScriptEngine.hh"
#include "usage/usage.hh"

using namespace ista;
using namespace ieda;

namespace {

class StaTest : public testing::Test {
  void SetUp() {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() { Log::end(); }
};

TEST_F(StaTest, simple_design) {
  Sta* ista = Sta::getOrCreateSta();
  if (ista) {
    LOG_INFO << "Start STA"
             << "\n";

    ista->set_num_threads(4);

    const char* design_work_space =
        "/home/taosimin/i-eda/src/iSTA/example/simple";
    ista->set_design_work_space(design_work_space);

    const char* verilog_file_name =
        "/home/taosimin/i-eda/src/iSTA/example/simple/writer_simple.v";

    std::string lib_name =
        Str::printf("%s/%s", design_work_space, "osu018_stdcells.lib");

    ista->readLiberty(lib_name.c_str());

    ista->set_top_module_name("simple");
    ista->readVerilogWithRustParser(
        "/home/taosimin/i-eda/src/iSTA/example/simple/simple.v");
    std::set<std::string> exclude_cell_names = {};
    ista->writeVerilog(verilog_file_name, exclude_cell_names);

    // ista->initScriptEngine();
    LOG_INFO << "read sdc start";
    Sta::initSdcCmd();

    auto* script_engine = ScriptEngine::getOrCreateInstance();

    unsigned result = script_engine->evalString(
        Str::printf("source %s/%s", design_work_space, "simple.sdc"));

    LOG_FATAL_IF(result == 1)
        << ScriptEngine::getOrCreateInstance()->evalString(
               R"(puts $errorInfo)");

    LOG_INFO << "read sdc end";

    ista->setReportSpec({"inp1"}, {{"u1:Y"}, {"u4:A"}}, {"f1:D"});

    ista->buildGraph();
    ista->updateTiming();
    ista->reportTiming();

    LOG_INFO << "The ista run success.";
  }
}

TEST_F(StaTest, read_error_file) {
  Sta* ista = Sta::getOrCreateSta();
  if (ista) {
    ista->readDesignWithRustParser("1.txt");
    ista->readLiberty("1.txt");
    ista->readSdc("1.txt");
    ista->readSpef("1.txt");
    ista->readAocv("1.txt");
    std::vector<std::string> error_files{"1.txt", "2.txt"};
    ista->readLiberty(error_files);
    ista->readAocv(error_files);
  }
}

}  // namespace