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
#include "gtest/gtest.h"
#include "liberty/Lib.hh"
#include "log/Log.hh"
#include "sdc-cmd/Cmd.hh"
#include "sta/Sta.hh"
#include "sta/StaApplySdc.hh"
#include "sta/StaBuildGraph.hh"
#include "tcl/ScriptEngine.hh"

using namespace ista;

namespace {

class SdcTest : public testing::Test {
  void SetUp() {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);

    Sta* ista = Sta::getOrCreateSta();

    ista->readLiberty("/home/taosimin/nangate45/lib/example1_fast.lib");

    ista->readVerilogWithRustParser(
        "/home/taosimin/nangate45/design/example/example1.v");
    ista->linkDesignWithRustParser("top");

    ista->getConstrain();
  }
  void TearDown() {
    Sta::destroySta();
    Log::end();
  }
};

// TEST_F(SdcTest, swig_test) {
//   LOG_INFO << "sdc test1";
//   Sta* ista = Sta::getOrCreateSta();
//   ista->initScriptEngine();

//   ScriptEngine::getOrCreateInstance()->evalString(R"(puts "hello sdc")");
//   int result =
//   ScriptEngine::getOrCreateInstance()->evalString(R"(test_sdc)");

//   EXPECT_TRUE(!result);

//   result = ScriptEngine::getOrCreateInstance()->evalString(
//       R"(create_clock -name clk -period 10 clk)");

//   LOG_FATAL_IF(result != 0)
//       << ScriptEngine::getOrCreateInstance()->evalString(R"(puts
//       $errorInfo)");

//   LOG_INFO << "sdc test1 finish";
// }

TEST_F(SdcTest, callback_test) {
  LOG_INFO << "sdc test2";

  int result = 0;

  CmdCreateClock* cmd_create_clock = new CmdCreateClock("create_clock");
  LOG_FATAL_IF(!cmd_create_clock);

  result = ScriptEngine::getOrCreateInstance()->evalString(
      R"(create_clock -name clk -period 10 clk)");

  LOG_FATAL_IF(result != 0)
      << ScriptEngine::getOrCreateInstance()->evalString(R"(puts $errorInfo)");

  LOG_INFO << "sdc test2 finish";
}

// string create_generated_clock
//    [-name clock_name]
//    -source master_pin
//    [-divide_by divide_factor | -multiply_by multiply_factor |
//     -edges edge_list ]
//    [-duty_cycle percent]
//    [-invert]
//    [-edge_shift edge_shift_list]
//    [-add]
//    [-master_clock clock]
//    [-comment comment_string]
//    source_objects

TEST_F(SdcTest, generate_clock) {
  int result = 1;

  auto cmd_create_clock = std::make_unique<CmdCreateClock>("create_clock");
  TclCmds::addTclCmd(std::move(cmd_create_clock));

  auto cmd_create_generate_clock =
      std::make_unique<CmdCreateGeneratedClock>("create_generated_clock");

  TclCmds::addTclCmd(std::move(cmd_create_generate_clock));

  result = ScriptEngine::getOrCreateInstance()->evalString(
      R"(create_clock -name clk -period 2.2 clk1)");

  result &= ScriptEngine::getOrCreateInstance()->evalString(
      R"(create_generated_clock -name CLKdiv2 -divide_by 2 -source clk clk2)");

  result &= ScriptEngine::getOrCreateInstance()->evalString(
      R"(create_generated_clock -name CLKdiv3 -divide_by 2 -invert -source clk clk2)");

  result &= ScriptEngine::getOrCreateInstance()->evalString(
      R"(create_generated_clock -name CLKdiv4 -multiply_by 2 -source clk clk2)");

  result &= ScriptEngine::getOrCreateInstance()->evalString(
      R"(create_generated_clock -name CLKdiv5 -edges { 3 5 9 } -source clk clk2)");

  result &= ScriptEngine::getOrCreateInstance()->evalString(
      R"(create_generated_clock -name CLKdiv6 -edges { 3 5 9 } -edge_shift {2.2 2.2 2.2} -source clk clk2)");

  LOG_FATAL_IF(result != 0)
      << ScriptEngine::getOrCreateInstance()->evalString(R"(puts $errorInfo)");
}

TEST_F(SdcTest, set_max_fanout) {
  Sta* ista = Sta::getOrCreateSta();
  ista->initSdcCmd();

  int result = 1;
  result = ScriptEngine::getOrCreateInstance()->evalString(
      R"(set_max_fanout 5.0 [current_design])");
  result &= ScriptEngine::getOrCreateInstance()->evalString(
      R"(set_max_transition 5.0 [current_design])");
  result &= ScriptEngine::getOrCreateInstance()->evalString(
      R"(set_max_capacitance 5.0 [current_design])");

  StaGraph& the_graph = ista->get_graph();
  Vector<std::function<unsigned(StaGraph*)>> funcs = {
      StaBuildGraph(), StaApplySdc(StaApplySdc::PropType::kApplySdcPreProp)};

  for (auto& func : funcs) {
    the_graph.exec(func);
  }
}

TEST_F(SdcTest, set_propagated_clock) {
  Sta* ista = Sta::getOrCreateSta();
  ista->initSdcCmd();

  int result = 1;
  result = ScriptEngine::getOrCreateInstance()->evalString(
      R"(create_clock -name clk -period 2.2 clk1)");
  result = ScriptEngine::getOrCreateInstance()->evalString(
      R"(set_propagated_clock [all_clocks])");

  StaGraph& the_graph = ista->get_graph();
  Vector<std::function<unsigned(StaGraph*)>> funcs = {
      StaBuildGraph(), StaApplySdc(StaApplySdc::PropType::kApplySdcPreProp)};

  for (auto& func : funcs) {
    the_graph.exec(func);
  }

  EXPECT_TRUE(result);
}

TEST_F(SdcTest, get_ports) {
  Sta* ista = Sta::getOrCreateSta();
  ista->initSdcCmd();

  int result = 1;
  result = ScriptEngine::getOrCreateInstance()->evalString(
      R"(create_clock -name clk -period 2.2 [get_ports clk1])");

  EXPECT_TRUE(result);
}

TEST_F(SdcTest, set_input_delay) {
  Sta* ista = Sta::getOrCreateSta();
  ista->initSdcCmd();

  int result = 1;
  result = ScriptEngine::getOrCreateInstance()->evalString(
      R"(create_clock -name clk -period 2.2 [get_ports clk1])");

  result &= ScriptEngine::getOrCreateInstance()->evalString(
      R"(set_input_delay 1.0 -clock clk [get_ports clk1])");

  result &= ScriptEngine::getOrCreateInstance()->evalString(
      R"(set_input_delay 1.0 -clock [get_clocks clk] [get_ports clk1])");
}

TEST_F(SdcTest, set_load) {
  Sta* ista = Sta::getOrCreateSta();
  ista->initSdcCmd();

  int result = 1;
  result = ScriptEngine::getOrCreateInstance()->evalString(
      R"(create_clock -name clk -period 2.2 [get_ports clk1])");

  result &= ScriptEngine::getOrCreateInstance()->evalString(
      R"(set_load 8.0 [get_ports clk1])");
}

TEST_F(SdcTest, zx_test) {
  Sta* ista = Sta::getOrCreateSta();
  ista->initSdcCmd();

  int result = 1;
  result = ScriptEngine::getOrCreateInstance()->evalString(
      R"(create_clock -name clk -period 2.2 -add [get_ports clk1])");

  result &= ScriptEngine::getOrCreateInstance()->evalString(
      R"(set_false_path -to [get_ports clk1])");

  result &= ScriptEngine::getOrCreateInstance()->evalString(
      R"(set_max_delay 1.0 -from [get_ports clk1] -to [get_ports out])");

  result &= ScriptEngine::getOrCreateInstance()->evalString(
      R"(set_min_delay 1.0 -from [get_ports clk1] -to [get_ports out])");
}

}  // namespace