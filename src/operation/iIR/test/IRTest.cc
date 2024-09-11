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
#include "api/iIR.hh"
#include "gtest/gtest.h"
#include "log/Log.hh"
#include "string/Str.hh"

using ieda::Log;

using namespace iir;

namespace {

class IRTest : public testing::Test {
  void SetUp() {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() { Log::end(); }
};

TEST_F(IRTest, ir_api) {
  const char* spef_file_path =
      "/home/taosimin/T28/spef/asic_top.spef_vdd_vss_1212.rcworst.0c.spef";
  const char* instance_power_path =
      "/home/shaozheqing/iEDA/bin/report_instance.csv";

  iIR ir_analysis;
  ir_analysis.readInstancePowerDB(instance_power_path);
  ir_analysis.readSpef(spef_file_path);
  ir_analysis.solveIRDrop("VDD");
}

TEST_F(IRTest, ir_small) {
  const char* spef_file_path = "/home/taosimin/ir_example/aes/aes_vdd_vss.spef";
  const char* instance_power_path =
      "/home/taosimin/ir_example/aes/aes_instance.csv";

  iIR ir_analysis;
  ir_analysis.init();
  ir_analysis.readInstancePowerDB(instance_power_path);
  ir_analysis.readSpef(spef_file_path);
  ir_analysis.solveIRDrop("VDD");
}

}  // namespace
