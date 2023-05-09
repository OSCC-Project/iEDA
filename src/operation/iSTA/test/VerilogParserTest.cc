// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#include "gtest/gtest.h"
#include "log/Log.hh"
#include "verilog/VerilogReader.hh"

using namespace ista;

namespace {

class VerilogParserTest : public testing::Test {
  void SetUp() {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() { Log::end(); }
};

TEST_F(VerilogParserTest, read) {
  VerilogReader verilog_reader;

  verilog_reader.read("/home/smtao/peda/iEDA/src/iSTA/example/simple/simple.v");
  EXPECT_TRUE(true);
}

TEST_F(VerilogParserTest, readNutshell) {
  VerilogReader verilog_reader;

  verilog_reader.read("/home/smtao/peda/iEDA/src/iSTA/example/simple/simple.v");
  EXPECT_TRUE(true);
}

TEST_F(VerilogParserTest, flatten) {
  VerilogReader verilog_reader;

  bool is_ok = verilog_reader.read("/home/taosimin/ysyx/asic_top.v");
  verilog_reader.flattenModule("asic_top");
  EXPECT_TRUE(is_ok);
}

}  // namespace
