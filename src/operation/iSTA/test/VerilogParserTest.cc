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
#include "log/Log.hh"
#include "sta/Sta.hh"

using namespace ieda;
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
  Sta* ista = Sta::getOrCreateSta();

  const char* verilog_file_name =
        "/home/taosimin/ysyx_test25/2025-05-27/test.v";
  unsigned ret = ista->readVerilogWithRustParser(verilog_file_name);

  LOG_INFO << "Read Verilog file: " << verilog_file_name
           << ", return code: " << ret;

}

}  // namespace
