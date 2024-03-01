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
#include "ops/read_vcd/RustVCDParserWrapper.hh"

using namespace ipower;
using namespace ieda;

namespace {

class VCDParserWrapperTest : public testing::Test {
  void SetUp() final {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() final { Log::end(); }
};

TEST_F(VCDParserWrapperTest, rust_reader) {
  ipower::RustVcdParserWrapper vcd_reader;

  vcd_reader.readVcdFile(
      "/home/shaozheqing/iEDA/src/database/manager/parser/vcd/vcd_parser/"
      "benchmark/test1.vcd");

  vcd_reader.buildAnnotateDB("top_i");
  vcd_reader.calcScopeToggleAndSp("top_i");
  vcd_reader.printAnnotateDB(std::cout);
}

}  // namespace
