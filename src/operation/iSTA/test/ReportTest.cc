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
#include <iostream>
#include <string>

#include "gtest/gtest.h"
#include "libfort/fort.hpp"
#include "log/Log.hh"

using namespace ieda;

namespace {
class ReportTest : public testing::Test {
  void SetUp() final {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() final { Log::end(); }
};

TEST_F(ReportTest, report_table) {
  fort::char_table table;
  table << fort::header;
  /* Fill each cell with operator[] */
  table[0][0] = "Endpoint";
  table[0][1] = "Path Delay";
  table[0][2] = "Path Required";
  table[0][3] = "CPPR";
  table[0][4] = "Slack";
  table << fort::endr;

  /* Fill with iostream operator<< */
  table << "Endpoint1" << 1 << 47.362 << 340 << 1234 << fort::endr;

  /* Fill row explicitly with strings */
  table.write_ln("Endpoint2", "2", "35.02", "737", "4567");

  /* Fill row with data from the container */
  std::vector<std::string> arr = {"Endpoint3", "3", "35.02", "737", "9999"};
  table.range_write_ln(std::begin(arr), std::end(arr));

  std::cout << table.to_string() << std::endl;
}

}  // namespace
