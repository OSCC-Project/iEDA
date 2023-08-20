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
#include <string>

#include "Config.hpp"
#include "EvalAPI.hpp"
#include "EvalLog.hpp"
#include "gtest/gtest.h"
#include "manager.hpp"
#include "usage/usage.hh"

namespace eval {

class WirelengthAPITest : public testing::Test
{
  void SetUp()
  {  // Read Def, Lef
    std::string idb_json_file = "/home/yhqiu/iEDA/bin/db_ispd19.json";
    dmInst->init(idb_json_file);
  }
  void TearDown() final {}
};

TEST_F(WirelengthAPITest, sample)
{
  EvalAPI& eval_api = EvalAPI::initInst();
  eval_api.initWLDataFromIDB();
  LOG_INFO << "HPWL is " << eval_api.evalTotalWL("kHPWL");
  LOG_INFO << "B2B is " << eval_api.evalTotalWL("kB2B");
  LOG_INFO << "FLUTE is " << eval_api.evalTotalWL("kFlute");
  // LOG_INFO << "EGR is " << eval_api.evalEGRWL();
  LOG_INFO << "HPWL is " << eval_api.evalTotalWL(WIRELENGTH_TYPE::kHPWL);
  LOG_INFO << "B2B is " << eval_api.evalTotalWL(WIRELENGTH_TYPE::kB2B);
  LOG_INFO << "FLUTE is " << eval_api.evalTotalWL(WIRELENGTH_TYPE::kFLUTE);

  EvalAPI::destroyInst();
}

}  // namespace eval
