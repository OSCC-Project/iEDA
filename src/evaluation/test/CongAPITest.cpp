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
#include "idm.h"
#include "manager.hpp"
#include "usage/usage.hh"

namespace eval {

class CongAPITest : public testing::Test
{
  void SetUp()
  {  // Read Def, Lef
    std::string idb_json_file = "/home/yhqiu/iEDA/bin/db_ispd15.json";
    dmInst->init(idb_json_file);
  }
  void TearDown() final {}
};

TEST_F(CongAPITest, sample)
{
  EvalAPI& eval_api = EvalAPI::initInst();

  int32_t bin_cnt_x = 256;
  int32_t bin_cnt_y = 256;
  auto inst_status = INSTANCE_STATUS::kFixed;

  eval_api.initCongDataFromIDB(bin_cnt_x, bin_cnt_y);
  eval_api.evalInstDens(inst_status);
  eval_api.evalPinDens(inst_status);
  eval_api.evalNetDens(inst_status);

  std::string plot_path = "/home/yhqiu/i-circuit-net/ispd2015/eval_routability_features/";
  eval_api.plotBinValue(plot_path, "macro_density", CONGESTION_TYPE::kInstDens);
  eval_api.plotBinValue(plot_path, "macro_pin_density", CONGESTION_TYPE::kPinDens);
  eval_api.plotBinValue(plot_path, "macro_net_density", CONGESTION_TYPE::kNetCong);

  EvalAPI::destroyInst();
}

}  // namespace eval
