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
    std::string idb_json_file = "/home/yhqiu/irefactor/bin/db_default_config.json";
    dmInst->init(idb_json_file);
  }
  void TearDown() final {}
};

TEST_F(CongAPITest, sample)
{
  EvalAPI& eval_api = EvalAPI::initInst();
  eval_api.initCongestionEval();

  std::vector<float> gr_congestion;
  gr_congestion = eval_api.evalGRCong();  // return <ACE,TOF,MOF>
  LOG_INFO << "ACE: " << gr_congestion[0];
  LOG_INFO << "TOF: " << gr_congestion[1];
  LOG_INFO << "MOF: " << gr_congestion[2];

  std::string plot_path = "/home/yhqiu/irefactor/bin/";
  std::string output_file_name = "CongestionMap_";
  eval_api.plotGRCong(plot_path, output_file_name);
  eval_api.plotOverflow(plot_path, output_file_name);

  EvalAPI::destroyInst();
}

}  // namespace eval
