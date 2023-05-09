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
#include "report/ReportTable.hh"


namespace eval {

class EGRWLTest : public testing::Test
{
  void SetUp()
  {  // Read Def, Lef
    std::string idb_json_file = "/home/yhqiu/irefactor/bin/db_default_config_t28.json";
    dmInst->init(idb_json_file);
  }
  void TearDown() final {}
};

TEST_F(EGRWLTest, sample)
{
  std::string json_file = "/home/yhqiu/irefactor/src/evaluation/source/config/ysyx_eval_t28.json";
  Config* config = Config::getOrCreateConfig(json_file);
  Manager::initInst(config);
  auto wirelength_evaluator = Manager::getInst().getWirelengthEval();
  auto& wl_net_list = wirelength_evaluator->get_net_list();

  EvalAPI& eval_api = EvalAPI::initInst();
  int64_t hpwl = eval_api.evalTotalWL("kHPWL",wl_net_list);
  int64_t htree = eval_api.evalTotalWL("kHTree",wl_net_list);
  int64_t vtree = eval_api.evalTotalWL("kVTree",wl_net_list);
  int64_t clique = eval_api.evalTotalWL("kClique",wl_net_list);
  int64_t star = eval_api.evalTotalWL("kStar",wl_net_list);
  int64_t B2B = eval_api.evalTotalWL("kB2B",wl_net_list);
  int64_t FLUTE = eval_api.evalTotalWL("kFlute",wl_net_list);
  double egr_wl = eval_api.evalEGRWL();

  LOG_INFO << "hpwl: " << hpwl;
  LOG_INFO << "htree: " << htree;
  LOG_INFO << "vtree: " << vtree;
  LOG_INFO << "clique: " << clique;
  LOG_INFO << "star: " << star;
  LOG_INFO << "B2B: " << B2B;
  LOG_INFO << "FLUTE: " << FLUTE;
  LOG_INFO << "eGR: " << egr_wl;

  auto report_tbl = std::make_unique<ieda::ReportTable>("table");
  (*report_tbl) << TABLE_HEAD;
  (*report_tbl)[0][0] = "Wirelength Info";
  (*report_tbl)[1][0] = "HPWL"; 
  (*report_tbl)[1][1] = std::to_string(hpwl);
  (*report_tbl)[2][0] = "B2B";
  (*report_tbl)[2][1] = std::to_string(B2B);
  (*report_tbl)[3][0] = "FLUTE";
  (*report_tbl)[3][1] = std::to_string(FLUTE);
  (*report_tbl)[4][0] = "earlyGR";
  (*report_tbl)[4][1] = std::to_string(int64_t(egr_wl));
  (*report_tbl) << TABLE_ENDLINE;
  std::cout << (*report_tbl).to_string() << std::endl;

  EvalAPI::destroyInst();
  Manager::destroyInst();
}

}  // namespace eval
