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
/*
 * @Author: sjchanson 13560469332@163.com
 * @Date: 2022-10-31 14:27:37
 * @LastEditors: sjchanson 13560469332@163.com
 * @LastEditTime: 2022-11-20 09:48:15
 * @FilePath: /irefactor/src/operation/iPL/test/CongEvalAPITest.cc
 * @Description:
 */

#include <string>

#include "PLAPI.hh"
#include "PlacerDB.hh"
#include "gtest/gtest.h"
#include "idm.h"

namespace ipl {
class CongEvalAPITest : public testing::Test
{
  void SetUp()
  {  // Read Def, Lef
    std::string idb_json_file = "/home/yhqiu/iEDA/bin/db_default_config_ispd2019.json";
    dmInst->init(idb_json_file);
  }
  void TearDown() final {}
};

TEST_F(CongEvalAPITest, run_cong_api)
{
  // flow test : pass
  std::string pl_json_file = "/home/yhqiu/iEDA/bin/pl_default_config_ispd2019.json";
  auto* idb_builder = dmInst->get_idb_builder();
  iPLAPIInst.initAPI(pl_json_file, idb_builder);
  iPLAPIInst.runGP();
  iPLAPIInst.writeBackSourceDataBase();
  // std::vector<float> gr_congestion = iPLAPIInst.evalGRCong();
  // std::cout << "Final ACE: " << gr_congestion[0];
  // std::cout << "Final TOF: " << gr_congestion[1];
  // std::cout << "Final MOF: " << gr_congestion[2];
  // iPLAPIInst.runDP();
  // iPLAPIInst.runLG();
  // std::string plot_path = "<local_path>";
  // std::string output_file_name = "CongestionMap";
  // iPLAPIInst.plotCongMap(plot_path, output_file_name);
  // iPLAPIInst.destroyCongEval();
  iPLAPIInst.destoryInst();
}

}  // namespace ipl
