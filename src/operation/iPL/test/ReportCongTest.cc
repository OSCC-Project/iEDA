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
 * @FilePath: /irefactor/src/operation/iPL/test/ReportCongTest.cc
 * @Description:
 */

#include <string>

#include "PlacerDB.hh"
#include "gtest/gtest.h"
#include "PLAPI.hh"
#include "idm.h"

namespace ipl {
class ReportCongTestInterface : public testing::Test
{
  void SetUp()
  {  // Read Def, Lef
    std::string idb_json_file = "<local_path>/db_default_config.json";
    dmInst->init(idb_json_file);
  }
  void TearDown() final {}
};

TEST_F(ReportCongTestInterface, run_report_cong)
{
  std::string pl_json_file = "<local_path>/pl_default_config.json";
  auto* idb_builder = dmInst->get_idb_builder();
  iPLAPIInst.initAPI(pl_json_file, idb_builder);
  iPLAPIInst.runGP();
  iPLAPIInst.writeBackSourceDataBase();
  // iPLAPIInst.reportCongestionInfo();
  iPLAPIInst.destoryInst();
}

}  // namespace ipl
