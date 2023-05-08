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
#include "iPL_API.hh"
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
  iPLAPIInst.reportCongestionInfo();
  iPLAPIInst.destoryInst();
}

}  // namespace ipl
