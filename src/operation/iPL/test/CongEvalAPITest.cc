/*
 * @Author: sjchanson 13560469332@163.com
 * @Date: 2022-10-31 14:27:37
 * @LastEditors: sjchanson 13560469332@163.com
 * @LastEditTime: 2022-11-20 09:48:15
 * @FilePath: /irefactor/src/operation/iPL/test/CongEvalAPITest.cc
 * @Description:
 */

#include <string>

#include "PlacerDB.hh"
#include "gtest/gtest.h"
#include "iPL_API.hh"
#include "idm.h"

namespace ipl {
class CongEvalAPITest : public testing::Test
{
  void SetUp()
  {  // Read Def, Lef
    std::string idb_json_file = "<local_path>/db_default_config.json";
    dmInst->init(idb_json_file);
  }
  void TearDown() final {}
};

TEST_F(CongEvalAPITest, run_cong_api)
{
  // flow test : pass
  std::string pl_json_file = "<local_path>/pl_default_config_t110.json";
  auto* idb_builder = dmInst->get_idb_builder();
  iPLAPIInst.initAPI(pl_json_file, idb_builder);
  iPLAPIInst.runRoutabilityGP();
  iPLAPIInst.writeBackSourceDataBase();
  std::vector<float> gr_congestion = iPLAPIInst.evalGRCong();
  std::cout << "Final ACE: " << gr_congestion[0];
  std::cout << "Final TOF: " << gr_congestion[1];
  std::cout << "Final MOF: " << gr_congestion[2];
  iPLAPIInst.runDP();
  iPLAPIInst.runLG();
  std::string plot_path = "<local_path>";
  std::string output_file_name = "CongestionMap";
  iPLAPIInst.plotCongMap(plot_path, output_file_name);
  iPLAPIInst.destroyCongEval();
  iPLAPIInst.destoryInst();
}

}  // namespace ipl
