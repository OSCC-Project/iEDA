#include <string>

#include "Config.hpp"
#include "EvalAPI.hpp"
#include "EvalLog.hpp"
#include "gtest/gtest.h"
#include "manager.hpp"
#include "usage/usage.hh"

namespace eval {

class GDSAPITest : public testing::Test
{
  void SetUp()
  {
    char config[] = "json_test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() final { Log::end(); }
};

TEST_F(GDSAPITest, sample)
{
  std::string json_file = "/home/yhqiu/irefactor/src/platform/evaluator/source/config/ysyx_eval.json";

  EvalAPI& eval_api = EvalAPI::initInst();
  std::vector<GDSNet*> gds_net_list = eval_api.wrapGDSNetlist(json_file);
  LOG_INFO << "SIZE :" << gds_net_list.size();
  EvalAPI::destroyInst();
}

}  // namespace eval
