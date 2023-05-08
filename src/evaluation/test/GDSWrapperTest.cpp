#include <string>

#include "Config.hpp"
#include "EvalLog.hpp"
#include "gtest/gtest.h"
#include "manager.hpp"
#include "usage/usage.hh"

namespace eval {

class GDSWrapperTest : public testing::Test
{
  void SetUp()
  {
    char config[] = "json_test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() final { Log::end(); }
};

TEST_F(GDSWrapperTest, sample)
{
  std::string json_file = "/home/yhqiu/irefactor/src/platform/evaluator/source/config/ysyx_eval.json";
  Config* config = Config::getOrCreateConfig(json_file);

  ieda::Stats eval_status;
  Manager::initInst(config);

  auto gds_wrapper = Manager::getInst().getGDSWrapper();
  auto gds_net_list = gds_wrapper->get_net_list();
  for (size_t i = 0; i < gds_net_list.size(); ++i) {
    std::string name = gds_net_list[i]->get_name();
    LOG_INFO << "net name " << name;
  }

  Manager::destroyInst();

  double memory_delta = eval_status.memoryDelta();
  LOG_INFO << "Evaluator memory usage " << memory_delta << "MB";
  double time_delta = eval_status.elapsedRunTime();
  LOG_INFO << "Evaluator time elapsed " << time_delta << "s";
}

}  // namespace eval
