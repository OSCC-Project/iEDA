#include <string>

#include "Config.hpp"
#include "EvalLog.hpp"
#include "gtest/gtest.h"
#include "manager.hpp"
#include "usage/usage.hh"

namespace eval {

class CongestionTest : public testing::Test
{
  void SetUp()
  {
    char config[] = "json_test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() final { Log::end(); }
};

TEST_F(CongestionTest, sample)
{
  std::string json_file = "/home/yhqiu/irefactor/src/platform/evaluator/source/config/ysyx_eval_t28.json";
  Config* config = Config::getOrCreateConfig(json_file);

  ieda::Stats eval_status;
  Manager::initInst(config);

  auto congestion_evaluator = Manager::getInst().getCongestionEval();
  std::string plot_path = config->get_cong_config().get_output_dir();
  std::string output_file_name = config->get_cong_config().get_output_filename();
  congestion_evaluator->reportCongestion(plot_path, output_file_name);

  // congestion_evaluator->mapNet2Bin();
  // congestion_evaluator->evalNetCong(config->get_cong_config().get_eval_type());
  // LOG_INFO << "Bin(0,0): " << congestion_evaluator->getBinNetCong(0, 0);
  // LOG_INFO << "Bin(0,1): " << congestion_evaluator->getBinNetCong(0, 1);

  Manager::destroyInst();

  double memory_delta = eval_status.memoryDelta();
  LOG_INFO << "Eval memory usage " << memory_delta << "MB";
  double time_delta = eval_status.elapsedRunTime();
  LOG_INFO << "Eval time elapsed " << time_delta << "s";
}

}  // namespace eval
