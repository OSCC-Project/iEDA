#include <string>

#include "Config.hpp"
#include "EvalLog.hpp"
#include "gtest/gtest.h"
#include "manager.hpp"
#include "usage/usage.hh"

namespace eval {

class WirelengthTest : public testing::Test
{
  void SetUp()
  {
    char config[] = "json_test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() final { Log::end(); }
};

TEST_F(WirelengthTest, sample)
{
  std::string json_file = "/home/yhqiu/irefactor/src/platform/evaluator/source/config/ysyx_eval_t28.json";
  Config* config = Config::getOrCreateConfig(json_file);

  ieda::Stats eval_status;
  Manager::initInst(config);

  auto wirelength_evaluator = Manager::getInst().getWirelengthEval();
  std::string plot_path = config->get_wl_config().get_output_dir();
  std::string output_file_name = config->get_wl_config().get_output_filename();
  wirelength_evaluator->reportWirelength(plot_path, output_file_name);

  // int64_t hpwl = wirelength_evaluator->evalTotalWL("kHPWL");
  // LOG_INFO << "hpwl = " << hpwl;
  // int64_t wl_test = wirelength_evaluator->evalTotalWL(config->get_wl_config().get_eval_type());
  // LOG_INFO << config->get_wl_config().get_eval_type() << " = " << wl_test;

  Manager::destroyInst();

  double memory_delta = eval_status.memoryDelta();
  LOG_INFO << "Evaluator memory usage " << memory_delta << "MB";
  double time_delta = eval_status.elapsedRunTime();
  LOG_INFO << "Evaluator time elapsed " << time_delta << "s";
}

}  // namespace eval
