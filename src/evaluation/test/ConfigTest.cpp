#include <string>

#include "Config.hpp"
#include "EvalLog.hpp"
#include "gtest/gtest.h"

namespace eval {

class ConfigTest : public testing::Test
{
  void SetUp()
  {
    char config[] = "json_test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() final { Log::end(); }
};

TEST_F(ConfigTest, sample)
{
  std::string json_file = "/home/yhqiu/irefactor/src/platform/evaluator/source/config/ysyx_eval.json";
  Config* config = Config::getOrCreateConfig(json_file);
  config->printConfig();
}

}  // namespace eval
