#include "api/TimingDBAdapter.hh"
#include "api/TimingEngine.hh"
#include "gtest/gtest.h"
#include "tcl/ScriptEngine.hh"

using namespace ista;

namespace {

class OdbToNetlistTest : public testing::Test {
  void SetUp() {
    char config[] = "test";
    char *argv[] = {config};
    Log::init(argv);
  }
  void TearDown() { Log::end(); }
};

TEST_F(OdbToNetlistTest, odb) {}

TEST_F(OdbToNetlistTest, odb_nutshell) {}

TEST_F(OdbToNetlistTest, odb_ysyx) {}

TEST_F(OdbToNetlistTest, odb_ysyx1111) {}

}  // namespace
