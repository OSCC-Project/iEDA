#include "gtest/gtest.h"
#include "log/Log.hh"
#include "verilog/VerilogReader.hh"

using namespace ista;

namespace {

class VerilogParserTest : public testing::Test {
  void SetUp() {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() { Log::end(); }
};

TEST_F(VerilogParserTest, read) {
  VerilogReader verilog_reader;

  verilog_reader.read("/home/smtao/peda/iEDA/src/iSTA/example/simple/simple.v");
  EXPECT_TRUE(true);
}

TEST_F(VerilogParserTest, readNutshell) {
  VerilogReader verilog_reader;

  verilog_reader.read("/home/smtao/peda/iEDA/src/iSTA/example/simple/simple.v");
  EXPECT_TRUE(true);
}

TEST_F(VerilogParserTest, flatten) {
  VerilogReader verilog_reader;

  bool is_ok = verilog_reader.read("/home/taosimin/ysyx/asic_top.v");
  verilog_reader.flattenModule("asic_top");
  EXPECT_TRUE(is_ok);
}

}  // namespace
