#include "gtest/gtest.h"
#include "log/Log.hh"
#include "ops/read_vcd/VCDParserWrapper.hh"

using namespace ipower;
using namespace ieda;

namespace {

class VCDParserWrapperTest : public testing::Test {
  void SetUp() final {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() final { Log::end(); }
};

TEST_F(VCDParserWrapperTest, read_vcd) {
  VcdParserWrapper vcd_parser_wrapper;
  vcd_parser_wrapper.readVCD(
      "/home/shaozheqing/edacontest-power/benchmark/UT/case0/test.vcd");
  std::string top_instance_name = "top_i";
  vcd_parser_wrapper.buildAnnotateDB(top_instance_name);
}

TEST_F(VCDParserWrapperTest, build_db) {
  VcdParserWrapper vcd_parser_wrapper;
  vcd_parser_wrapper.readVCD(
      "/home/shaozheqing/edacontest-power/benchmark/UT/case0/test.vcd");
  std::string top_instance_name = "top_i";
  vcd_parser_wrapper.buildAnnotateDB(top_instance_name);
}

TEST_F(VCDParserWrapperTest, calc_tc_sp) {
  VcdParserWrapper vcd_parser_wrapper;
  vcd_parser_wrapper.readVCD(
      "/home/shaozheqing/edacontest-power/benchmark/UT/case0/test.vcd");
  std::string top_instance_name = "top_i";
  vcd_parser_wrapper.buildAnnotateDB(top_instance_name);
  vcd_parser_wrapper.calcScopeToggleAndSp();
  vcd_parser_wrapper.printAnnotateDB(std::cout);
}

TEST_F(VCDParserWrapperTest, print_saif) {
  VcdParserWrapper vcd_parser_wrapper;
  vcd_parser_wrapper.readVCD(
      "/home/shaozheqing/edacontest-power/benchmark/UT/case0/test.vcd");
  std::string top_instance_name = "top_i";
  vcd_parser_wrapper.buildAnnotateDB(top_instance_name);
  vcd_parser_wrapper.calcScopeToggleAndSp();

  std::ofstream out_file;
  out_file.open("/home/shaozheqing/iSO/src/iPower/test/file.txt",
                std::ios::out | std::ios::trunc);

  vcd_parser_wrapper.printAnnotateDB(std::cout);
  out_file.close();
}

}  // namespace