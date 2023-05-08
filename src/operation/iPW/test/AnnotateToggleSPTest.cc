#include <signal.h>

#include <cstdlib>
#include <fstream>

#include "VCDFileParser.hpp"
#include "api/Power.hh"
#include "api/TimingEngine.hh"
#include "gtest/gtest.h"
#include "include/PwrConfig.hh"
#include "log/Log.hh"
#include "ops/build_graph/PwrBuildGraph.hh"
#include "ops/read_vcd/VCDParserWrapper.hh"
#include "usage/usage.hh"

using namespace ipower;
using namespace ieda;
using ieda::Stats;

namespace {
class AnnotateToggleSPTest : public testing::Test {
  void SetUp() final {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() final { Log::end(); }
};

TEST_F(AnnotateToggleSPTest, calc_tc_sp) {
  Stats stats;

  auto* timing_engine = TimingEngine::getOrCreateTimingEngine();
  timing_engine->set_num_threads(48);
  const char* design_work_space = "";
  timing_engine->set_design_work_space(design_work_space);

  std::vector<const char*> t28_lib_files = {};

  timing_engine->readLiberty(t28_lib_files);

  timing_engine->get_ista()->set_analysis_mode(ista::AnalysisMode::kMaxMin);
  timing_engine->get_ista()->set_n_worst_path_per_clock(1);

  timing_engine->get_ista()->set_top_module_name("asic_top");
  timing_engine->readDesign("");

  timing_engine->readSdc("");

  timing_engine->buildGraph();

  Power ipower(&(timing_engine->get_ista()->get_graph()));
  ipower.buildGraph();

  // vcd parser
  VcdParserWrapper vcd_parser_wrapper;
  vcd_parser_wrapper.readVCD("/home/shaozheqing/benchmark/asic_top.vcd",
                             std::make_pair(0, 326034000000));
  std::string top_instance_name = "u0_asic_top";
  vcd_parser_wrapper.buildAnnotateDB(top_instance_name);
  vcd_parser_wrapper.calcScopeToggleAndSp();

  std::ofstream out_file;
  out_file.open("/home/shaozheqing/iSO/src/iPower/test/file.txt",
                std::ios::out | std::ios::trunc);

  vcd_parser_wrapper.printAnnotateDB(out_file);
  out_file.close();
  auto* annotate_db = vcd_parser_wrapper.get_annotate_db();
  ipower.annotateToggleSP(annotate_db);

  // timing_engine->updateTiming();

  double memory_delta = stats.memoryDelta();
  LOG_INFO << "memory usage " << memory_delta << "MB";

  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "time elapsed " << time_delta << "s";
}
}  // namespace