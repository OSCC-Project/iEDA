// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************

#if CUDA_PROPAGATION

#include "api/TimingEngine.hh"
#include "gtest/gtest.h"
#include "log/Log.hh"
#include "propagation-cuda/lib_arc.cuh"
using namespace ista;
using ieda::Log;
using ieda::Stats;

namespace {

class LibDataGPUTest : public testing::Test {
  void SetUp() {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() { Log::end(); }
};

TEST_F(LibDataGPUTest, test) {
  Stats stats;

  auto* timing_engine = TimingEngine::getOrCreateTimingEngine();
  timing_engine->set_num_threads(48);
  const char* design_work_space = "/home/longshuaiying/cuda_delay";
  timing_engine->set_design_work_space(design_work_space);

  std::vector<const char*> lib_files{
      "/home/taosimin/nangate45/lib/"
      "NangateOpenCellLibrary_typical.lib"};
  timing_engine->readLiberty(lib_files);

  timing_engine->get_ista()->set_analysis_mode(ista::AnalysisMode::kMaxMin);
  timing_engine->get_ista()->set_n_worst_path_per_clock(1);

  timing_engine->get_ista()->set_top_module_name("top");

  timing_engine->readDesign(
      "/home/taosimin/nangate45/design/example/example1.v");
  timing_engine->linkDesign("top");

  timing_engine->readSdc(
      "/home/taosimin/nangate45/design/example/example1.sdc");

  timing_engine->buildGraph();
  timing_engine->get_ista()->buildLibArcsGPU();
  auto& lib_arcs_gpu =
      timing_engine->get_ista()->get_lib_gpu_arcs();
  Lib_Data_GPU lib_data_gpu;
  std::vector<Lib_Table_GPU> lib_tables_gpu;
  std::vector<Lib_Table_GPU*> lib_gpu_table_ptrs;
  build_lib_data_gpu(lib_data_gpu, lib_tables_gpu, lib_gpu_table_ptrs, lib_arcs_gpu);
  find_value_test(lib_data_gpu, 0.0449324, 0.0449324);

  double memory_delta = stats.memoryDelta();
  LOG_INFO << "memory usage " << memory_delta << "MB";

  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "time elapsed " << time_delta << "s";
}

}  // namespace

#endif