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

// #include <gperftools/heap-profiler.h>

#include "AI-inference/AISta.hh"
#include "api/TimingEngine.hh"
#include "gtest/gtest.h"
#include "log/Log.hh"

using ieda::Log;

using namespace ista;

namespace {

class AIModelTest : public testing::Test {
  void SetUp() {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() { Log::end(); }
};

TEST_F(AIModelTest, calibration) {
  TimingEngine* timing_engine = TimingEngine::getOrCreateTimingEngine();
  timing_engine->set_num_threads(1);

  const char* design_work_space = "/home/taosimin/skywater130/rpt";
  timing_engine->set_design_work_space(design_work_space);

  std::vector<const char*> lib_files = {
      "/home/taosimin/skywater130/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"};

  const char* verilog_file =
      "/home/taosimin/skywater130/design/aes_cipher_top.v";
  const char* sdc_file = "/home/taosimin/skywater130/design/aes_cipher_top.sdc";
  const char* spef_file = "/home/taosimin/skywater130/spef/aes_cipher_top.spef";

  timing_engine->get_ista()->set_top_module_name("aes_cipher_top");
  timing_engine->readLiberty(lib_files);
  timing_engine->readDesign(verilog_file);
  timing_engine->readSdc(sdc_file);

  timing_engine->buildGraph();
  timing_engine->buildRCTree(spef_file, DelayCalcMethod::kElmore);

  timing_engine->updateTiming();
  timing_engine->reportTiming();

  auto* worst_path = timing_engine->get_ista()->getWorstSeqData(
     AnalysisMode::kMax, TransType::kRiseFall);

  std::map<AICalibratePathDelay::AIModeType, std::string> model_to_path{
      {AICalibratePathDelay::AIModeType::kSky130CalibratePathDelay,
       "/home/taosimin/iEDA24/iEDA/src/operation/iSTA/source/data/model/"
       "calibration/skywater130/test_model.onnx"}};
  std::string cell_list_path =
      "/home/taosimin/iEDA24/iEDA/src/operation/iSTA/source/data/model/"
      "calibration/skywater130/sky130_cells.txt";
  std::string pin_list_path =
      "/home/taosimin/iEDA24/iEDA/src/operation/iSTA/source/data/model/"
      "calibration/skywater130/sky130_pins.txt";
  AICalibratePathDelay path_calibration(
      std::move(model_to_path),
      AICalibratePathDelay::AIModeType::kSky130CalibratePathDelay,
      std::move(cell_list_path), std::move(pin_list_path));
  path_calibration.init();

  auto input_tensor = path_calibration.createInputTensor(worst_path);
  auto output_tensor = path_calibration.infer(input_tensor);
  auto output_value = path_calibration.getOutputResult(output_tensor);

  for(auto value : output_value) {
    LOG_INFO << "output result: " << value;
  }

  worst_path->get_delay_data()->set_calibrated_derate(output_value[0]);
  timing_engine->reportTiming();
}

}  // namespace