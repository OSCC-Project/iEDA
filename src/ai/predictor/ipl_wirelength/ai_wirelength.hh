// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/*
 * @Description: AI-based wirelength evaluator
 * @FilePath: /iEDA/src/ai/predictor/ipl_wirelength/ai_wirelength.hh
 */

#ifndef IPL_EVALUATOR_AI_WIRELENGTH_H
#define IPL_EVALUATOR_AI_WIRELENGTH_H

#include <memory>

#include "Wirelength.hh"
#include "wirelength_predictor.hh"

#define aiPLWireLengthInst ipl::AIWirelength::getInstance()

namespace ipl {

class TopologyManager;

class AIWirelength
{
 public:
  static AIWirelength* getInstance()
  {
    if (!_instance) {
      _instance = new AIWirelength;
    }
    return _instance;
  }

  bool init(const std::string& model_path, const std::string& params_path, TopologyManager* topology_manager);

  // Load ONNX model for wirelength prediction
  bool loadModel(const std::string& model_path);
  bool loadNormalizationParams(const std::string& params_path);

  // Check if model is loaded
  bool isModelLoaded() const { return _is_model_loaded; }

  // Override virtual methods from Wirelength base class
  int64_t obtainTotalWirelength();
  int64_t obtainNetWirelength(int32_t net_id);
  int64_t obtainPartOfNetWirelength(int32_t net_id, int32_t sink_pin_id);

  // Extract features for a net
  std::vector<float> extractNetFeatures(int32_t net_id);

 private:
  static AIWirelength* _instance;
  TopologyManager* _topology_manager = nullptr;
  std::unique_ptr<WirelengthPredictor> _predictor = std::make_unique<WirelengthPredictor>();
  bool _is_model_loaded = false;

  AIWirelength() = default;
  ~AIWirelength() = default;
};

}  // namespace ipl

#endif