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
 * @FilePath: /iEDA/src/iPL/src/evaluator/wirelength/AIWirelength.hh
 */

#ifndef IPL_EVALUATOR_AI_WIRELENGTH_H
#define IPL_EVALUATOR_AI_WIRELENGTH_H

#include "Wirelength.hh"

#include <memory>

#include "wirelength_predictor.hh"


namespace ipl {

class AIWirelength : public Wirelength
{
 public:
  AIWirelength() = delete;
  explicit AIWirelength(TopologyManager* topology_manager);
  AIWirelength(const AIWirelength&) = delete;
  AIWirelength(AIWirelength&&) = delete;
  ~AIWirelength() override = default;

  AIWirelength& operator=(const AIWirelength&) = delete;
  AIWirelength& operator=(AIWirelength&&) = delete;

  // Load ONNX model for wirelength prediction
  bool loadModel(const std::string& model_path);
  bool loadNormalizationParams(const std::string& params_path);

  // Check if model is loaded
  bool isModelLoaded() const;

  // Override virtual methods from Wirelength base class
  int64_t obtainTotalWirelength() override;
  int64_t obtainNetWirelength(int32_t net_id) override;
  int64_t obtainPartOfNetWirelength(int32_t net_id, int32_t sink_pin_id) override;

  // Extract features for a net
  std::vector<float> extractNetFeatures(int32_t net_id);

 private:
  std::unique_ptr<WirelengthPredictor> _predictor;
  bool _is_model_loaded = false;
};

inline AIWirelength::AIWirelength(TopologyManager* topology_manager) : Wirelength(topology_manager),
                                                                       _predictor(std::make_unique<WirelengthPredictor>())
{
}

inline bool AIWirelength::isModelLoaded() const
{
  return _is_model_loaded;
}

}  // namespace ipl

#endif