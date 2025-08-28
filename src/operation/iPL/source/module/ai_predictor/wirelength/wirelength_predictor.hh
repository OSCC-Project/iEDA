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
#pragma once
#include <string>
#include <memory>
#include <vector>

#include "onnx_model_handler.hh"
#include "normalization_handler.hh"

namespace ipl {

class WirelengthPredictor
{
 public:
  WirelengthPredictor();
  ~WirelengthPredictor() {}
  
  // Load wirelength prediction model
  bool loadModel(const std::string& model_path);

  // Predict wirelength for a net based on its features
  float predictWirelength(const std::vector<float>& features);

  // Predict via count for a net based on its features
  float predictViaCount(int net_id, const std::vector<float>& features);

  // Load normalization parameters for via prediction
  bool loadViaNormalizationParams(const std::string& params_path);

  // Load normalization parameters for wirelength prediction
  bool loadWirelengthNormalizationParams(const std::string& params_path);

  // Get required feature count for the model
  int getRequiredFeatureCount() const;

  // Check if model is loaded
  bool isModelLoaded() const;

 private:
  std::unique_ptr<ONNXModelHandler> _model_handler;
  std::unique_ptr<NormalizationHandler> _via_normalizer;
  std::unique_ptr<NormalizationHandler> _wirelength_normalizer;
  bool _is_wirelength_model; 

  // Normalize features based on prediction type
  std::vector<float> normalizeFeatures(const std::vector<float>& features, bool is_wirelength);
};

}  // namespace ipl