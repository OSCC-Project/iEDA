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

#include "wirelength_predictor.hh"

#include <iostream>

namespace ipl {

WirelengthPredictor::WirelengthPredictor() : 
    _model_handler(std::make_unique<ONNXModelHandler>()),
    _via_normalizer(std::make_unique<NormalizationHandler>()),
    _wirelength_normalizer(std::make_unique<NormalizationHandler>()),
    _is_wirelength_model(false)
{
  std::cout << "Wirelength predictor initialized" << std::endl;
}

bool WirelengthPredictor::loadModel(const std::string& model_path)
{
    if (!_model_handler->loadModel(model_path)) {
        return false;
    }

    // Assume all models are wirelength models by default
    _is_wirelength_model = true; 

    std::cout << "Successfully loaded wirelength prediction model from " << model_path << std::endl;
    return true;
}

float WirelengthPredictor::predictWirelength(const std::vector<float>& features)
{
    if (!isModelLoaded()) {
        std::cerr << "Model not loaded" << std::endl;
        return -1.0f;
    }

    if (!_is_wirelength_model) {
        std::cerr << "Loaded model is not a wirelength prediction model" << std::endl;
        return -1.0f;
    }

    std::vector<float> normalized_features = normalizeFeatures(features, true);

    std::vector<float> output = _model_handler->predict(normalized_features);
    if (output.empty()) {
        std::cerr << "Prediction failed" << std::endl;
        return -1.0f;
    }

    float prediction = output[0];
    std::cout << "Net wirelength prediction: " << prediction << std::endl;
    return prediction;
}

float WirelengthPredictor::predictViaCount(int net_id, const std::vector<float>& features)
{
    if (!isModelLoaded()) {
        std::cerr << "Model not loaded" << std::endl;
        return -1.0f;
    }

    if (_is_wirelength_model) {
        std::cerr << "Loaded model is not a via count prediction model" << std::endl;
        return -1.0f;
    }

    std::vector<float> normalized_features = normalizeFeatures(features, false);

    std::vector<float> output = _model_handler->predict(normalized_features);
    if (output.empty()) {
        std::cerr << "Prediction failed" << std::endl;
        return -1.0f;
    }

    float prediction = output[0];
    std::cout << "Net " << net_id << " via count prediction: " << prediction << std::endl;
    return prediction;
}

bool WirelengthPredictor::loadViaNormalizationParams(const std::string& params_path)
{
    if (!_via_normalizer->loadMinMaxParams(params_path)) {
        std::cerr << "Failed to load via normalization parameters: " << params_path << std::endl;
        return false;
    }
    std::cout << "Successfully loaded via normalization parameters: " << params_path << std::endl;
    return true;
}

bool WirelengthPredictor::loadWirelengthNormalizationParams(const std::string& params_path)
{
    if (!_wirelength_normalizer->loadMinMaxParams(params_path)) {
        std::cerr << "Failed to load wirelength normalization parameters: " << params_path << std::endl;
        return false;
    }
    std::cout << "Successfully loaded wirelength normalization parameters: " << params_path << std::endl;
    return true;
}

int WirelengthPredictor::getRequiredFeatureCount() const
{
    if (!isModelLoaded()) {
        std::cerr << "Model not loaded" << std::endl;
        return 0;
    }

    return _model_handler->getInputFeatureCount();
}

bool WirelengthPredictor::isModelLoaded() const
{
    // Check if model handler has loaded a model
    return _model_handler->getInputFeatureCount() > 0;
}

std::vector<float> WirelengthPredictor::normalizeFeatures(const std::vector<float>& features, bool is_wirelength)
{
    if (is_wirelength && _wirelength_normalizer && _wirelength_normalizer->isReady()) {
        std::vector<float> normalized = _wirelength_normalizer->normalize(features);
        return normalized;
    } else if (!is_wirelength && _via_normalizer && _via_normalizer->isReady()) {
        std::vector<float> normalized = _via_normalizer->normalize(features);
        return normalized;
    } else {
        std::cerr << "Warning: normalization parameters not loaded, using raw features" << std::endl;
        return features;
    }
}

} // namespace ipl