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

#include "normalization_handler.hh"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include "Log.hh"
#include "json.hpp"

namespace ipl {

NormalizationHandler::NormalizationHandler() : _is_loaded(false)
{
}

NormalizationHandler::~NormalizationHandler()
{
}

bool NormalizationHandler::loadMinMaxParams(const std::string& params_path)
{
  return _parseJsonParams(params_path);
}

void NormalizationHandler::setMinMaxParams(const std::vector<float>& data_min, const std::vector<float>& data_max,
                                           const std::vector<std::string>& feature_names)
{
  if (data_min.size() != data_max.size()) {
    LOG_WARNING << "Error: data_min and data_max must have same size";
    return;
  }

  _data_min = data_min;
  _data_max = data_max;
  _feature_names = feature_names;
  _is_loaded = true;
}

std::vector<float> NormalizationHandler::normalize(const std::vector<float>& features) const
{
  if (!_is_loaded) {
    LOG_WARNING << "Error: Normalization parameters not loaded";
    return features;
  }

  if (features.size() != _data_min.size()) {
    LOG_WARNING << "Error: Feature size mismatch. Expected " << _data_min.size() << ", got " << features.size();
    return features;
  }

  std::vector<float> normalized_features;
  normalized_features.reserve(features.size());

  for (size_t i = 0; i < features.size(); ++i) {
    float range = _data_max[i] - _data_min[i];
    if (range == 0.0f) {
      // if max == min, normailzed = 0
      normalized_features.push_back(0.0f);
    } else {
      // MinMax normalization: (x - min) / (max - min)
      float normalized = (features[i] - _data_min[i]) / range;

      normalized = std::max(0.0f, std::min(1.0f, normalized));
      normalized_features.push_back(normalized);
    }
  }

  return normalized_features;
}

bool NormalizationHandler::isReady() const
{
  return _is_loaded;
}

std::vector<std::string> NormalizationHandler::getFeatureNames() const
{
  return _feature_names;
}

size_t NormalizationHandler::getFeatureCount() const
{
  return _data_min.size();
}

bool NormalizationHandler::_parseJsonParams(const std::string& params_path)
{
  std::ifstream file(params_path);
  if (!file.is_open()) {
    LOG_WARNING << "Error: Cannot open normalization parameters file: " << params_path;
    return false;
  }

  try {
    nlohmann::json j;
    file >> j;

    // parse data
    if (j.contains("data_min") && j.contains("data_max")) {
      _data_min = j["data_min"].get<std::vector<float>>();
      _data_max = j["data_max"].get<std::vector<float>>();

      if (j.contains("feature_names")) {
        _feature_names = j["feature_names"].get<std::vector<std::string>>();
      }

      if (_data_min.size() != _data_max.size() || _data_min.empty()) {
        LOG_WARNING << "Error: Invalid normalization parameters - size mismatch";
        return false;
      }

      _is_loaded = true;

      LOG_INFO << "Successfully loaded normalization parameters:";
      LOG_INFO << "  Features: " << _data_min.size();
      LOG_INFO << "  Feature names: ";
      for (const auto& name : _feature_names) {
        LOG_INFO << name << " ";
      }

      return true;
    } else {
      LOG_WARNING << "Error: Missing required fields in JSON";
      return false;
    }

  } catch (const std::exception& e) {
    LOG_WARNING << "Error parsing JSON: " << e.what();
    return false;
  }
}

}  // namespace ipl
