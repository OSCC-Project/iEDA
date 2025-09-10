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
 * @Description: AI-based wirelength evaluator implementation
 * @FilePath: /iEDA/src//ai/predictor/ipl_wirelength/ai_wirelength.cc
 */

#include "ai_wirelength.hh"

#include <vector>

#include "Log.hh"
#include "Point.hh"
#include "Rectangle.hh"
#include "TopologyManager.hh"

namespace ipl {

AIWirelength* AIWirelength::_instance = nullptr;

bool AIWirelength::init(const std::string& model_path, const std::string& params_path, TopologyManager* topology_manager)
{
  _topology_manager = topology_manager;

  return loadModel(model_path) && loadNormalizationParams(params_path);
}

bool AIWirelength::loadModel(const std::string& model_path)
{
  if (_predictor->loadModel(model_path)) {
    _is_model_loaded = true;
    LOG_INFO << "Successfully loaded AI wirelength prediction model: " << model_path;
    return true;
  } else {
    LOG_ERROR << "Failed to load AI wirelength prediction model: " << model_path;
    return false;
  }
}

bool AIWirelength::loadNormalizationParams(const std::string& params_path)
{
  if (_predictor->loadWirelengthNormalizationParams(params_path)) {
    LOG_INFO << "Successfully loaded wirelength normalization parameters: " << params_path;
    return true;
  } else {
    LOG_ERROR << "Failed to load wirelength normalization parameters: " << params_path;
    return false;
  }
}

int64_t AIWirelength::obtainTotalWirelength()
{
  if (!_is_model_loaded) {
    LOG_ERROR << "AI wirelength model not loaded";
    return 0;
  }

  int64_t total_wirelength = 0;

  for (auto* network : _topology_manager->get_network_list()) {
    int64_t net_wirelength = obtainNetWirelength(network->get_network_id());
    total_wirelength += net_wirelength;
  }

  return total_wirelength;
}

int64_t AIWirelength::obtainNetWirelength(int32_t net_id)
{
  auto* network = _topology_manager->findNetworkById(net_id);
  if (!network) {
    LOG_ERROR << "Network with ID " << net_id << " not found";
    return 0;
  }

  // Extract features for this net
  std::vector<float> features = extractNetFeatures(net_id);

  // Predict wirelength using the AI model
  float predicted_wirelength_ratio = _predictor->predictWirelength(features);

  int64_t predicted_wirelength = features[5] * predicted_wirelength_ratio;

  // Convert float to int64_t (assuming wirelength is in integer units)
  return predicted_wirelength;
}

int64_t AIWirelength::obtainPartOfNetWirelength(int32_t net_id, int32_t sink_pin_id)
{
  // For simplicity, we'll just return the full net wirelength
  // In a real implementation, you might want to predict partial wirelength
  return obtainNetWirelength(net_id);
}

std::vector<float> AIWirelength::extractNetFeatures(int32_t net_id)
{
  auto* network = _topology_manager->findNetworkById(net_id);
  if (!network) {
    LOG_ERROR << "Network with ID " << net_id << " not found";
    return {};
  }

  std::vector<float> features;

  // Bounding box dimensions (0:width, 1:height)
  Rectangle<int32_t> net_bbox = network->obtainNetWorkShape();
  int width = net_bbox.get_width();
  int height = net_bbox.get_height();
  features.push_back(static_cast<float>(width));
  features.push_back(static_cast<float>(height));

  // Number of pins (2:pin_num)
  int num_pins = network->get_node_list().size();
  features.push_back(static_cast<float>(num_pins));

  // Aspect ratio (3:aspect_ratio)
  float aspect_ratio = (height > 0) ? static_cast<float>(width) / height : 0.0f;
  features.push_back(aspect_ratio);

  // Lness (4:l_ness)
  // For simplicity, we set a constant value; in practice, compute based on pin distribution
  float lness = 0.5f;
  features.push_back(lness);

  // Steiner Tree (5:rsmt)
  // For simplicity, we assume HPWL as the Steiner tree length
  int64_t rsmt = network->obtainNetWorkShape().get_half_perimeter();
  features.push_back(static_cast<float>(rsmt));

  // area (6:area)
  int area = width * height;
  features.push_back(static_cast<float>(area));

  // route_ratio_x (7:route_ratio_x)
  float route_ratio_x = (width > 0) ? static_cast<float>(width) / area : 0.0f;
  features.push_back(route_ratio_x);

  // route_ratio_y (8:route_ratio_y)
  float route_ratio_y = (height > 0) ? static_cast<float>(height) / area : 0.0f;
  features.push_back(route_ratio_y);

  return features;
}

}  // namespace ipl