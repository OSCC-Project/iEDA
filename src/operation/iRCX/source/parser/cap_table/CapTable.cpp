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
#include <fstream>
#include <sstream>

#include "CapTable.hpp"
#include "log/Log.hh"
namespace ircx {
namespace parser {

namespace {

std::string formatConfigKey(const std::string& layer_name,
                            const std::string& overLayer,
                            const std::string& underLayer)
{
  std::ostringstream oss;
  oss << layer_name << " OVER " << overLayer;
  if (!underLayer.empty()) {
    oss << " UNDER " << underLayer;
  }
  return oss.str();
}

}  // namespace

bool CapTable::loadFromFile(const std::string& filePath)
{
  std::ifstream file(filePath);
  if (!file.is_open()) {
    LOG_ERROR << "Failed to open cap table file: " << filePath;
    return false;
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  return loadFromString(buffer.str());
}

bool CapTable::loadFromString(const std::string& content)
{
  std::map<std::string, CapTableConfig> configs;

  std::istringstream iss(content);
  std::string line;
  std::string headerLine;
  std::vector<std::string> dataLines;

  auto flushConfig = [&]() -> bool {
    if (headerLine.empty()) {
      return true;
    }
    if (!parseConfig(headerLine, dataLines, configs)) {
      return false;
    }
    dataLines.clear();
    return true;
  };

  while (std::getline(iss, line)) {
    // Trim leading/trailing whitespace.
    line.erase(0, line.find_first_not_of(" \t\r\n"));
    line.erase(line.find_last_not_of(" \t\r\n") + 1);

    // Skip empty lines and comments.
    if (line.empty() || line[0] == '#') {
      continue;
    }

    // Detect a new-format header line.
    bool isHeader = false;
    if (line.size() >= 2) {
      if ((line[0] == 'A' || line[0] == 'B') && line[1] == ' ') {
        if (line.find("OVER") != std::string::npos) {
          isHeader = true;
        }
      }
    }

    if (isHeader) {
      if (!flushConfig()) {
        configs_.clear();
        return false;
      }
      headerLine = line;
    } else {
      if (!headerLine.empty()) {
        dataLines.push_back(line);
      }
    }
  }

  if (!flushConfig()) {
    configs_.clear();
    return false;
  }

  configs_ = std::move(configs);
  LOG_INFO << "Loaded " << configs_.size() << " cap table configs";
  return !configs_.empty();
}

bool CapTable::parseConfig(const std::string& headerLine,
                           const std::vector<std::string>& dataLines)
{
  return parseConfig(headerLine, dataLines, configs_);
}

bool CapTable::parseConfig(const std::string& headerLine,
                           const std::vector<std::string>& dataLines,
                           std::map<std::string, CapTableConfig>& configs)
{
  if (headerLine.empty() || dataLines.empty()) {
    return false;
  }

  std::istringstream iss(headerLine);
  std::string type, layer_name, overKw, overLayer;

  if (!(iss >> type >> layer_name >> overKw >> overLayer)) {
    LOG_ERROR << "Invalid cap table header format: " << headerLine;
    return false;
  }

  if (type != "A" && type != "B") {
    LOG_ERROR << "Invalid cap table type (expected A or B): " << type;
    return false;
  }

  if (overKw != "OVER") {
    LOG_ERROR << "Missing OVER keyword in cap table header: " << headerLine;
    return false;
  }

  std::string underLayer;
  std::string underKw;
  if (iss >> underKw) {
    if (underKw == "UNDER") {
      if (!(iss >> underLayer)) {
        LOG_ERROR << "Missing layer after UNDER in cap table header: " << headerLine;
        return false;
      }
    } else {
      LOG_ERROR << "Unexpected keyword in cap table header (expected UNDER): " << underKw;
      return false;
    }
  }

  if (type == "A" && !underLayer.empty()) {
    LOG_ERROR << "Type A should not have UNDER clause: " << headerLine;
    return false;
  }
  if (type == "B" && underLayer.empty()) {
    LOG_ERROR << "Type B must have UNDER clause: " << headerLine;
    return false;
  }

  std::string key = makeKey(layer_name, overLayer, underLayer);

  CapTableConfig config;
  config.key = key;
  config.type = type;
  config.layer_name = layer_name;
  config.over_layer = overLayer;
  config.under_layer = underLayer;

  for (const auto& dataLine : dataLines) {
    std::istringstream dataIss(dataLine);
    CapTableEntry entry;
    if (!(dataIss >> entry.distance >> entry.coupling_cap >> entry.ground_cap)) {
      LOG_ERROR << "Invalid cap table data line: " << dataLine;
      continue;
    }
    config.data.push_back(entry);
  }

  if (config.data.empty()) {
    LOG_ERROR << "No valid data for cap table config: " << key;
    return false;
  }

  configs[key] = config;
  return true;
}

const CapTableConfig* CapTable::get_config(
    const std::string& layer_name,
    const std::string& overLayer,
    const std::string& underLayer) const
{
  std::string key = makeKey(layer_name, overLayer, underLayer);
  auto iter = configs_.find(key);
  if (iter != configs_.end()) {
    return &(iter->second);
  }
  return nullptr;
}

std::vector<std::string> CapTable::get_all_keys() const
{
  std::vector<std::string> keys;
  keys.reserve(configs_.size());
  for (const auto& [key, config] : configs_) {
    keys.push_back(key);
  }
  return keys;
}

std::string CapTable::makeKey(const std::string& layer_name,
                              const std::string& overLayer,
                              const std::string& underLayer)
{
  std::ostringstream oss;
  if (underLayer.empty()) {
    oss << "A " << layer_name << " OVER " << overLayer;
  } else {
    oss << "B " << layer_name << " OVER " << overLayer << " UNDER " << underLayer;
  }
  return oss.str();
}

CapacitanceResult CapTable::interpolate(
    const std::string& layer_name,
    const std::string& overLayer,
    const std::string& underLayer,
    double neighborDistance) const
{
  CapacitanceResult result;

  const CapTableConfig* config = get_config(layer_name, overLayer, underLayer);
  if (!config) {
    LOG_ERROR << "Cap table config not found: "
              << formatConfigKey(layer_name, overLayer, underLayer);
    return result;
  }

  const auto& data = config->data;

  if (data.empty()) {
    LOG_ERROR << "No data points in cap table config: " << config->key;
    return result;
  }

  if (neighborDistance < 0.0) {
    return isolatedResult(*config);
  }

  if (data.size() == 1) {
    result.ground_cap = data[0].ground_cap;
    result.coupling_cap = data[0].coupling_cap;
    return result;
  }

  auto [idx1, idx2] = findBracketingIndices(data, neighborDistance);

  if (idx1 < 0 || idx2 < 0) {
    if (neighborDistance <= data.front().distance) {
      result.ground_cap = data.front().ground_cap;
      result.coupling_cap = data.front().coupling_cap;
    } else {
      result.ground_cap = data.back().ground_cap;
      result.coupling_cap = data.back().coupling_cap;
    }
    return result;
  }

  const auto& p1 = data[idx1];
  const auto& p2 = data[idx2];

  result.ground_cap = linearInterpolate(
      neighborDistance, p1.distance, p1.ground_cap, p2.distance, p2.ground_cap);
  result.coupling_cap = linearInterpolate(
      neighborDistance, p1.distance, p1.coupling_cap, p2.distance, p2.coupling_cap);

  return result;
}

CapacitanceResult CapTable::isolatedResult(const CapTableConfig& config) const
{
  CapacitanceResult result = farthestResult(config);
  result.coupling_cap = 0.0;
  return result;
}

CapacitanceResult CapTable::farthestResult(const CapTableConfig& config) const
{
  CapacitanceResult result;
  if (config.data.empty()) {
    return result;
  }

  const auto& last = config.data.back();
  result.ground_cap = last.ground_cap;
  result.coupling_cap = last.coupling_cap;
  return result;
}

double CapTable::linearInterpolate(
    double x, double x1, double y1, double x2, double y2) const
{
  if (std::abs(x2 - x1) < 1e-9) {
    return (y1 + y2) / 2.0;
  }
  return y1 + (((y2 - y1) * (x - x1)) / (x2 - x1));
}

std::pair<int, int> CapTable::findBracketingIndices(
    const std::vector<CapTableEntry>& data,
    double distance) const
{
  if (data.size() < 2) {
    return {-1, -1};
  }

  for (size_t i = 0; i < data.size() - 1; ++i) {
    if (data[i].distance <= distance && distance <= data[i + 1].distance) {
      return {static_cast<int>(i), static_cast<int>(i + 1)};
    }
  }

  return {-1, -1};
}

// ======================== 查询 ========================

CapacitanceResult CapTable::queryTwoLayerCap(
    const std::string& layer_name,
    const std::string& belowLayer,
    double neighborDistance) const
{
  return interpolate(layer_name, belowLayer, "", neighborDistance);
}

CapacitanceResult CapTable::queryThreeLayerCap(
    const std::string& layer_name,
    const std::string& belowLayer,
    const std::string& aboveLayer,
    double neighborDistance) const
{
  return interpolate(layer_name, belowLayer, aboveLayer, neighborDistance);
}

CapacitanceResult CapTable::queryTwoLayerIsolatedCap(
    const std::string& layer_name,
    const std::string& belowLayer) const
{
  const CapTableConfig* config = get_config(layer_name, belowLayer, "");
  if (!config) {
    LOG_ERROR << "Cap table config not found: "
              << formatConfigKey(layer_name, belowLayer, "");
    return {};
  }
  return isolatedResult(*config);
}

CapacitanceResult CapTable::queryThreeLayerIsolatedCap(
    const std::string& layer_name,
    const std::string& belowLayer,
    const std::string& aboveLayer) const
{
  const CapTableConfig* config = get_config(layer_name, belowLayer, aboveLayer);
  if (!config) {
    LOG_ERROR << "Cap table config not found: "
              << formatConfigKey(layer_name, belowLayer, aboveLayer);
    return {};
  }
  return isolatedResult(*config);
}

CapacitanceResult CapTable::queryTwoLayerFarthestCap(
    const std::string& layer_name,
    const std::string& belowLayer) const
{
  const CapTableConfig* config = get_config(layer_name, belowLayer, "");
  if (!config) {
    LOG_ERROR << "Cap table config not found: "
              << formatConfigKey(layer_name, belowLayer, "");
    return {};
  }
  return farthestResult(*config);
}

CapacitanceResult CapTable::queryThreeLayerFarthestCap(
    const std::string& layer_name,
    const std::string& belowLayer,
    const std::string& aboveLayer) const
{
  const CapTableConfig* config = get_config(layer_name, belowLayer, aboveLayer);
  if (!config) {
    LOG_ERROR << "Cap table config not found: "
              << formatConfigKey(layer_name, belowLayer, aboveLayer);
    return {};
  }
  return farthestResult(*config);
}

}  // namespace parser
}  // namespace ircx
