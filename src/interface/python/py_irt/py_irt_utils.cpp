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
#include <tcl_util.h>

#include <string>

#include "RTInterface.hpp"
#include "json_parser.h"
#include "py_irt.h"

namespace python_interface {
using tcl::ValueType;

std::map<std::string, double> strToDoubleMap(const std::string& input)
{
  // Trim the leading and trailing whitespaces from the input
  std::string input_trimmed = input;
  input_trimmed.erase(input_trimmed.begin(),
                      std::find_if(input_trimmed.begin(), input_trimmed.end(), [](int ch) { return !std::isspace(ch); }));
  input_trimmed.erase(std::find_if(input_trimmed.rbegin(), input_trimmed.rend(), [](int ch) { return !std::isspace(ch); }).base(),
                      input_trimmed.end());
  std::map<std::string, double> result;
  std::stringstream ss(input_trimmed.substr(1, input_trimmed.length() - 2));
  std::string item;
  while (std::getline(ss, item, ',')) {
    size_t pos = item.find(":");
    std::string key = item.substr(0, pos);
    // Trim whitespaces from the key and value
    key.erase(std::remove_if(key.begin(), key.end(), [](unsigned char c) { return std::isspace(c); }), key.end());
    std::string value_str = item.substr(pos + 1);
    value_str.erase(std::remove_if(value_str.begin(), value_str.end(), [](unsigned char c) { return std::isspace(c); }), value_str.end());
    double value = std::stod(value_str);
    result[key] = value;
  }
  return result;
}

std::vector<std::string> strToVector(const std::string& input)
{
  std::vector<std::string> result;
  std::string token;
  for (char c : input) {
    switch (c) {
      case ' ':
      case ',': {
        if (!token.empty()) {
          result.push_back(token);
          token.clear();
        }
        break;
      }
      default:
        token.push_back(c);
    }
  }
  return result;
}

std::vector<double> strToDoubleVec(const std::string& input)
{
  auto tmp = strToVector(input);
  std::vector<double> result;
  result.reserve(tmp.size());
  for_each(tmp.begin(), tmp.end(), [&result](const std::string& s) { result.push_back(std::stod(s)); });
  return result;
}

std::vector<int> strToIntVec(const std::string& input)
{
  auto tmp = strToVector(input);
  std::vector<int> result;
  result.reserve(tmp.size());
  for_each(tmp.begin(), tmp.end(), [&result](const std::string& s) { result.push_back(std::stoi(s)); });
  return result;
}

// 通过json文件对config进行初始化
bool initConfigMapByJSON(const std::string& config, std::map<std::string, std::any>& config_map)
{
  auto config_file = std::ifstream(config);
  if (!config_file.is_open()) {
    return false;
  }
  nlohmann::json json;
  config_file >> json;
  nlohmann::json rt_json = json["RT"];
  std::string value;
  value = ieda::getJsonData(json, {"RT", "-temp_directory_path"});
  config_map.insert(std::make_pair("-temp_directory_path", value));
  value = ieda::getJsonData(json, {"RT", "-bottom_routing_layer"});
  config_map.insert(std::make_pair("-bottom_routing_layer", value));
  value = ieda::getJsonData(json, {"RT", "-top_routing_layer"});
  config_map.insert(std::make_pair("-top_routing_layer", value));
  value = ieda::getJsonData(json, {"RT", "-thread_number"});
  config_map.insert(std::make_pair("-thread_number", std::stoi(value)));
  value = ieda::getJsonData(json, {"RT", "-enable_timing"});
  config_map.insert(std::make_pair("-enable_timing", std::stoi(value)));
  value = ieda::getJsonData(json, {"RT", "-output_inter_result"});
  config_map.insert(std::make_pair("-output_inter_result", std::stoi(value)));

  for (nlohmann::json::iterator item = rt_json.begin(); item != rt_json.end(); ++item) {
    config_map.insert(std::make_pair(item.key(), item.value()));
  }

  return true;
}

}  // namespace python_interface