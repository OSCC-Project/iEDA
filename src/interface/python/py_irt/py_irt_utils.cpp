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

#include <iRT/api/RTInterface.hpp>
#include <string>

#include "py_irt.h"

namespace python_interface {
using tcl::ValueType;
static std::vector<std::pair<std::string, ValueType>> _config_list;
void initConfigList()
{
  // string temp_directory_path
  _config_list.push_back(std::make_pair("-temp_directory_path", ValueType::kString));
  // string bottom_routing_layer
  _config_list.push_back(std::make_pair("-bottom_routing_layer", ValueType::kString));
  // string top_routing_layer
  _config_list.push_back(std::make_pair("-top_routing_layer", ValueType::kString));
  // irt_int ta_panel_max_iter_num
  _config_list.push_back(std::make_pair("-ta_panel_max_iter_num", ValueType::kInt));
  // irt_int dr_box_max_iter_num
  _config_list.push_back(std::make_pair("-dr_box_max_iter_num", ValueType::kInt));
  // // irt_int rt_log_verbose;
  // _config_list.push_back(std::make_pair("-rt_log_verbose", ValueType::kInt));
  // // std::string rt_temp_directory_path;
  // _config_list.push_back(std::make_pair("-rt_temp_directory_path", ValueType::kString));
  // // std::string log_file_path;
  // _config_list.push_back(std::make_pair("-log_file_path", ValueType::kString));
  // // std::string gw_temp_directory_path;
  // _config_list.push_back(std::make_pair("-gw_temp_directory_path", ValueType::kString));
  // // double gm_global_utilization_ratio;
  // _config_list.push_back(std::make_pair("-gm_global_utilization_ratio", ValueType::kDouble));
  // // std::string gm_temp_directory_path;
  // _config_list.push_back(std::make_pair("-gm_temp_directory_path", ValueType::kString));
  // // irt_int gp_routing_size;
  // _config_list.push_back(std::make_pair("-gp_routing_size", ValueType::kInt));
  // // std::string gp_temp_directory_path;
  // _config_list.push_back(std::make_pair("-gp_temp_directory_path", ValueType::kString));
  // // std::string gp_guide_file_path;
  // _config_list.push_back(std::make_pair("-gp_guide_file_path", ValueType::kString));
  // // irt_int la_max_segment_length;
  // _config_list.push_back(std::make_pair("-la_max_segment_length", ValueType::kInt));
  // // double la_via_weight;
  // _config_list.push_back(std::make_pair("-la_via_weight", ValueType::kDouble));
  // // double la_congestion_weight;
  // _config_list.push_back(std::make_pair("-la_congestion_weight", ValueType::kDouble));
  // // std::string la_clock_net_lowest_routing_layer_name;
  // _config_list.push_back(std::make_pair("-la_clock_net_lowest_routing_layer_name", ValueType::kString));
  // // irt_int la_opt_tree_iteration_num;
  // _config_list.push_back(std::make_pair("-la_opt_tree_iteration_num", ValueType::kInt));
  // // std::string la_temp_directory_path;
  // _config_list.push_back(std::make_pair("-la_temp_directory_path", ValueType::kString));
  // // std::string pa_temp_directory_path;
  // _config_list.push_back(std::make_pair("-pa_temp_directory_path", ValueType::kString));
  // // irt_int pr_single_enlarge_range;
  // _config_list.push_back(std::make_pair("-pr_single_enlarge_range", ValueType::kInt));
  // // irt_int pr_max_enlarge_times;
  // _config_list.push_back(std::make_pair("-pr_max_enlarge_times", ValueType::kInt));
  // // double pr_resource_weight;
  // _config_list.push_back(std::make_pair("-pr_resource_weight", ValueType::kDouble));
  // // double pr_congestion_weight;
  // _config_list.push_back(std::make_pair("-pr_congestion_weight", ValueType::kDouble));
  // // irt_int pr_opt_tree_iteration_num;
  // _config_list.push_back(std::make_pair("-pr_opt_tree_iteration_num", ValueType::kInt));
  // // std::string pr_temp_directory_path;
  // _config_list.push_back(std::make_pair("-pr_temp_directory_path", ValueType::kString));
  // // std::string ps_temp_directory_path;
  // _config_list.push_back(std::make_pair("-ps_temp_directory_path", ValueType::kString));
  // // std::string rm_temp_directory_path;
  // _config_list.push_back(std::make_pair("-rm_temp_directory_path", ValueType::kString));
  // // double rr_grid_cost;
  // _config_list.push_back(std::make_pair("-rr_grid_cost", ValueType::kDouble));
  // // double rr_around_cost;
  // _config_list.push_back(std::make_pair("-rr_around_cost", ValueType::kDouble));
  // // double rr_via_cost;
  // _config_list.push_back(std::make_pair("-rr_via_cost", ValueType::kDouble));
  // // double rr_drc_cost;
  // _config_list.push_back(std::make_pair("-rr_drc_cost", ValueType::kDouble));
  // // irt_int rr_routing_size;
  // _config_list.push_back(std::make_pair("-rr_routing_size", ValueType::kInt));
  // // std::string rr_temp_directory_path;
  // _config_list.push_back(std::make_pair("-rr_temp_directory_path", ValueType::kString));
  // // double ra_initial_penalty_para;
  // _config_list.push_back(std::make_pair("-ra_initial_penalty_para", ValueType::kDouble));
  // // double ra_penalty_para_drop_rate;
  // _config_list.push_back(std::make_pair("-ra_penalty_para_drop_rate", ValueType::kDouble));
  // // irt_int ra_max_outer_iter_num;
  // _config_list.push_back(std::make_pair("-ra_max_outer_iter_num", ValueType::kInt));
  // // irt_int ra_max_inner_iter_num;
  // _config_list.push_back(std::make_pair("-ra_max_inner_iter_num", ValueType::kInt));
  // // std::string ra_temp_directory_path;
  // _config_list.push_back(std::make_pair("-ra_temp_directory_path", ValueType::kString));
  // // std::string ro_temp_directory_path;
  // _config_list.push_back(std::make_pair("-ro_temp_directory_path", ValueType::kString));
  // // double sr_grid_cost;
  // _config_list.push_back(std::make_pair("-sr_grid_cost", ValueType::kDouble));
  // // double sr_around_cost;
  // _config_list.push_back(std::make_pair("-sr_around_cost", ValueType::kDouble));
  // // double sr_via_cost;
  // _config_list.push_back(std::make_pair("-sr_via_cost", ValueType::kDouble));
  // // double sr_drc_cost;
  // _config_list.push_back(std::make_pair("-sr_drc_cost", ValueType::kDouble));
  // // irt_int sr_routing_size;
  // _config_list.push_back(std::make_pair("-sr_routing_size", ValueType::kInt));
  // // std::string sr_temp_directory_path;
  // _config_list.push_back(std::make_pair("-sr_temp_directory_path", ValueType::kString));
  // // std::string tg_temp_directory_path;
  // _config_list.push_back(std::make_pair("-tg_temp_directory_path", ValueType::kString));
  // // std::string tg_slute_file_path;
  // _config_list.push_back(std::make_pair("-tg_slute_file_path", ValueType::kString));
  // // double ta_adjacent_segment_ratio;
  // _config_list.push_back(std::make_pair("-ta_adjacent_segment_ratio", ValueType::kDouble));
  // // double ta_connect_pin_ratio;
  // _config_list.push_back(std::make_pair("-ta_connect_pin_ratio", ValueType::kDouble));
  // // double ta_pin_obs_ratio;
  // _config_list.push_back(std::make_pair("-ta_pin_obs_ratio", ValueType::kDouble));
  // // double ta_pa_cost;
  // _config_list.push_back(std::make_pair("-ta_pa_cost", ValueType::kDouble));
  // // double ta_overlap_ratio;
  // _config_list.push_back(std::make_pair("-ta_overlap_ratio", ValueType::kDouble));
  // // double ta_drc_ratio;
  // _config_list.push_back(std::make_pair("-ta_drc_ratio", ValueType::kDouble));
  // // irt_int ta_routing_size;
  // _config_list.push_back(std::make_pair("-ta_routing_size", ValueType::kInt));
  // // std::string ta_temp_directory_path;
  // _config_list.push_back(std::make_pair("-ta_temp_directory_path", ValueType::kString));
}

// get_value from

//
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
  initConfigList();
  auto config_file = std::ifstream(config);
  if (!config_file.is_open()) {
    return false;
  }
  nlohmann::json json;
  config_file >> json;
  nlohmann::json rt_json = json["RT"];
  for(nlohmann::json::iterator item = rt_json.begin(); item != rt_json.end(); ++item){
    for(auto& kv : _config_list){
      std::any value;
      if(item.key() == kv.first){
        switch (kv.second)
        {
        case tcl::ValueType::kInt:
          value = std::stoi(item.value().get<std::string>());
          break;
        case tcl::ValueType::kIntList:
          value = strToIntVec(item.value());
        case tcl::ValueType::kDouble:
          value = std::stod(item.value().get<std::string>());
          break;
        case tcl::ValueType::kDoubleList:
          value = strToDoubleVec(item.value());
          break;
        case tcl::ValueType::kString:
          value = std::string(item.value());
          break;
        case tcl::ValueType::kStringList:
          value = strToVector(std::string(item.value()));
          break;
        case tcl::ValueType::kStringDoubleMap:
          value = strToDoubleMap(std::string(item.value()));
          break;
        default:
          break;
        }
        config_map.insert(std::make_pair(kv.first, value));
      }
    }
  }
  return true;
}

// 使用字典对config进行初始化
bool initConfigMapByDict(std::map<std::string, std::string>& config_dict, std::map<std::string, std::any>& config_map)
{
  for (auto& kv : _config_list) {
    if (!config_map.contains(kv.first)) {
      continue;
    }
    std::string key = kv.first;
    std::string val_str = config_dict[key];
    std::any value;
    switch (kv.second) {
      case tcl::ValueType::kInt:
        value = std::stoi(config_dict[kv.first]);
        break;
      case tcl::ValueType::kIntList:
        value = strToIntVec(val_str);
        break;
      case tcl::ValueType::kDouble:
        value = std::stod(val_str);
        break;
      case tcl::ValueType::kDoubleList:
        value = strToDoubleVec(val_str);
        break;
      case tcl::ValueType::kString:
        value = val_str;
        break;
      case tcl::ValueType::kStringList:
        value = strToVector(val_str);
        break;
      case tcl::ValueType::kStringDoubleMap:
        value = strToDoubleMap(val_str);
        break;
    }
    config_map.insert({key, value});
  }
  return true;
}
}  // namespace python_interface