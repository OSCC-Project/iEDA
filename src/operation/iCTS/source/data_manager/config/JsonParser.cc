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
/**
 * @file JsonParser.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "JsonParser.hh"

#include "COMUtil.hh"
#include "CTSAPI.hh"
#include "json/json.hpp"
#include "log/Log.hh"
namespace icts {

JsonParser::JsonParser()
{
  char program_name[] = "JsonParser";
  char* argv[] = {program_name};
  // We need to initialize the log system here, because JsonParser() may be called in pybind,
  // which does not have a main function to initialize the log system.
  const std::string log_dir = "/tmp/icts_logs/";
  Log::makeSureDirectoryExist(log_dir);
  Log::init(argv, log_dir);
}

JsonParser& JsonParser::getInstance()
{
  static JsonParser parser;
  return parser;
}

void JsonParser::parse(const string& json_file, CtsConfig* config) const
{
  std::ifstream& ifs = COMUtil::getInputFileStream(json_file);
  if (!ifs) {
    LOG_FATAL << "no config file: " << json_file;
  } else {
    LOG_INFO << "read config success : " << json_file;
  }
  nlohmann::json json;
  std::string data_type_string = "string";

  ifs >> json;

  {
    if (COMUtil::getData(json, {"use_skew_tree_alg"}) != nullptr) {
      std::string use_skew_tree_alg = COMUtil::getData(json, {"use_skew_tree_alg"});
      if (use_skew_tree_alg == "true" || use_skew_tree_alg == "True" || use_skew_tree_alg == "TRUE" || use_skew_tree_alg == "On"
          || use_skew_tree_alg == "ON" || use_skew_tree_alg == "on") {
        config->set_use_skew_tree_alg(true);
      } else {
        config->set_use_skew_tree_alg(false);
      }
    }
    if (COMUtil::getData(json, {"skew_bound"}) != nullptr) {
      std::string skew_bound = COMUtil::getData(json, {"skew_bound"});
      config->set_skew_bound(std::stod(skew_bound));
    }
    if (COMUtil::getData(json, {"max_buf_tran"}) != nullptr) {
      std::string max_buf_tran = COMUtil::getData(json, {"max_buf_tran"});
      config->set_max_buf_tran(std::stod(max_buf_tran));
    }
    if (COMUtil::getData(json, {"max_sink_tran"}) != nullptr) {
      std::string max_sink_tran = COMUtil::getData(json, {"max_sink_tran"});
      config->set_max_sink_tran(std::stod(max_sink_tran));
    }
    if (COMUtil::getData(json, {"max_cap"}) != nullptr) {
      std::string max_cap = COMUtil::getData(json, {"max_cap"});
      config->set_max_cap(std::stod(max_cap));
    }
    if (COMUtil::getData(json, {"max_fanout"}) != nullptr) {
      std::string max_fanout = COMUtil::getData(json, {"max_fanout"});
      config->set_max_fanout(std::stoi(max_fanout));
    }
    if (COMUtil::getData(json, {"min_length"}) != nullptr) {
      std::string min_length = COMUtil::getData(json, {"min_length"});
      config->set_min_length(std::stod(min_length));
    }
    if (COMUtil::getData(json, {"max_length"}) != nullptr) {
      std::string max_length = COMUtil::getData(json, {"max_length"});
      config->set_max_length(std::stod(max_length));
    }
    if (COMUtil::getData(json, {"routing_layer"}) != nullptr) {
      std::vector<int> routing_layers = COMUtil::getData(json, {"routing_layer"});
      config->set_routing_layers(routing_layers);
      config->set_h_layer(routing_layers.front());
      config->set_v_layer(routing_layers.back());
    }
    if (COMUtil::getData(json, {"buffer_type"}) != nullptr) {
      config->set_buffer_types(COMUtil::getData(json, {"buffer_type"}));
    }
    if (COMUtil::getData(json, {"root_buffer_type"}) != nullptr) {
      config->set_root_buffer_type(COMUtil::getData(json, {"root_buffer_type"}));
    }
    if (COMUtil::getData(json, {"root_buffer_required"}) != nullptr) {
      std::string root_buffer_required = COMUtil::getData(json, {"root_buffer_required"});
      if (root_buffer_required == "true" || root_buffer_required == "True" || root_buffer_required == "TRUE" || root_buffer_required == "On"
          || root_buffer_required == "ON" || root_buffer_required == "on") {
        config->set_root_buffer_required(true);
      } else {
        config->set_root_buffer_required(false);
      }
    }
    if (COMUtil::getData(json, {"inherit_root"}) != nullptr) {
      std::string inherit_root = COMUtil::getData(json, {"inherit_root"});
      if (inherit_root == "true" || inherit_root == "True" || inherit_root == "TRUE" || inherit_root == "On" || inherit_root == "ON"
          || inherit_root == "on") {
        config->set_inherit_root(true);
      } else {
        config->set_inherit_root(false);
      }
    }
    if (COMUtil::getData(json, {"break_long_wire"}) != nullptr) {
      std::string break_long_wire = COMUtil::getData(json, {"break_long_wire"});
      if (break_long_wire == "true" || break_long_wire == "True" || break_long_wire == "TRUE" || break_long_wire == "On"
          || break_long_wire == "ON" || break_long_wire == "on") {
        config->set_break_long_wire(true);
      } else {
        config->set_break_long_wire(false);
      }
    }
    if (COMUtil::getData(json, {"level_max_length"}) != nullptr) {
      std::vector<std::string> level_max_length = COMUtil::getData(json, {"level_max_length"});
      std::vector<double> level_max_length_double;
      std::transform(level_max_length.begin(), level_max_length.end(), std::back_inserter(level_max_length_double),
                     [](const std::string& str) { return std::stod(str); });
      config->set_level_max_length(level_max_length_double);
    }
    if (COMUtil::getData(json, {"level_max_fanout"}) != nullptr) {
      std::vector<int> level_max_fanout = COMUtil::getData(json, {"level_max_fanout"});
      config->set_level_max_fanout(level_max_fanout);
    }
    if (COMUtil::getData(json, {"level_max_cap"}) != nullptr) {
      std::vector<std::string> level_max_cap = COMUtil::getData(json, {"level_max_cap"});
      std::vector<double> level_max_cap_double;
      std::transform(level_max_cap.begin(), level_max_cap.end(), std::back_inserter(level_max_cap_double),
                     [](const std::string& str) { return std::stod(str); });
      config->set_level_max_cap(level_max_cap_double);
    }
    if (COMUtil::getData(json, {"level_skew_bound"}) != nullptr) {
      std::vector<std::string> level_skew_bound = COMUtil::getData(json, {"level_skew_bound"});
      std::vector<double> level_skew_bound_double;
      std::transform(level_skew_bound.begin(), level_skew_bound.end(), std::back_inserter(level_skew_bound_double),
                     [](const std::string& str) { return std::stod(str); });
      config->set_level_skew_bound(level_skew_bound_double);
    }
    if (COMUtil::getData(json, {"level_cluster_ratio"}) != nullptr) {
      std::vector<std::string> level_cluster_ratio = COMUtil::getData(json, {"level_cluster_ratio"});
      std::vector<double> level_cluster_ratio_double;
      std::transform(level_cluster_ratio.begin(), level_cluster_ratio.end(), std::back_inserter(level_cluster_ratio_double),
                     [](const std::string& str) { return std::stod(str); });
      config->set_level_cluster_ratio(level_cluster_ratio_double);
    }
    if (COMUtil::getData(json, {"shift_level"}) != nullptr) {
      int shift_level = COMUtil::getData(json, {"shift_level"});
      config->set_shift_level(shift_level);
    }
    if (COMUtil::getData(json, {"latency_opt_level"}) != nullptr) {
      int latency_opt_level = COMUtil::getData(json, {"latency_opt_level"});
      config->set_latency_opt_level(latency_opt_level);
    }
    if (COMUtil::getData(json, {"global_latency_opt_ratio"}) != nullptr) {
      std::string global_latency_opt_ratio = COMUtil::getData(json, {"global_latency_opt_ratio"});
      config->set_global_latency_opt_ratio(std::stod(global_latency_opt_ratio));
    }
    if (COMUtil::getData(json, {"local_latency_opt_ratio"}) != nullptr) {
      std::string local_latency_opt_ratio = COMUtil::getData(json, {"local_latency_opt_ratio"});
      config->set_local_latency_opt_ratio(std::stod(local_latency_opt_ratio));
    }

    if (COMUtil::getData(json, {"use_netlist"}) != nullptr) {
      config->set_use_netlist(COMUtil::getData(json, {"use_netlist"}));
    }
    nlohmann::json json_netlist = COMUtil::getData(json, {"net_list"});
    {
      auto clock_name_list = COMUtil::getSerializeObjectData(json_netlist, "clock_name", data_type_string);
      auto net_name_list = COMUtil::getSerializeObjectData(json_netlist, "net_name", data_type_string);

      int size = clock_name_list.size();
      std::vector<std::pair<std::string, std::string>> clock_net_list;
      for (int i = 0; i < size; ++i) {
        clock_net_list.push_back(std::make_pair(clock_name_list[i], net_name_list[i]));
      }

      config->set_netlist(clock_net_list);
    }
  }

  ifs.close();
}

std::string JsonParser::resolvePath(const std::string& path) const
{
  std::string resolved_path = path;
  size_t start_pos = resolved_path.find('$');
  while (start_pos != std::string::npos) {
    size_t end_pos = resolved_path.find('/', start_pos);
    if (end_pos == std::string::npos) {
      end_pos = resolved_path.length();
    }

    std::string var = resolved_path.substr(start_pos + 1, end_pos - start_pos - 1);
    const char* env_val = std::getenv(var.c_str());
    if (env_val != nullptr) {
      resolved_path.replace(start_pos, end_pos - start_pos, env_val);
    }

    start_pos = resolved_path.find('$', start_pos + 1);
  }

  return resolved_path;
}

}  // namespace icts