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
#include "JsonParser.h"

#include "COMUtil.h"
#include "CTSAPI.hpp"
#include "json/json.hpp"
#include "log/Log.hh"
namespace icts {

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
    if (COMUtil::getData(json,
                         {
                             "file_path",
                             "sta_work_dir",
                         })
        != nullptr) {
      config->set_sta_workspace(COMUtil::getData(json, {"file_path", "sta_work_dir"}));
    }
    if (COMUtil::getData(json,
                         {
                             "file_path",
                             "output_def_path",
                         })
        != nullptr) {
      config->set_output_def_path(COMUtil::getData(json, {"file_path", "output_def_path"}));
    }
    if (COMUtil::getData(json,
                         {
                             "file_path",
                             "log_file",
                         })
        != nullptr) {
      config->set_log_file(COMUtil::getData(json, {"file_path", "log_file"}));
    }
    if (COMUtil::getData(json,
                         {
                             "file_path",
                             "gds_file",
                         })
        != nullptr) {
      config->set_gds_file(COMUtil::getData(json, {"file_path", "gds_file"}));
    }

    if (COMUtil::getData(json, {"router_type"}) != nullptr) {
      config->set_router_type(COMUtil::getData(json, {"router_type"}));
    }
    if (COMUtil::getData(json, {"delay_type"}) != nullptr) {
      config->set_delay_type(COMUtil::getData(json, {"delay_type"}));
    }
    if (COMUtil::getData(json, {"skew_bound"}) != nullptr) {
      string skew_bound = COMUtil::getData(json, {"skew_bound"});
      config->set_skew_bound(std::stod(skew_bound));
    }
    if (COMUtil::getData(json, {"max_buf_tran"}) != nullptr) {
      string max_buf_tran = COMUtil::getData(json, {"max_buf_tran"});
      config->set_max_buf_tran(std::stod(max_buf_tran));
    }
    if (COMUtil::getData(json, {"max_sink_tran"}) != nullptr) {
      string max_sink_tran = COMUtil::getData(json, {"max_sink_tran"});
      config->set_max_sink_tran(std::stod(max_sink_tran));
    }
    if (COMUtil::getData(json, {"max_cap"}) != nullptr) {
      string max_cap = COMUtil::getData(json, {"max_cap"});
      config->set_max_cap(std::stod(max_cap));
    }
    if (COMUtil::getData(json, {"max_fanout"}) != nullptr) {
      string max_fanout = COMUtil::getData(json, {"max_fanout"});
      config->set_max_fanout(std::stoi(max_fanout));
    }
    if (COMUtil::getData(json, {"max_length"}) != nullptr) {
      string max_length = COMUtil::getData(json, {"max_length"});
      config->set_max_length(std::stod(max_length));
    }
    if (COMUtil::getData(json, {"scale_size"}) != nullptr) {
      int scale_size = COMUtil::getData(json, {"scale_size"});
      config->set_scale_size(scale_size);
    }
    if (COMUtil::getData(json, {"cluster_type"}) != nullptr) {
      config->set_cluster_type(COMUtil::getData(json, {"cluster_type"}));
    }
    if (COMUtil::getData(json, {"cluster_size"}) != nullptr) {
      config->set_cluster_size(COMUtil::getData(json, {"cluster_size"}));
    }
    if (COMUtil::getData(json, {"buffer_type"}) != nullptr) {
      config->set_buffer_types(COMUtil::getData(json, {"buffer_type"}));
    }
    if (COMUtil::getData(json, {"routing_layer"}) != nullptr) {
      std::vector<int> routing_layers = COMUtil::getData(json, {"routing_layer"});
      config->set_routing_layers(routing_layers);
      config->set_h_layer(routing_layers.front());
      config->set_v_layer(routing_layers.back());
    }
    if (COMUtil::getData(json, {"use_netlist"}) != nullptr) {
      config->set_use_netlist(COMUtil::getData(json, {"use_netlist"}));
    }

    nlohmann::json json_netlist = COMUtil::getData(json, {"net_list"});
    {
      auto clock_name_list = COMUtil::getSerializeObjectData(json_netlist, "clock_name", data_type_string);
      auto net_name_list = COMUtil::getSerializeObjectData(json_netlist, "net_name", data_type_string);

      int size = clock_name_list.size();
      std::vector<std::pair<string, string>> clock_net_list;
      for (int i = 0; i < size; ++i) {
        clock_net_list.push_back(std::make_pair(clock_name_list[i], net_name_list[i]));
      }

      config->set_netlist(clock_net_list);
    }

    nlohmann::json json_ext_models = COMUtil::getData(json, {"external_model"});
    {
      auto net_name_list = COMUtil::getSerializeObjectData(json_ext_models, "net_name", data_type_string);
      auto model_path_list = COMUtil::getSerializeObjectData(json_ext_models, "model_path", data_type_string);

      std::vector<std::pair<string, string>> external_models;
      for (size_t i = 0; i < net_name_list.size(); ++i) {
        external_models.push_back(std::make_pair(net_name_list[i], model_path_list[i]));
      }

      config->set_external_models(external_models);
    }
  }

  ifs.close();
}

}  // namespace icts