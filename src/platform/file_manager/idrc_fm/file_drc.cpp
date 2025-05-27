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
 * @project		iplf
 * @file		file_cts.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Process file.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "file_drc.h"

#include <cstring>
#include <iomanip>
#include <iostream>

#include "DRCViolationType.h"
#include "IdbLayer.h"
#include "idm.h"
#include "idrc_io.h"
#include "ids.hpp"
#include "json_parser.h"

using namespace std;

namespace iplf {

bool FileDrcManager::readFile()
{
  return readJson();
}

bool FileDrcManager::saveFileData()
{
  return saveJson();
}

bool FileDrcManager::saveJson()
{
  auto get_layer_dict = [](std::string layer, std::map<std::string, json>& layer_dict) -> json& {
    if (layer_dict.find(layer) == layer_dict.end()) {
      json json_layer;
      json_layer["number"] = 0;
      json json_list = json::array();
      json_layer["list"] = json_list;
      layer_dict[layer] = json_layer;
    }

    return layer_dict[layer];
  };
  auto path = get_data_path();
  std::string tail_str = path.substr(path.length() - 4);
  if (tail_str != "json") {
    return false;
  }
  std::cout << std::endl << "Begin save feature json, path = " << path << std::endl;

  //   auto idb_insts = dmInst->get_idb_design()->get_instance_list();
  auto idb_nets = dmInst->get_idb_design()->get_net_list();

  json drc_json;
  drc_json["file_path"] = path;
  drc_json["drc"]["number"] = 0;
  int total = 0;  /// drc total number

  json json_distribution;
  std::map<std::string, std::map<std::string, std::vector<ids::Violation>>>& detail_rule_map = drcInst->getDetailCheckResult();

  for (auto& [type, drc_list_map] : detail_rule_map) {
    json json_rule;

    int drc_list_num = 0;
    for (auto& [layer_name, drc_list] : drc_list_map) {
      drc_list_num += drc_list.size();
    }

    json_rule["number"] = drc_list_num;
    json_rule["layers"] = {};

    total += drc_list_num;

    std::map<std::string, json> layer_dict;
    for (auto& [layer_name, drc_list] : drc_list_map) {
      for (auto drc_spot : drc_list) {
        auto& json_layer = get_layer_dict(layer_name, layer_dict);

        json json_drc;

        json_drc["net"] = json::array();
        json_drc["inst"] = json::array();

        for (auto net_id : drc_spot.violation_net_set) {
          if (net_id == NET_ID_VDD || net_id == NET_ID_VSS) {
            /// pdn net
            std::string net_name = net_id == NET_ID_VDD ? "VDD" : "VSS";
            json_drc["net"].push_back(net_name);
          } else {
            /// regular net
            auto net = idb_nets->find_net(net_id);
            if (net != nullptr) {
              json_drc["net"].push_back(net->get_net_name());
            } else {
              json_drc["net"].push_back("-1");  /// save -1 as a blockage
            }
          }
        }

        //   for (auto inst_id : drc_rect->get_inst_ids()) {
        //     auto inst = idb_insts->find_instance(inst_id);
        //     if (inst != nullptr) {
        //       auto inst_name = inst->get_name();
        //       json_drc["inst"].push_back(inst_name);
        //     } else {
        //       json_drc["inst"].push_back("-1");  /// save -1 as a blockage
        //     }
        //   }

        json_drc["llx"] = drc_spot.ll_x;
        json_drc["lly"] = drc_spot.ll_y;
        json_drc["urx"] = drc_spot.ur_x;
        json_drc["ury"] = drc_spot.ur_y;
        json_drc["required_size"] = drc_spot.required_size;

        json_layer["list"].push_back(json_drc);
        int number = json_layer["number"];
        json_layer["number"] = number + 1;
      }
    }

    for (auto& [layer, node] : layer_dict) {
      json_rule["layers"][layer] = node;
    }

    json_distribution[type] = json_rule;
  }

  drc_json["drc"]["number"] = total;
  drc_json["drc"]["distribution"] = json_distribution;

  std::ofstream file_stream(path);
  file_stream << std::setw(4) << drc_json;

  file_stream.close();

  std::cout << std::endl << "Save feature json success, path = " << path << " total violation : " << total << std::endl;
  return true;
}

bool FileDrcManager::readJson()
{
  auto path = get_data_path();

  parseJson(path);

  return true;
}

void FileDrcManager::parseJson(std::string path)
{
  std::map<std::string, std::map<std::string, std::vector<ids::Violation>>>& detail_rule_map = drcInst->getDetailCheckResult();
  detail_rule_map.clear();

  nlohmann::json json;

  ieda::initJson(path, json);

  /// total number
  auto total_drc = ieda::getJsonData(json, {"drc", "number"});
  auto json_distribution = ieda::getJsonData(json, {"drc", "distribution"});

  /// drc distribution
  for (auto& json_drc_type : json_distribution.items()) {
    /// drc type
    auto drc_type = json_drc_type.key();

    /// drc number for all layers
    auto number = json_drc_type.value()["number"];

    /// drc in each layer
    auto& json_layers = json_drc_type.value()["layers"];

    for (auto& json_layer : json_layers.items()) {
      /// layer name
      auto layer = json_layer.key();
      /// layer drc number
      auto number = json_layer.value()["number"];

      /// drc list
      auto& json_drc_list = json_layer.value()["list"];

      for (auto& json_drc : json_drc_list.items()) {
        std::set<int> net_ids;
        auto json_nets = json_drc.value()["net"];
        for (auto& json_net : json_nets.items()) {
          std::string net_name = json_net.value();
          auto net = dmInst->get_idb_design()->get_net_list()->find_net(net_name);
          if (net != nullptr) {
            net_ids.insert(net->get_id());
          } else {
            net_ids.insert(-1);  /// blockage
          }
        }

        std::set<int> inst_ids;
        auto json_insts = json_drc.value()["inst"];
        for (auto& json_inst : json_insts.items()) {
          std::string inst_name = json_inst.value();

          auto inst = dmInst->get_idb_design()->get_instance_list()->find_instance(inst_name);
          if (inst != nullptr) {
            inst_ids.insert(inst->get_id());
          } else {
            inst_ids.insert(-1);  /// blockage
          }
        }

        ids::Violation violation;

        auto* idb_layer = dmInst->get_idb_layout()->get_layers()->find_layer(layer);

        violation.violation_type = drc_type;
        violation.ll_x = json_drc.value()["llx"];
        violation.ll_y = json_drc.value()["lly"];
        violation.ur_x = json_drc.value()["urx"];
        violation.ur_y = json_drc.value()["ury"];
        violation.layer_idx = idb_layer->get_id();
        violation.violation_net_set = net_ids;
        violation.required_size = json_drc.value()["required_size"];

        /// add to spot list
        detail_rule_map[drc_type][idb_layer->get_name()].push_back(violation);
      }
    }
  }
}

}  // namespace iplf
