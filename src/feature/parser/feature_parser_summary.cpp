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
 * @project		iEDA
 * @file		feature_parser.cpp
 * @author		Yell
 * @date		10/08/2023
 * @version		0.1
 * @description


        feature parser
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IdbCore.h"
#include "IdbDesign.h"
#include "IdbDie.h"
#include "IdbEnum.h"
#include "IdbInstance.h"
#include "IdbLayout.h"
#include "IdbNet.h"
#include "IdbRow.h"
#include "IdbTrackGrid.h"
#include "feature_parser.h"
#include "feature_summary.h"
#include "flow_config.h"
#include "idm.h"
#include "iomanip"
#include "json_parser.h"

namespace ieda_feature {

json FeatureParser::buildSummaryInfo()
{
  json node;

  auto& db_summary = _summary->get_db();

  node["eda_tool"] = db_summary.info.eda_tool;
  node["eda_version"] = db_summary.info.eda_version;
  node["design_name"] = db_summary.info.design_name;
  node["design_version"] = db_summary.info.design_version;
  node["flow_stage"] = db_summary.info.flow_stage;
  node["flow_runtime"] = db_summary.info.flow_runtime;
  node["flow_memory"] = db_summary.info.flow_memory;

  return node;
}

json FeatureParser::buildSummaryLayout()
{
  json node;

  auto& db_summary = _summary->get_db();

  node["design_dbu"] = db_summary.layout.design_dbu;

  node["die_area"] = db_summary.layout.die_area;
  node["die_usage"] = db_summary.layout.die_usage;
  node["die_bounding_width"] = db_summary.layout.die_bounding_width;
  node["die_bounding_height"] = db_summary.layout.die_bounding_height;

  node["core_area"] = db_summary.layout.core_area;
  node["core_usage"] = db_summary.layout.core_usage;
  node["core_bounding_width"] = db_summary.layout.core_bounding_width;
  node["core_bounding_height"] = db_summary.layout.core_bounding_height;

  return node;
}

json FeatureParser::buildSummaryStatis()
{
  json node;

  auto& db_summary = _summary->get_db();

  node["num_layers"] = db_summary.statis.num_layers;
  node["num_layers_routing"] = db_summary.statis.num_layers_routing;
  node["num_layers_cut"] = db_summary.statis.num_layers_cut;
  node["num_iopins"] = db_summary.statis.num_iopins;
  node["num_instances"] = db_summary.statis.num_instances;
  node["num_nets"] = db_summary.statis.num_nets;
  node["num_pdn"] = db_summary.statis.num_pdn;

  return node;
}

json FeatureParser::buildSummaryInstances()
{
  json summary_instance;

  auto& db_summary = _summary->get_db();

  json all_instance;
  all_instance["num"] = db_summary.instances.total.num;
  all_instance["num_ratio"] = db_summary.instances.total.num_ratio;
  all_instance["area"] = db_summary.instances.total.area;
  all_instance["area_ratio"] = db_summary.instances.total.area_ratio;
  all_instance["die_usage"] = db_summary.instances.total.die_usage;
  all_instance["core_usage"] = db_summary.instances.total.core_usage;
  all_instance["pin_num"] = db_summary.instances.total.pin_num;
  all_instance["pin_ratio"] = db_summary.instances.total.pin_ratio;
  summary_instance["total"] = all_instance;

  json pad;
  pad["num"] = db_summary.instances.iopads.num;
  pad["num_ratio"] = db_summary.instances.iopads.num_ratio;
  pad["area"] = db_summary.instances.iopads.area;
  pad["area_ratio"] = db_summary.instances.iopads.area_ratio;
  pad["die_usage"] = db_summary.instances.iopads.die_usage;
  pad["core_usage"] = db_summary.instances.iopads.core_usage;
  pad["pin_num"] = db_summary.instances.iopads.pin_num;
  pad["pin_ratio"] = db_summary.instances.iopads.pin_ratio;
  summary_instance["iopads"] = pad;

  json macros;
  macros["num"] = db_summary.instances.macros.num;
  macros["num_ratio"] = db_summary.instances.macros.num_ratio;
  macros["area"] = db_summary.instances.macros.area;
  macros["area_ratio"] = db_summary.instances.macros.area_ratio;
  macros["die_usage"] = db_summary.instances.macros.die_usage;
  macros["core_usage"] = db_summary.instances.macros.core_usage;
  macros["pin_num"] = db_summary.instances.macros.pin_num;
  macros["pin_ratio"] = db_summary.instances.macros.pin_ratio;
  summary_instance["macros"] = macros;

  json core_logic;
  core_logic["num"] = db_summary.instances.logic.num;
  core_logic["num_ratio"] = db_summary.instances.logic.num_ratio;
  core_logic["area"] = db_summary.instances.logic.area;
  core_logic["area_ratio"] = db_summary.instances.logic.area_ratio;
  core_logic["die_usage"] = db_summary.instances.logic.die_usage;
  core_logic["core_usage"] = db_summary.instances.logic.core_usage;
  core_logic["pin_num"] = db_summary.instances.logic.pin_num;
  core_logic["pin_ratio"] = db_summary.instances.logic.pin_ratio;
  summary_instance["logic"] = core_logic;

  json clock;
  clock["num"] = db_summary.instances.clock.num;
  clock["num_ratio"] = db_summary.instances.clock.num_ratio;
  clock["area"] = db_summary.instances.clock.area;
  clock["area_ratio"] = db_summary.instances.clock.area_ratio;
  clock["die_usage"] = db_summary.instances.clock.die_usage;
  clock["core_usage"] = db_summary.instances.clock.core_usage;
  clock["pin_num"] = db_summary.instances.clock.pin_num;
  clock["pin_ratio"] = db_summary.instances.clock.pin_ratio;
  summary_instance["clock"] = clock;

  return summary_instance;
}

json FeatureParser::buildSummaryNets()
{
  json summary_net;

  auto& db_summary = _summary->get_db();

  summary_net["num_total"] = db_summary.nets.num_total;
  summary_net["num_signal"] = db_summary.nets.num_signal;
  summary_net["num_clock"] = db_summary.nets.num_clock;
  summary_net["num_pins"] = db_summary.nets.num_pins;
  summary_net["num_segment"] = db_summary.nets.num_segment;
  summary_net["num_via"] = db_summary.nets.num_via;
  summary_net["num_wire"] = db_summary.nets.num_wire;
  summary_net["num_patch"] = db_summary.nets.num_patch;

  summary_net["wire_len"] = db_summary.nets.wire_len;
  summary_net["wire_len_signal"] = db_summary.nets.wire_len_signal;
  summary_net["ratio_signal"] = db_summary.nets.ratio_signal;
  summary_net["wire_len_clock"] = db_summary.nets.wire_len_clock;
  summary_net["ratio_clock"] = db_summary.nets.ratio_clock;

  return summary_net;
}

json FeatureParser::buildSummaryMacrosStatis()
{
  json summary_macro;

  // eval::EvalAPI& eval_api = eval::EvalAPI::initInst();
  // eval_api.initCongDataFromIDB(256, 256);

  // summary_macro["Channel Util"] = eval_api.evalMacroChannelUtil(0.5);
  // summary_macro["Channel Pin Util"] = eval_api.evalMacroChannelPinRatio(0.5);
  // summary_macro["Max Continuous White Space Ratio"] = eval_api.evalMaxContinuousSpace();
  return summary_macro;
}

json FeatureParser::buildSummaryMacros()
{
  json summary_macro;

  // int dbu = _design->get_units()->get_micron_dbu() < 0 ? _layout->get_units()->get_micron_dbu() : _design->get_units()->get_micron_dbu();

  // eval::EvalAPI& eval_api = eval::EvalAPI::initInst();
  // // eval_api.initCongDataFromIDB(256, 256);

  // auto macro_list = eval_api.evalMacrosInfo();

  // for (int i = 0; i < (int) macro_list.size(); i++) {
  //   summary_macro[i]["Type"] = std::get<std::string>(macro_list[i]["Type"]);
  //   summary_macro[i]["Orient"] = std::get<std::string>(macro_list[i]["Orient"]);
  //   summary_macro[i]["Area"] = std::get<float>(macro_list[i]["Area"]) / dbu / dbu;
  //   summary_macro[i]["Area Ratio"] = std::get<float>(macro_list[i]["Area Ratio"]);
  //   summary_macro[i]["Lx"] = std::get<float>(macro_list[i]["Lx"]) / dbu;
  //   summary_macro[i]["Ly"] = std::get<float>(macro_list[i]["Ly"]) / dbu;
  //   summary_macro[i]["Width"] = std::get<float>(macro_list[i]["Width"]) / dbu;
  //   summary_macro[i]["Height"] = std::get<float>(macro_list[i]["Height"]) / dbu;
  //   summary_macro[i]["#Pins"] = std::get<float>(macro_list[i]["#Pins"]);
  //   summary_macro[i]["Peri Bias"] = std::get<float>(macro_list[i]["Peri Bias"]) / dbu / dbu;
  // }

  return summary_macro;
}

json FeatureParser::buildSummaryLayers()
{
  json json_layer;

  auto& db_summary = _summary->get_db();

  json_layer["num_layers"] = db_summary.layers.num_layers;
  json_layer["num_layers_routing"] = db_summary.layers.num_layers_routing;
  json_layer["num_layers_cut"] = db_summary.layers.num_layers_cut;

  json json_layer_routing;
  for (int i = 0; i < (int) db_summary.layers.routing_layers.size(); i++) {
    json_layer_routing[i]["layer_name"] = db_summary.layers.routing_layers[i].layer_name;
    json_layer_routing[i]["layer_order"] = db_summary.layers.routing_layers[i].layer_order;
    json_layer_routing[i]["wire_len"] = db_summary.layers.routing_layers[i].wire_len;
    json_layer_routing[i]["wire_ratio"] = db_summary.layers.routing_layers[i].wire_ratio;
    json_layer_routing[i]["wire_num"] = db_summary.layers.routing_layers[i].wire_num;
    json_layer_routing[i]["patch_num"] = db_summary.layers.routing_layers[i].patch_num;
  }
  json_layer["routing_layers"] = json_layer_routing;

  json json_layer_cut;
  for (int i = 0; i < (int) db_summary.layers.cut_layers.size(); i++) {
    json_layer_cut[i]["layer_name"] = db_summary.layers.cut_layers[i].layer_name;
    json_layer_cut[i]["layer_order"] = db_summary.layers.cut_layers[i].layer_order;
    json_layer_cut[i]["via_num"] = db_summary.layers.cut_layers[i].via_num;
    json_layer_cut[i]["via_ratio"] = db_summary.layers.cut_layers[i].via_ratio;
  }
  json_layer["cut_layers"] = json_layer_cut;

  return json_layer;
}

json FeatureParser::buildSummaryPdn()
{
  json node;

  return node;
}

json FeatureParser::buildSummaryPins()
{
  json json_pins;

  auto& db_summary = _summary->get_db();
  json_pins["max_fanout"] = db_summary.pins.max_fanout;

  json json_distribution;
  for (int i = 0; i < (int) db_summary.pins.pin_distribution.size(); i++) {
    if (db_summary.pins.pin_distribution[i].pin_num > db_summary.pins.max_fanout) {
      json_distribution[i]["pin_num"] = "> 32";
    } else {
      json_distribution[i]["pin_num"] = db_summary.pins.pin_distribution[i].pin_num;
    }
    json_distribution[i]["net_num"] = db_summary.pins.pin_distribution[i].net_num;
    json_distribution[i]["net_ratio"] = db_summary.pins.pin_distribution[i].net_ratio;
    json_distribution[i]["inst_num"] = db_summary.pins.pin_distribution[i].inst_num;
    json_distribution[i]["inst_ratio"] = db_summary.pins.pin_distribution[i].inst_ratio;
  }

  json_pins["pin_distribution"] = json_distribution;

  return json_pins;
}

}  // namespace ieda_feature
