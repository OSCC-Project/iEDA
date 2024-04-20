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

#include "CTSAPI.hh"
#include "EvalAPI.hpp"
#include "Evaluator.hh"
#include "IdbCore.h"
#include "IdbDesign.h"
#include "IdbDie.h"
#include "IdbEnum.h"
#include "IdbInstance.h"
#include "IdbLayout.h"
#include "IdbNet.h"
#include "IdbRow.h"
#include "IdbTrackGrid.h"
#include "PLAPI.hh"
#include "PlacerDB.hh"
#include "RTInterface.hpp"
#include "TimingEngine.hh"
#include "ToApi.hpp"
#include "feature_parser.h"
#include "flow_config.h"
#include "idm.h"
#include "iomanip"
#include "json_parser.h"
#include "report_evaluator.h"
#include "summary.h"

namespace idb {

bool FeatureParser::buildReportSummary(std::string json_path, std::string step)
{
  std::ofstream& file_stream = ieda::getOutputFileStream(json_path);
  json root;

  root["Design Information"] = buildSummaryInfo();

  root["Design Layout"] = buildSummaryLayout();

  root["Design Statis"] = buildSummaryStatis();

  root["Instances"] = buildSummaryInstances();

  root["Macros Statis"] = buildSummaryMacrosStatis();

  root["Macros"] = buildSummaryMacros();

  root["Nets"] = buildSummaryNets();

  root["PDN"] = buildSummaryPdn();

  root["Layers"] = buildSummaryLayers();

  root["Pins"] = buildSummaryPins();

  // root["Place"] = buildSummaryPL(json_path);
  // root["CTS"] = buildSummaryCTS();
  // root["DRC"] = buildSummaryDRC();
  // root["TO"] = buildSummaryTO();
  if (!step.empty())
    root[step] = flowSummary(step);

  file_stream << std::setw(4) << root;

  ieda::closeFileStream(file_stream);

  std::cout << std::endl << "Save feature json success, path = " << json_path << std::endl;
  return true;
}

bool FeatureParser::buildReportSummaryMap(std::string csv_path, int bin_cnt_x, int bin_cnt_y)
{
  eval::EvalAPI& eval_api = eval::EvalAPI::initInst();
  eval_api.initCongDataFromIDB(bin_cnt_x, bin_cnt_y);

  auto inst_status = eval::INSTANCE_STATUS::kFixed;
  eval_api.evalInstDens(inst_status);
  eval_api.plotBinValue(csv_path, "macro_density", eval::CONGESTION_TYPE::kInstDens);
  eval_api.evalPinDens(inst_status);
  eval_api.plotBinValue(csv_path, "macro_pin_density", eval::CONGESTION_TYPE::kPinDens);
  eval_api.evalNetDens(inst_status);
  eval_api.plotBinValue(csv_path, "macro_net_density", eval::CONGESTION_TYPE::kNetCong);

  eval_api.plotMacroChannel(0.5, csv_path+"macro_channel.csv" );
  eval_api.evalMacroMargin();
  eval_api.plotBinValue(csv_path, "macro_margin_h", eval::CONGESTION_TYPE::kMacroMarginH);
  eval_api.plotBinValue(csv_path, "macro_margin_v", eval::CONGESTION_TYPE::kMacroMarginV);
  double space_ratio = eval_api.evalMaxContinuousSpace();
  eval_api.plotBinValue(csv_path, "macro_continuous_white_space", eval::CONGESTION_TYPE::kContinuousWS);
  eval_api.evalIOPinAccess( csv_path+"io_pin_access.csv");

  std::cout << std::endl << "Save feature map success, path = " << csv_path << std::endl;
  return true;
}


json FeatureParser::buildSummaryInfo()
{
  json node;

  node["EDA Tool"] = "iEDA";
  node["EDA Version"] = iplf::flowConfigInst->get_env_info_software_version();
  node["Design Name"] = _design->get_design_name();
  node["Design Version"] = _design->get_version();
  node["Design DBU"] = _design->get_units()->get_micron_dbu();
  node["Flow Stage"] = iplf::flowConfigInst->get_status_stage();
  node["Flow Runtime"] = iplf::flowConfigInst->get_status_runtime_string();
  node["Flow Memmory"] = iplf::flowConfigInst->get_status_memmory_string();

  return node;
}

json FeatureParser::buildSummaryLayout()
{
  json node;

  int dbu = _design->get_units()->get_micron_dbu() < 0 ? _layout->get_units()->get_micron_dbu() : _design->get_units()->get_micron_dbu();

  node["DIE Area ( um^2 )"] = dmInst->dieAreaUm();
  node["DIE Usage"] = dmInst->dieUtilization();
  node["DIE Width ( um )"] = ((double) _layout->get_die()->get_width()) / dbu;
  node["DIE Height ( um )"] = ((double) _layout->get_die()->get_height()) / dbu;

  node["CORE Area ( um^2 )"] = dmInst->coreAreaUm();
  node["CORE Usage"] = dmInst->coreUtilization();
  node["CORE Width ( um )"] = ((double) _layout->get_core()->get_bounding_box()->get_width()) / dbu;
  node["CORE Height ( um )"] = ((double) _layout->get_core()->get_bounding_box()->get_height()) / dbu;

  return node;
}

json FeatureParser::buildSummaryStatis()
{
  json node;

  /// layers
  node["Layers"] = _design->get_layout()->get_layers()->get_layers_num();
  node["Routing Layers"] = _design->get_layout()->get_layers()->get_routing_layers_number();
  node["Cut Layers"] = _design->get_layout()->get_layers()->get_cut_layers_number();

  /// IOs
  node["IO Pins"] = _design->get_io_pin_list()->get_pin_num();

  /// instances
  node["Instances"] = _design->get_instance_list()->get_num();
  //   node["Instances - Macros"] = _design->get_instance_list()->get_num_block();
  //   node["Instances - Pads"] = _design->get_instance_list()->get_num_pad();
  //   node["Instances - Physical"] = _design->get_instance_list()->get_num_physics();
  //   node["Instances - Endcaps"] = _design->get_instance_list()->get_num_endcap();
  //   node["Instances - Tapcells"] = _design->get_instance_list()->get_num_tapcell();

  node["Fillers"] = _design->get_fill_list()->get_num_fill();
  node["Blockages"] = _design->get_blockage_list()->get_num();

  /// nets
  node["Nets"] = _design->get_net_list()->get_num();

  /// pdn
  node["Special Nets"] = _design->get_special_net_list()->get_num();

  return node;
}

json FeatureParser::buildSummaryInstances()
{
  json summary_instance;

  int dbu = _design->get_units()->get_micron_dbu() < 0 ? _layout->get_units()->get_micron_dbu() : _design->get_units()->get_micron_dbu();

  double insts_area = dmInst->instanceArea(IdbInstanceType::kMax);
  int64_t inst_num = _design->get_instance_list()->get_num();

  json all_instance;
  all_instance["Number"] = inst_num;
  all_instance["Number Ratio"] = (double) (1);
  all_instance["Area ( um^2 )"] = insts_area;
  all_instance["Area Ratio"] = (double) (1);
  summary_instance["All Instances"] = all_instance;

  //   json netlist;
  //   netlist["Number"] = _design->get_instance_list()->get_num(IdbInstanceType::kNetlist);
  //   netlist["Number Ratio"]
  //       = ((double) _design->get_instance_list()->get_num(IdbInstanceType::kNetlist)) / _design->get_instance_list()->get_num();
  //   netlist["Area ( um^2 )"] = dmInst->netlistInstArea();
  //   netlist["Area Ratio"] = ((double) dmInst->netlistInstArea()) / insts_area;
  //   summary_instance["Netlist"] = netlist;

  //   json physical;
  //   physical["Number"] = _design->get_instance_list()->get_num(IdbInstanceType::kDist);
  //   physical["Number Ratio"]
  //       = ((double) _design->get_instance_list()->get_num(IdbInstanceType::kDist)) / _design->get_instance_list()->get_num();
  //   physical["Area ( um^2 )"] = dmInst->distInstArea();
  //   physical["Area Ratio"] = ((double) dmInst->distInstArea()) / insts_area;
  //   summary_instance["Physical"] = physical;

  //   json timing;
  //   timing["Number"] = _design->get_instance_list()->get_num(IdbInstanceType::kTiming);
  //   timing["Number Ratio"]
  //       = ((double) _design->get_instance_list()->get_num(IdbInstanceType::kTiming)) / _design->get_instance_list()->get_num();
  //   timing["Area ( um^2 )"] = dmInst->timingInstArea();
  //   timing["Area Ratio"] = ((double) dmInst->timingInstArea()) / insts_area;
  //   summary_instance["Timing"] = timing;

  json core;
  double core_area = ((double) _design->get_instance_list()->get_area_core()) / dbu / dbu;
  int64_t core_num = _design->get_instance_list()->get_num_core();

  core["Number"] = core_num;
  core["Number Ratio"] = ((double) core_num) / inst_num;
  core["Area ( um^2 )"] = core_area;
  core["Area Ratio"] = core_area / insts_area;
  summary_instance["Core"] = core;

  json core_logic;
  double core_logic_area = ((double) _design->get_instance_list()->get_area_core_logic()) / dbu / dbu;
  int64_t core_logic_num = _design->get_instance_list()->get_num_core_logic();
  core_logic["Number"] = core_logic_num;
  core_logic["Number Ratio"] = ((double) core_logic_num) / inst_num;
  core_logic["Area ( um^2 )"] = core_logic_area;
  core_logic["Area Ratio"] = core_logic_area / insts_area;
  summary_instance["Core - logic"] = core_logic;

  json pad;
  double pad_area = ((double) _design->get_instance_list()->get_area_pad()) / dbu / dbu;
  int64_t pad_num = _design->get_instance_list()->get_num_pad();
  pad["Number"] = pad_num;
  pad["Number Ratio"] = ((double) pad_num) / inst_num;
  pad["Area ( um^2 )"] = pad_area;
  pad["Area Ratio"] = pad_area / insts_area;
  summary_instance["Pad"] = pad;

  json block;
  double block_area = ((double) _design->get_instance_list()->get_area_block()) / dbu / dbu;
  int64_t block_num = _design->get_instance_list()->get_num_block();
  block["Number"] = block_num;
  block["Number Ratio"] = ((double) block_num) / inst_num;
  block["Area ( um^2 )"] = block_area;
  block["Area Ratio"] = block_area / insts_area;
  summary_instance["Block"] = block;

  json endcap;
  double endcap_area = ((double) _design->get_instance_list()->get_area_endcap()) / dbu / dbu;
  int64_t endcap_num = _design->get_instance_list()->get_num_endcap();
  endcap["Number"] = endcap_num;
  endcap["Number Ratio"] = ((double) endcap_num) / inst_num;
  endcap["Area ( um^2 )"] = endcap_area;
  endcap["Area Ratio"] = endcap_area / insts_area;
  summary_instance["Endcap"] = endcap;

  json tapcell;
  double tapcell_area = ((double) _design->get_instance_list()->get_area_tapcell()) / dbu / dbu;
  int64_t tapcell_num = _design->get_instance_list()->get_num_tapcell();
  tapcell["Number"] = tapcell_num;
  tapcell["Number Ratio"] = ((double) tapcell_num) / inst_num;
  tapcell["Area ( um^2 )"] = tapcell_area;
  tapcell["Area Ratio"] = tapcell_area / insts_area;
  summary_instance["tapcell"] = tapcell;

  json cover;
  double cover_area = ((double) _design->get_instance_list()->get_area_cover()) / dbu / dbu;
  int64_t cover_num = _design->get_instance_list()->get_num_cover();
  cover["Number"] = cover_num;
  cover["Number Ratio"] = ((double) cover_num) / inst_num;
  cover["Area ( um^2 )"] = cover_area;
  cover["Area Ratio"] = cover_area / insts_area;
  summary_instance["Cover"] = cover;

  json ring;
  double ring_area = ((double) _design->get_instance_list()->get_area_ring()) / dbu / dbu;
  int64_t ring_num = _design->get_instance_list()->get_num_ring();
  ring["Number"] = ring_num;
  ring["Number Ratio"] = ((double) ring_num) / inst_num;
  ring["Area ( um^2 )"] = ring_area;
  ring["Area Ratio"] = ring_area / insts_area;
  summary_instance["Ring"] = ring;

  return summary_instance;
}

json FeatureParser::buildSummaryMacrosStatis()
{
  json summary_macro;

  eval::EvalAPI& eval_api = eval::EvalAPI::initInst();
  eval_api.initCongDataFromIDB(256, 256);

  summary_macro["Channel Util"] = eval_api.evalMacroChannelUtil(0.5);
  summary_macro["Channel Pin Util"] = eval_api.evalMacroChannelPinRatio(0.5);
  summary_macro["Max Continuous White Space Ratio"] = eval_api.evalMaxContinuousSpace();
  return summary_macro;
}

json FeatureParser::buildSummaryMacros()
{
  json summary_macro;

  int64_t block_num = _design->get_instance_list()->get_num_block();
  int dbu = _design->get_units()->get_micron_dbu() < 0 ? _layout->get_units()->get_micron_dbu() : _design->get_units()->get_micron_dbu();

  eval::EvalAPI& eval_api = eval::EvalAPI::initInst();
  // eval_api.initCongDataFromIDB(256, 256);

  auto macro_list = eval_api.evalMacrosInfo();

  for (int i = 0; i < block_num; i++) {
    summary_macro[i]["Type"] = std::get<std::string>(macro_list[i]["Type"]);
    summary_macro[i]["Orient"] = std::get<std::string>(macro_list[i]["Orient"]);
    summary_macro[i]["Area"] = std::get<float>(macro_list[i]["Area"]) / dbu / dbu ;
    summary_macro[i]["Area Ratio"] =  std::get<float>(macro_list[i]["Area Ratio"]);
    summary_macro[i]["Lx"] =  std::get<float>(macro_list[i]["Lx"]) / dbu;
    summary_macro[i]["Ly"] =  std::get<float>(macro_list[i]["Ly"]) / dbu;
    summary_macro[i]["Width"] =  std::get<float>(macro_list[i]["Width"])  / dbu;
    summary_macro[i]["Height"] =  std::get<float>(macro_list[i]["Height"])  / dbu;
    summary_macro[i]["#Pins"] =  std::get<float>(macro_list[i]["#Pins"]) ;
    summary_macro[i]["Peri Bias"] = std::get<float>(macro_list[i]["Peri Bias"]) / dbu / dbu;
  }

  return summary_macro;
}



json FeatureParser::buildSummaryLayers()
{
  json summary_layer;

  struct SummaryLayerValue
  {
    std::string layer_name;
    int32_t layer_order;
    uint64_t wire_len;
    uint64_t seg_num;
    uint64_t wire_num;
    uint64_t via_num;
    uint64_t patch_num;
    int32_t type;
  };

  int dbu = _design->get_units()->get_micron_dbu() < 0 ? _layout->get_units()->get_micron_dbu() : _design->get_units()->get_micron_dbu();

  std::vector<SummaryLayerValue> layer_net_value_list;
  std::vector<SummaryLayerValue> layer_specialnet_value_list;
  for (auto layer : _layout->get_layers()->get_layers()) {
    SummaryLayerValue layer_value;
    layer_value.layer_name = layer->get_name();
    layer_value.layer_order = layer->get_order();
    layer_value.wire_len = 0;
    layer_value.seg_num = 0;
    layer_value.wire_num = 0;
    layer_value.via_num = 0;
    layer_value.patch_num = 0;
    if (layer->is_routing()) {
      layer_value.type = 1;
    } else if (layer->is_cut()) {
      layer_value.type = 2;
    } else {
      layer_value.type = 0;
    }

    layer_net_value_list.push_back(layer_value);
    layer_specialnet_value_list.push_back(layer_value);
  }

  for (auto net : _design->get_net_list()->get_net_list()) {
    for (auto wire : net->get_wire_list()->get_wire_list()) {
      for (auto segment : wire->get_segment_list()) {
        size_t order = segment->get_layer()->get_order();
        if (order >= layer_net_value_list.size()) {
          continue;
        }

        if (segment->is_wire()) {
          layer_net_value_list[order].seg_num += 1;
          layer_net_value_list[order].wire_num += 1;
          layer_net_value_list[order].wire_len += segment->length();
        }

        if (segment->is_rect()) {
          layer_net_value_list[order].seg_num += 1;
          layer_net_value_list[order].patch_num += 1;
          layer_net_value_list[order].wire_len += segment->length();
        }

        if (segment->is_via()) {
          auto via_list = segment->get_via_list();
          for (auto via : via_list) {
            auto layer_shape = via->get_cut_layer_shape();

            order = layer_shape.get_layer()->get_order();
            layer_net_value_list[order].seg_num += 1;
            layer_net_value_list[order].via_num += 1;
          }
        }
      }
    }
  }

  for (auto special_net : _design->get_special_net_list()->get_net_list()) {
    for (auto special_wire : special_net->get_wire_list()->get_wire_list()) {
      for (auto special_segment : special_wire->get_segment_list()) {
        size_t order = special_segment->get_layer()->get_order();
        if (order < 0 || order >= layer_specialnet_value_list.size()) {
          continue;
        }

        if (special_segment->is_line()) {
          layer_specialnet_value_list[order].seg_num += 1;
          layer_specialnet_value_list[order].wire_num += 1;
          layer_specialnet_value_list[order].wire_len += special_segment->length();
        }

        if (special_segment->is_via()) {
          auto via = special_segment->get_via();
          auto layer_shape = via->get_cut_layer_shape();
          order = layer_shape.get_layer()->get_order();
          layer_specialnet_value_list[order].seg_num += 1;
          layer_specialnet_value_list[order].via_num += 1;
        }
      }
    }
  }
  int layer_num = _layout->get_layers()->get_layers().size();
  uint64_t all_nets_length = dmInst->allNetLength();

  for (int i = 0; i < layer_num; i++) {
    auto net_value = layer_net_value_list[i];
    auto special_net_value = layer_specialnet_value_list[i];
    if (net_value.type == 1) {
      /// routing
      summary_layer[net_value.layer_name]["Net - Wire Length (um)"] = ((double) net_value.wire_len) / dbu;
      summary_layer[net_value.layer_name]["Net - Wire Ratio"] = ((double) net_value.wire_len) / all_nets_length;
      summary_layer[net_value.layer_name]["Net - Wire Number"] = net_value.wire_num;
      summary_layer[net_value.layer_name]["Net - Patch Number"] = net_value.patch_num;
      summary_layer[net_value.layer_name]["Special Net - Wire Length (um)"] = ((double) special_net_value.wire_len) / dbu;
      summary_layer[net_value.layer_name]["Special Net - Wire Number"] = special_net_value.wire_num;
    } else if (net_value.type == 2) {
      /// cut
      summary_layer[net_value.layer_name]["Net - Via Number"] = net_value.via_num;
      summary_layer[net_value.layer_name]["Special Net - Via Number"] = special_net_value.via_num;
    } else {
      /// do nothing
      continue;
    }
  }

  return summary_layer;
}

json FeatureParser::buildSummaryNets()
{
  json summary_net;

  int dbu = _design->get_units()->get_micron_dbu() < 0 ? _layout->get_units()->get_micron_dbu() : _design->get_units()->get_micron_dbu();
  uint64_t all_nets_length = dmInst->allNetLength();

  json all_net;
  all_net["Number"] = _design->get_net_list()->get_num();
  all_net["Number Ratio"] = 1;
  all_net["Length (um)"] = ((double) all_nets_length) / dbu;
  all_net["Length Ratio"] = all_nets_length == 0 ? 0 : 1;
  summary_net["All Nets"] = all_net;

  json signal;
  signal["Number"] = _design->get_net_list()->get_num_signal();
  signal["Number Ratio"] = ((double) _design->get_net_list()->get_num_signal()) / _design->get_net_list()->get_num();
  signal["Length (um)"] = ((double) dmInst->getSignalNetListLength()) / dbu;
  signal["Length Ratio"] = all_nets_length == 0 ? 0 : ((double) dmInst->getSignalNetListLength()) / all_nets_length;
  summary_net["Signal"] = signal;

  json clock;
  clock["Number"] = _design->get_net_list()->get_num_clock();
  clock["Number Ratio"] = ((double) _design->get_net_list()->get_num_clock()) / _design->get_net_list()->get_num();
  clock["Length (um)"] = ((double) dmInst->getClockNetListLength()) / dbu;
  clock["Length Ratio"] = all_nets_length == 0 ? 0 : ((double) dmInst->getClockNetListLength()) / all_nets_length;
  summary_net["Clock"] = clock;

  json pdn;
  pdn["Number"] = _design->get_net_list()->get_num_pdn();
  pdn["Number Ratio"] = ((double) _design->get_net_list()->get_num_pdn()) / _design->get_net_list()->get_num();
  pdn["Length (um)"] = ((double) dmInst->getPdnNetListLength()) / dbu;
  pdn["Length Ratio"] = all_nets_length == 0 ? 0 : ((double) dmInst->getPdnNetListLength()) / all_nets_length;
  summary_net["Power & Ground"] = pdn;

  json io_pin_nets;
  io_pin_nets["Length (um)"] = ((double) dmInst->getIONetListLength()) / dbu;
  io_pin_nets["Length Ratio"] = all_nets_length == 0 ? 0 : ((double) dmInst->getIONetListLength()) / all_nets_length;
  summary_net["Nets with IO"] = io_pin_nets;

  return summary_net;
}

json FeatureParser::buildSummaryPdn()
{
  json node;

  return node;
}

json FeatureParser::buildSummaryPins()
{
  json summary_pin;

  const int max_num = 34;
  const int max_fanout = 32;

  int instance_total = _design->get_instance_list()->get_instance_list().size();
  int net_total = _design->get_net_list()->get_net_list().size();

  std::vector<int> net_array(max_num, 0);
  for (auto net : _design->get_net_list()->get_net_list()) {
    auto pin_num = net->get_pin_number();
    if (pin_num >= 0 && pin_num <= max_fanout) {
      net_array[pin_num] += 1;
    } else {
      net_array[max_num - 1] += 1;
    }
  }

  std::vector<int> inst_array(max_num, 0);
  for (auto instance : _design->get_instance_list()->get_instance_list()) {
    auto pin_num = instance->get_logic_pin_num();
    if (pin_num >= 0 && pin_num <= max_fanout) {
      inst_array[pin_num] += 1;
    } else {
      inst_array[max_num - 1] += 1;
    }
  }
  for (int i = 0; i <= max_fanout; i++) {
    summary_pin[i]["Pin Number"] = i;
    summary_pin[i]["Net Number"] = net_array[i];
    summary_pin[i]["Net Ratio"] = ((double) net_array[i]) / net_total;
    summary_pin[i]["Instance Number"] = inst_array[i];
    summary_pin[i]["Instance Ratio"] = ((double) inst_array[i]) / instance_total;
  }
  summary_pin[max_fanout + 1]["Pin Number"] = ieda::Str::printf(">= %d ", max_fanout);
  summary_pin[max_fanout + 1]["Net Number"] = net_array[max_num - 1];
  summary_pin[max_fanout + 1]["Net Ratio"] = ((double) net_array[max_num - 1]) / net_total;
  summary_pin[max_fanout + 1]["Instance Number"] = inst_array[max_num - 1];
  summary_pin[max_fanout + 1]["Instance Ratio"] = ((double) inst_array[max_num - 1]) / instance_total;

  return summary_pin;
}
/**
 * if step = "", only save idb summary
 */
json FeatureParser::flowSummary(std::string step)
{
  using SummaryBuilder = std::function<json()>;
  auto stepToBuilder = std::unordered_map<std::string, SummaryBuilder>{{"place", [this, step]() { return buildSummaryPL(step); }},
                                                                       {"legalization", [this, step]() { return buildSummaryPL(step); }},
                                                                       {"CTS", [this]() { return buildSummaryCTS(); }},
                                                                       {"optDrv", [this, step]() { return buildSummaryTO(step); }},
                                                                       {"optHold", [this, step]() { return buildSummaryTO(step); }},
                                                                       {"optSetup", [this, step]() { return buildSummaryTO(step); }},
                                                                       {"sta", [this]() { return buildSummarySTA(); }},
                                                                       {"drc", [this]() { return buildSummaryDRC(); }},
                                                                       {"route", [this]() { return buildSummaryRT(); }}};

  return stepToBuilder[step]();
}

json FeatureParser::buildSummaryPL(std::string step)
{
  json summary_pl;
  // 1:全局布局、详细布局、合法化都需要存储的数据参数，需要根据step存储不同的值
  auto place_density = PlacerDBInst.place_density;
  auto pin_density = PlacerDBInst.pin_density;
  auto HPWL = PlacerDBInst.PL_HPWL;
  auto STWL = PlacerDBInst.PL_STWL;
  auto GRWL = PlacerDBInst.PL_GRWL;
  auto congestion = PlacerDBInst.congestion;
  auto tns = PlacerDBInst.tns;
  auto wns = PlacerDBInst.wns;
  auto suggest_freq = PlacerDBInst.suggest_freq;

  // 2:全局布局、详细布局需要存储的数据参数
  if (step == "place") {
    summary_pl["gplace"]["place_density"] = place_density[0];
    summary_pl["gplace"]["pin_density"] = pin_density[0];
    summary_pl["gplace"]["HPWL"] = HPWL[0];
    summary_pl["gplace"]["STWL"] = STWL[0];
    summary_pl["gplace"]["global_routing_WL"] = GRWL[0];
    summary_pl["gplace"]["congestion"] = congestion[0];
    summary_pl["gplace"]["tns"] = tns[0];
    summary_pl["gplace"]["wns"] = wns[0];
    summary_pl["gplace"]["suggest_freq"] = suggest_freq[0];

    summary_pl["dplace"]["place_density"] = place_density[1];
    summary_pl["dplace"]["pin_density"] = pin_density[1];
    summary_pl["dplace"]["HPWL"] = HPWL[1];
    summary_pl["dplace"]["STWL"] = STWL[1];
    summary_pl["dplace"]["global_routing_WL"] = GRWL[1];
    summary_pl["dplace"]["congestion"] = congestion[1];
    summary_pl["dplace"]["tns"] = tns[1];
    summary_pl["dplace"]["wns"] = wns[1];
    summary_pl["dplace"]["suggest_freq"] = suggest_freq[1];

    auto* pl_design = PlacerDBInst.get_design();
    summary_pl["instance"] = pl_design->get_instances_range();
    int fix_inst_cnt = 0;
    for (auto* inst : pl_design->get_instance_list()) {
      if (inst->isFixed()) {
        fix_inst_cnt++;
      }
    }

    summary_pl["fix_instances"] = fix_inst_cnt;
    summary_pl["nets"] = pl_design->get_nets_range();
    summary_pl["total_pins"] = pl_design->get_pins_range();
    summary_pl["core_area"] = std::to_string(PlacerDBInst.get_layout()->get_core_shape().get_width()) + " * "
                              + std::to_string(PlacerDBInst.get_layout()->get_core_shape().get_height());

    summary_pl["bin_number"] = PlacerDBInst.get_placer_config()->get_nes_config().get_bin_cnt_x()
                               * PlacerDBInst.get_placer_config()->get_nes_config().get_bin_cnt_y();
    summary_pl["bin_size"] = std::to_string(PlacerDBInst.bin_size_x) + " * " + std::to_string(PlacerDBInst.bin_size_y);
    summary_pl["overflow_number"] = PlacerDBInst.gp_overflow_number;
    summary_pl["overflow"] = PlacerDBInst.gp_overflow;
  }
  // 3:合法化需要存储的数据参数
  else if (step == "legalization") {
    summary_pl["legalization"]["place_density"] = place_density[2];
    summary_pl["legalization"]["pin_density"] = pin_density[2];
    summary_pl["legalization"]["HPWL"] = HPWL[2];
    summary_pl["legalization"]["STWL"] = STWL[2];
    summary_pl["legalization"]["global_routing_WL"] = GRWL[2];
    summary_pl["legalization"]["congestion"] = congestion[2];
    summary_pl["legalization"]["tns"] = tns[2];
    summary_pl["legalization"]["wns"] = wns[2];
    summary_pl["legalization"]["suggest_freq"] = suggest_freq[2];

    summary_pl["total_movement"] = PlacerDBInst.lg_total_movement;
    summary_pl["max_movement"] = PlacerDBInst.lg_max_movement;
  }
  // std::ofstream& file_stream = ieda::getOutputFileStream(json_path);
  // file_stream << std::setw(4) << summary_pl;

  // ieda::closeFileStream(file_stream);

  // std::cout << std::endl << "Save feature json success, path = " << json_path << std::endl;

  return summary_pl;
}

json FeatureParser::buildSummaryCTS()
{
  // get CTS data
  json summary_cts;

  CTSAPIInst.initEvalInfo();
  summary_cts["design_area"] = dmInst->dieAreaUm();
  summary_cts["design_utilization"] = dmInst->dieUtilization();

  summary_cts["clock_buffer"] = CTSAPIInst.getInsertCellNum();
  summary_cts["clock_buffer_area"] = CTSAPIInst.getInsertCellArea();
  summary_cts["clock_nets"] = _design->get_net_list()->get_num_clock();
  auto path_info = CTSAPIInst.getPathInfos();
  int max_path = path_info[0].max_depth;
  int min_path = path_info[0].min_depth;

  for (auto path : path_info) {
    max_path = std::max(max_path, path.max_depth);
    min_path = std::min(min_path, path.min_depth);
  }
  auto max_level_of_clock_tree = max_path;

  summary_cts["clock_path_min_buffer"] = min_path;
  summary_cts["clock_path_max_buffer"] = max_path;
  summary_cts["max_level_of_clock_tree"] = max_level_of_clock_tree;
  summary_cts["max_clock_wirelength"] = CTSAPIInst.getMaxClockNetWL();
  summary_cts["total_clock_wirelength"] = CTSAPIInst.getTotalClockNetWL();
  // CTSAPIInst.startDbSta();
  auto _timing_engine = ista::TimingEngine::getOrCreateTimingEngine();
  // 可能有多个clk_name，每一个时钟都需要报告tns、wns、freq
  auto clk_list = _timing_engine->getClockList();
  std::ranges::for_each(clk_list, [&](ista::StaClock* clk) {
    auto clk_name = clk->get_clock_name();
    auto setup_tns = _timing_engine->reportTNS(clk_name, AnalysisMode::kMax);
    auto setup_wns = _timing_engine->reportWNS(clk_name, AnalysisMode::kMax);
    auto hold_tns = _timing_engine->reportTNS(clk_name, AnalysisMode::kMin);
    auto hold_wns = _timing_engine->reportWNS(clk_name, AnalysisMode::kMin);
    auto suggest_freq = 1000.0 / (clk->getPeriodNs() - setup_wns);
    summary_cts[clk_name]["setup_tns"] = setup_tns;
    summary_cts[clk_name]["setup_wns"] = setup_wns;
    summary_cts[clk_name]["hold_tns"] = hold_tns;
    summary_cts[clk_name]["hold_wns"] = hold_wns;
    summary_cts[clk_name]["suggest_freq"] = suggest_freq;
  });

  return summary_cts;
}

json FeatureParser::buildSummaryTO(std::string step)
{
  json summary_to;

#if 1
  // instances, nets, total_pins, core_area, utilization
  // 这些指标在summary里都有
  summary_to["instances"] = _design->get_instance_list()->get_num();
  summary_to["nets"] = _design->get_net_list()->get_num();
  // summary_to["total_pins"] =
  summary_to["core_area"] = dmInst->coreAreaUm();
  summary_to["utilization"] = dmInst->coreUtilization();
#endif

  // HPWL, STWL, Global_routing_WL, congestion
  auto& nets = dmInst->get_idb_design()->get_net_list()->get_net_list();
  auto wl_nets = iplf::EvalWrapper::parallelWrap<eval::WLNet>(nets, iplf::EvalWrapper::wrapWLNet);
  summary_to["HPWL"] = EvalInst.evalTotalWL("kHPWL", wl_nets);
  summary_to["STWL"] = EvalInst.evalTotalWL("kFlute", wl_nets);
  // auto Global_routing_WL =
  // auto congestion =

  // max_fanout, min_slew_slack, min_cap_slack

  // before: 初始值，tns，wns，freq
  json summary_subto;
  auto to_eval_data = ToApiInst.getEvalData();
  for (auto eval_data : to_eval_data) {
    auto clk_name = eval_data.name;
    summary_subto[clk_name]["initial_tns"] = eval_data.initial_tns;
    summary_subto[clk_name]["initial_wns"] = eval_data.initial_wns;
    summary_subto[clk_name]["initial_suggest_freq"] = eval_data.initial_freq;
  }

  // after: 优化后的值
  auto _timing_engine = ista::TimingEngine::getOrCreateTimingEngine();
  auto clk_list = _timing_engine->getClockList();

  std::ranges::for_each(clk_list, [&](ista::StaClock* clk) {
    auto clk_name = clk->get_clock_name();
    auto drv_tns = _timing_engine->reportTNS(clk_name, AnalysisMode::kMax);
    auto drv_wns = _timing_engine->reportWNS(clk_name, AnalysisMode::kMax);
    auto suggest_freq = 1000.0 / (clk->getPeriodNs() - drv_wns);
    summary_subto[clk_name]["optimized_tns"] = drv_tns;
    summary_subto[clk_name]["optimized_wns"] = drv_wns;
    summary_subto[clk_name]["optimized_suggest_freq"] = suggest_freq;
  });

  // delta: 迭代的值，优化后的值减去初始值
  for (auto eval_data : to_eval_data) {
    auto clk_name = eval_data.name;
    summary_subto[clk_name]["delta_tns"]
        = static_cast<double>(summary_subto[clk_name]["optimized_tns"]) - static_cast<double>(summary_subto[clk_name]["initial_tns"]);
    summary_subto[clk_name]["delta_wns"]
        = static_cast<double>(summary_subto[clk_name]["optimized_wns"]) - static_cast<double>(summary_subto[clk_name]["initial_wns"]);
    summary_subto[clk_name]["delta_suggest_freq"] = static_cast<double>(summary_subto[clk_name]["optimized_suggest_freq"])
                                                    - static_cast<double>(summary_subto[clk_name]["initial_suggest_freq"]);
  }

  summary_to["sta"] = summary_subto;

  return summary_to;
}

json FeatureParser::buildSummarySTA()
{
  json summary_sta;
  auto timing_engine = ista::TimingEngine::getOrCreateTimingEngine();
  auto* ista = timing_engine->get_ista();
  auto& all_clocks = ista->get_clocks();

  // iterate the clock group of all the clocks.
  for (unsigned id = 1; auto& clock_group : all_clocks) {
    json::value_type path_group;
    path_group["timing_path_group"]["id"] = id++;
    std::string group_name = clock_group->get_clock_name();
    path_group["timing_path_group"]["name"] = group_name;
    double wns = ista->getWNS(group_name.c_str(), ista::AnalysisMode::kMax);
    double tns = ista->getTNS(group_name.c_str(), ista::AnalysisMode::kMax);
    path_group["timing_path_group"]["WNS"] = wns;
    path_group["timing_path_group"]["TNS"] = tns;
    path_group["timing_path_group"]["NVP"] = 0;  // TBD for negative violated points.
    double freq = 1000.0 / (clock_group->getPeriodNs() - wns);
    path_group["timing_path_group"]["FREQ"] = freq;
    double hold_wns = ista->getWNS(group_name.c_str(), ista::AnalysisMode::kMin);
    double hold_tns = ista->getTNS(group_name.c_str(), ista::AnalysisMode::kMin);
    path_group["timing_path_group"]["hold_WNS"] = wns;
    path_group["timing_path_group"]["hold_TNS"] = tns;
    path_group["timing_path_group"]["hold_NVP"] = 0;  // TBD for hold negative violated points.

    FOREACH_MODE(mode)
    {
      json::value_type analysis_mode;
      analysis_mode["analysis_mode"] = (mode == AnalysisMode::kMax) ? "max_delay/setup" : "min_delay/hold";
      analysis_mode["levels_of_logic"] = 0;       // TBD
      analysis_mode["critical_path_length"] = 0;  // TBD
      analysis_mode["critical_path_slack"] = (mode == AnalysisMode::kMax) ? wns : hold_wns;
      analysis_mode["total_negative_slack"] = (mode == AnalysisMode::kMax) ? tns : hold_tns;
      path_group["timing_path_group"]["analysis_mode_infos"].push_back(analysis_mode);
    }
    summary_sta.push_back(path_group);
  }

  return summary_sta;
}

json FeatureParser::buildSummaryDRC()
{
  json summary_drc;

  //   auto drc_map = idrc::DrcAPIInst.getCheckResult();
  //   // summary_drc["short_nums"] = drc_map
  //   for (auto& [key, value] : drc_map) {
  //     summary_drc[key] = value;
  //   }

  return summary_drc;
}

json FeatureParser::buildSummaryRT()
{
  json summary_rt;

  auto& rt_sum = dmInst->get_feature_summary().getRTSummary();
  json rt_pa;
  for (auto routing_access_point_num : rt_sum.pa_summary.routing_access_point_num_map) {
    rt_pa["routing_access_point_num_map"][std::to_string(routing_access_point_num.first)] = routing_access_point_num.second;
  }
  for (auto type_access_point_num : rt_sum.pa_summary.type_access_point_num_map) {
    rt_pa["routing_access_point_num_map"][type_access_point_num.first] = type_access_point_num.second;
  }
  rt_pa["routing_access_point_num_map"]["total_access_point_num"] = rt_sum.pa_summary.total_access_point_num;
  summary_rt["PA"] = rt_pa;

  auto& sa_sum = rt_sum.sa_summary;
  json rt_sa;
  for (auto routing_supply_num : rt_sum.sa_summary.routing_supply_map) {
    rt_sa["routing_supply_num_map"][std::to_string(routing_supply_num.first)] = routing_supply_num.second;
  }
  rt_sa["routing_supply_num_map"]["total_supply_num"] = rt_sum.sa_summary.total_supply;

  json rt_ir;
  for (auto demand : rt_sum.ir_summary.routing_demand_map) {
    rt_ir["routing_demand_map"][std::to_string(demand.first)] = demand.second;
  }
  rt_ir["routing_demand_map"]["total_demand"] = rt_sum.ir_summary.total_demand;
  for (auto routing_overflow : rt_sum.ir_summary.routing_overflow_map) {
    rt_ir["routing_overflow_map"][std::to_string(routing_overflow.first)] = routing_overflow.second;
  }
  rt_ir["routing_overflow_map"]["total_overflow"] = rt_sum.ir_summary.total_overflow;
  for (auto routing_wire_length : rt_sum.ir_summary.routing_wire_length_map) {
    rt_ir["routing_wire_length_map"][std::to_string(routing_wire_length.first)] = routing_wire_length.second;
  }
  rt_ir["routing_wire_length_map"]["total_wire_length"] = rt_sum.ir_summary.total_wire_length;
  for (auto cut_via_num : rt_sum.ir_summary.cut_via_num_map) {
    rt_ir["routing_cut_via_num_map"][std::to_string(cut_via_num.first)] = cut_via_num.second;
  }
  rt_ir["routing_cut_via_num_map"]["total_cut_via_num"] = rt_sum.ir_summary.total_via_num;
  for (auto timing : rt_sum.ir_summary.timing) {
    rt_ir["routing_timing_map"][timing.first] = timing.second;
  }
  summary_rt["IR"] = rt_ir;

  // // GR
  for (auto [id, gr_sum] : rt_sum.iter_gr_summary_map) {
    json rt_gr;
    // 和ir一样
    for (auto demand : gr_sum.routing_demand_map) {
      rt_gr["routing_demand_map"][std::to_string(demand.first)] = demand.second;
    }
    rt_gr["routing_demand_map"]["total_demand"] = gr_sum.total_demand;
    for (auto routing_overflow : gr_sum.routing_overflow_map) {
      rt_gr["routing_overflow_map"][std::to_string(routing_overflow.first)] = routing_overflow.second;
    }
    rt_gr["routing_overflow_map"]["total_overflow"] = gr_sum.total_overflow;
    for (auto routing_wire_length : gr_sum.routing_wire_length_map) {
      rt_gr["routing_wire_length_map"][std::to_string(routing_wire_length.first)] = routing_wire_length.second;
    }
    rt_gr["routing_wire_length_map"]["total_wire_length"] = gr_sum.total_wire_length;
    for (auto cut_via_num : gr_sum.cut_via_num_map) {
      rt_gr["routing_cut_via_num_map"][std::to_string(cut_via_num.first)] = cut_via_num.second;
    }
    rt_gr["routing_cut_via_num_map"]["total_cut_via_num"] = gr_sum.total_via_num;
    for (auto timing : gr_sum.timing) {
      rt_gr["routing_timing_map"][timing.first] = timing.second;
    }
    summary_rt["GR"][std::to_string(id)] = rt_gr;
  }
  // TA
  json rt_ta;
  // wirelength, violation
  for (auto routing_wire_length : rt_sum.ta_summary.routing_wire_length_map) {
    rt_ta["routing_wire_length_map"][std::to_string(routing_wire_length.first)] = routing_wire_length.second;
  }
  rt_ta["routing_wire_length_map"]["total_wire_length"] = rt_sum.ta_summary.total_wire_length;
  for (auto routing_violation : rt_sum.ta_summary.routing_violation_num_map) {
    rt_ta["routing_violation_map"][std::to_string(routing_violation.first)] = routing_violation.second;
  }
  rt_ta["routing_violation_map"]["total_violation"] = rt_sum.ta_summary.total_violation_num;
  summary_rt["TA"] = rt_ta;

  // DR
  for (auto [id, dr_sum] : rt_sum.iter_dr_summary_map) {
    json rt_dr;
    for (auto routing_wire_length : dr_sum.routing_wire_length_map) {
      rt_dr["routing_wire_length_map"][std::to_string(routing_wire_length.first)] = routing_wire_length.second;
    }
    rt_dr["routing_wire_length_map"]["total_wire_length"] = dr_sum.total_wire_length;
    for (auto cut_via_num : dr_sum.cut_via_num_map) {
      rt_dr["routing_cut_via_num_map"][std::to_string(cut_via_num.first)] = cut_via_num.second;
    }
    rt_dr["routing_cut_via_num_map"]["total_cut_via_num"] = dr_sum.total_via_num;
    // violation
    for (auto routing_violation : dr_sum.routing_violation_num_map) {
      rt_dr["routing_violation_map"][std::to_string(routing_violation.first)] = routing_violation.second;
    }
    rt_dr["routing_violation_map"]["total_violation"] = dr_sum.total_violation_num;
    for (auto routing_patch_num : dr_sum.routing_patch_num_map) {
      rt_dr["routing_patch_num_map"][std::to_string(routing_patch_num.first)] = routing_patch_num.second;
    }
    rt_dr["routing_patch_num_map"]["total_patch_num"] = dr_sum.total_patch_num;
    for (auto timing : dr_sum.timing) {
      rt_dr["routing_timing_map"][timing.first] = timing.second;
    }
    summary_rt["DR"][std::to_string(id)] = rt_dr;
  }
  return summary_rt;
}

}  // namespace idb
