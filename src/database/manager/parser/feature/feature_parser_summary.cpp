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
#include "PlacerDB.hh"
#include "PLAPI.hh"
#include "CTSAPI.hh"
#include "TimingEngine.hh"
#include "feature_parser.h"
#include "flow_config.h"
#include "idm.h"
#include "iomanip"
#include "json_parser.h"

namespace idb {

bool FeatureParser::buildReportSummary(std::string json_path)
{
  std::ofstream& file_stream = ieda::getOutputFileStream(json_path);
  json root;

  root["Design Information"] = buildSummaryInfo();

  root["Design Layout"] = buildSummaryLayout();

  root["Design Statis"] = buildSummaryStatis();

  root["Instances"] = buildSummaryInstances();

  root["Nets"] = buildSummaryNets();

  root["PDN"] = buildSummaryPdn();

  root["Layers"] = buildSummaryLayers();

  root["Pins"] = buildSummaryPins();

  // root["Place"] = buildSummaryPL(json_path);

  file_stream << std::setw(4) << root;

  ieda::closeFileStream(file_stream);

  std::cout << std::endl << "Save feature json success, path = " << json_path << std::endl;
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

json FeatureParser::buildSummaryPL(std::string json_path)
{

  std::string path = json_path;
  // get step
  size_t lastSlash = path.find_last_of('/');
  size_t lastfirstUnderline = path.find_first_of('_', lastSlash + 1);
  size_t lastsecondUnderline = path.find_first_of('_', lastfirstUnderline + 1);
  std::string step = path.substr(lastfirstUnderline + 1, lastsecondUnderline - lastfirstUnderline - 1);

  // 按照step获取index
  // int index_step = [&step]()->int{
  //   if(step == "place") return 0;
  //   else if(step == "dplace") return 1;
  //   return 2;
  // }();

  int index_step;
  if(step == "place")index_step = 0;
  else if(step == "dplace")index_step = 1;
  else index_step = 2;
  
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
  
  summary_pl["place_density"] = place_density[index_step];
  summary_pl["pin_density"] = pin_density[index_step];
  summary_pl["HPWL"] = HPWL[index_step];
  summary_pl["STWL"] = STWL[index_step];
  summary_pl["global_routing_WL"] = GRWL[index_step];
  summary_pl["congestion"] = congestion[index_step];
  summary_pl["tns"] = tns[index_step];
  summary_pl["wns"] = wns[index_step];
  summary_pl["suggest_freq"] = suggest_freq[index_step];

  // 2:全局布局、详细布局需要存储的数据参数
  #if 1
  if(index_step != 2){
    auto* pl_design = PlacerDBInst.get_design();
    summary_pl["instance"] = pl_design->get_instances_range();
    int fix_inst_cnt = 0;
    for(auto* inst : pl_design->get_instance_list()){
      if(inst->isFixed()){
        fix_inst_cnt++;
      }
    }

    summary_pl["fix_instances"] = fix_inst_cnt;
    summary_pl["nets"] = pl_design->get_nets_range();
    summary_pl["total_pins"] = pl_design->get_pins_range();
    summary_pl["core_area"] = std::to_string(PlacerDBInst.get_layout()->get_core_shape().get_width()) + " * " + std::to_string(PlacerDBInst.get_layout()->get_core_shape().get_height());

    summary_pl["bin_number"] = PlacerDBInst.get_placer_config()->get_nes_config().get_bin_cnt_x() * PlacerDBInst.get_placer_config()->get_nes_config().get_bin_cnt_y();
    summary_pl["bin_size"] = std::to_string(PlacerDBInst.bin_size_x) + " * " + std::to_string(PlacerDBInst.bin_size_y);
    summary_pl["overflow_number"] = PlacerDBInst.gp_overflow_number;
    summary_pl["overflow"] = PlacerDBInst.gp_overflow;
  }
  #endif

  // 3:合法化需要存储的数据参数
  if(index_step == 2){
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

  auto _timing_engine = ista::TimingEngine::getOrCreateTimingEngine();


  // 可能有多个clk_name，每一个时钟都需要报告tns、wns、freq
  std::map<std::string, vector<double>> clk_sta_eval;
  auto clk_list = _timing_engine->getClockList();
  std::ranges::for_each(clk_list, [&](ista::StaClock* clk){
    auto clk_name = clk->get_clock_name();
    auto setup_tns = _timing_engine->reportTNS(clk_name, AnalysisMode::kMax);
    auto setup_wns = _timing_engine->reportWNS(clk_name, AnalysisMode::kMax);
    auto hold_tns = _timing_engine->reportTNS(clk_name, AnalysisMode::kMin);
    auto hold_wns = _timing_engine->reportWNS(clk_name, AnalysisMode::kMin);
    auto suggest_freq = 1000.0 / (clk->getPeriodNs() - setup_wns);
    
  });

  auto design_area = dmInst->dieAreaUm();
  auto design_utilization = dmInst->dieUtilization();
  auto clock_nets = _design->get_net_list()->get_num_clock();
  
  // auto setup_tns = _timing_engine->reportTNS();

  return json();
}

}  // namespace idb
