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

#include "feature_parser.h"

#include "IdbCore.h"
#include "IdbDesign.h"
#include "IdbDie.h"
#include "IdbEnum.h"
#include "IdbInstance.h"
#include "IdbLayout.h"
#include "IdbNet.h"
#include "IdbRow.h"
#include "IdbTrackGrid.h"
#include "iomanip"
#include "json_parser.h"
#include "flow_config.h"
#include "idm.h"

namespace idb {

bool FeatureParser::buildLayout(std::string json_path)
{
  nlohmann::json root_json;
  std::ofstream& file_stream = ieda::getOutputFileStream(json_path);
  {
    root_json["top_name"] = _design->get_design_name();
    root_json["dbu"] = _layout->get_units()->get_micron_dbu();

    // Die
    root_json["die"]["llx"] = _layout->get_die()->get_llx();
    root_json["die"]["lly"] = _layout->get_die()->get_lly();
    root_json["die"]["urx"] = _layout->get_die()->get_urx();
    root_json["die"]["ury"] = _layout->get_die()->get_ury();

    // Core
    root_json["core"]["llx"] = _layout->get_core()->get_bounding_box()->get_low_x();
    root_json["core"]["lly"] = _layout->get_core()->get_bounding_box()->get_low_y();
    root_json["core"]["urx"] = _layout->get_core()->get_bounding_box()->get_high_x();
    root_json["core"]["ury"] = _layout->get_core()->get_bounding_box()->get_high_y();

    // Rows
    root_json["rows"]["num_rows"] = _layout->get_rows()->get_row_num();
    root_json["rows"]["row_width"] = _layout->get_core()->get_bounding_box()->get_width();  // iEDA using the core width as the row width
    root_json["rows"]["row_height"] = _layout->get_rows()->get_row_height();
  }
  // tracks
  {
    json array = json::array();
    for (auto* track : _layout->get_track_grid_list()->get_track_grid_list()) {
      nlohmann::json json;
      json["layer"] = track->get_first_layer()->get_name();
      json["prefer_dir"] = track->get_track()->is_track_horizontal() ? "H" : "V";
      json["num"] = track->get_track_num();
      json["start"] = track->get_track()->get_start();
      json["step"] = track->get_track()->get_pitch();

      array.push_back(json);
    }
    root_json["tracks"] = array;
  }

  // layers
  {
    json array = json::array();
    IdbLayerProperty layer_property;
    for (auto* layer : _layout->get_layers()->get_routing_layers()) {
      IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(layer);

      nlohmann::json json;
      json["name"] = routing_layer->get_name();
      json["type"] = layer_property.get_name(routing_layer->get_type());
      json["id"] = routing_layer->get_id();
      json["order"] = routing_layer->get_order();
      json["min_width"] = routing_layer->get_min_width();
      json["max_width"] = routing_layer->get_max_width();
      json["width"] = routing_layer->get_width();
      json["area"] = routing_layer->get_area();

      array.push_back(json);
    }

    root_json["routing_layers"] = array;
  }

  file_stream << std::setw(4) << root_json;

  ieda::closeFileStream(file_stream);

  std::cout << std::endl << "Save feature json success, path = " << json_path << std::endl;

  return true;
}

bool FeatureParser::buildInstances(std::string json_path)
{
  nlohmann::json root_json;
  std::ofstream& file_stream = ieda::getOutputFileStream(json_path);
  // instance list
  {
    auto array_instance = json::array();
    IdbCellProperty cell_property;
    IdbSiteProperty orient_property;
    IdbInstancePropertyMap instance_property;
    int index = 0;
    for (auto* instacne : _design->get_instance_list()->get_instance_list()) {
      nlohmann::json json_instance;
      json_instance["name"] = instacne->get_name();
      json_instance["master"] = instacne->get_cell_master()->get_name();
      json_instance["type"] = cell_property.get_name(instacne->get_cell_master()->get_type());
      json_instance["llx"] = instacne->get_coordinate()->get_x();
      json_instance["lly"] = instacne->get_coordinate()->get_y();
      json_instance["urx"] = instacne->get_bounding_box()->get_high_x();
      json_instance["ury"] = instacne->get_bounding_box()->get_high_y();
      json_instance["orient"] = orient_property.get_orient_name(instacne->get_orient());
      json_instance["status"] = instance_property.get_status_str(instacne->get_status());

      auto array_pins = json::array();
      for (auto* pin : instacne->get_pin_list()->get_pin_list()) {
        if (pin->get_term()->is_pdn() || pin->get_net() == nullptr)
          continue;

        nlohmann::json json_pin;

        json_pin["name"] = pin->get_term()->get_name();
        json_pin["c_x"] = pin->get_average_coordinate()->get_x();
        json_pin["c_y"] = pin->get_average_coordinate()->get_y();
        json_pin["net"] = pin->get_net()->get_net_name();

        array_pins.push_back(json_pin);
      }
      json_instance["pin"] = array_pins;

      array_instance.push_back(json_instance);

      index++;
      if (index % 1000 == 0) {
        std::cout << "-" << std::flush;
        if (index % 100000 == 0 || index == _design->get_instance_list()->get_num()) {
          std::cout << std::endl;
        }
      }
    }

    root_json["instances"] = array_instance;
  }

  file_stream << std::setw(4) << root_json;

  ieda::closeFileStream(file_stream);

  std::cout << std::endl << "Save feature json success, path = " << json_path << std::endl;

  return true;
}

bool FeatureParser::buildNets(std::string json_path)
{
  nlohmann::json root_json;
  std::ofstream& file_stream = ieda::getOutputFileStream(json_path);
  // net list
  {
    auto array_net = json::array();

    IdbConnectProperty connect_property;
    int index = 0;
    for (auto* net : _design->get_net_list()->get_net_list()) {
      nlohmann::json json_net;
      json_net["name"] = net->get_net_name();
      json_net["type"] = connect_property.get_type_name(net->get_connect_type());

      auto array_pins = json::array();
      /// io pin
      auto* io_pin = net->get_io_pin();
      if (io_pin != nullptr) {
        nlohmann::json json_io_pin;
        json_io_pin["name"] = io_pin->get_term()->get_name();
        json_io_pin["c_x"] = io_pin->get_average_coordinate()->get_x();
        json_io_pin["c_y"] = io_pin->get_average_coordinate()->get_y();
        json_io_pin["instance"] = "";

        array_pins.push_back(json_io_pin);
      }

      // instance pins
      for (auto* pin : net->get_instance_pin_list()->get_pin_list()) {
        nlohmann::json json_pin;
        json_pin["name"] = pin->get_term()->get_name();
        json_pin["c_x"] = pin->get_average_coordinate()->get_x();
        json_pin["c_y"] = pin->get_average_coordinate()->get_y();
        json_pin["instance"] = pin->get_instance() == nullptr ? "" : pin->get_instance()->get_name();

        array_pins.push_back(json_pin);
      }
      json_net["pin"] = array_pins;

      array_net.push_back(json_net);

      index++;
      if (index % 1000 == 0) {
        std::cout << "-" << std::flush;
        if (index % 100000 == 0 || index == _design->get_instance_list()->get_num()) {
          std::cout << std::endl;
        }
      }
    }

    root_json["nets"] = array_net;
  }

  file_stream << std::setw(4) << root_json;

  ieda::closeFileStream(file_stream);

  std::cout << std::endl << "Save feature json success, path = " << json_path << std::endl;

  return true;
}

bool FeatureParser::buildReportSummary(std::string json_path)
{
  
  std::ofstream& file_stream = ieda::getOutputFileStream(json_path);
  json root;

  // step1: tittle
  json tittle;
  tittle["Version"] = iplf::flowConfigInst->get_env_info_software_version();
  tittle["Stage"] = iplf::flowConfigInst->get_status_stage();
  tittle["Runtime"] = iplf::flowConfigInst->get_status_runtime_string();
  tittle["Memmory"] = iplf::flowConfigInst->get_status_memmory_string();
  tittle["Design Name"] = _design->get_design_name();
  tittle["DEF&LEF Version"] = _design->get_version();
  tittle["DBU"] = _design->get_units()->get_micron_dbu();
  root["title"] = tittle;

  // step2: summary
  json summary;
  int dbu = _design->get_units()->get_micron_dbu() < 0 ? _layout->get_units()->get_micron_dbu()
                                                          : _design->get_units()->get_micron_dbu();
  auto* idb_die = _layout->get_die();
  auto die_width = ((double) idb_die->get_width()) / dbu;
  auto die_height = ((double) idb_die->get_height()) / dbu;
  summary["DIE Area ( um^2 )"] = ieda::Str::printf("%f = %03f * %03f", die_width * die_height, die_width, die_height) ;
  summary["DIE Usage"] = dmInst->dieUtilization();
  auto idb_core_box = _layout->get_core()->get_bounding_box();
  auto core_width = ((double) idb_core_box->get_width()) / dbu;
  auto core_height = ((double) idb_core_box->get_height()) / dbu;
  summary["CORE Area ( um^2 )"] = ieda::Str::printf("%f = %03f * %03f", core_width * core_height, core_width, core_height);
  summary["CORE Usage"] = dmInst->coreUtilization();
  summary["Number - Site"] = _design->get_layout()->get_sites()->get_sites_num();
  summary["Number - Row"] = _design->get_layout()->get_rows()->get_row_num();
  summary["Number - Track"] = _design->get_layout()->get_track_grid_list()->get_track_grid_num();
  summary["Number - Layer"] = _design->get_layout()->get_layers()->get_layers_num();
  summary["Number - Routing Layer"] = _design->get_layout()->get_layers()->get_routing_layers_number();
  summary["Number - Cut Layer"] = _design->get_layout()->get_layers()->get_cut_layers_number();
  summary["Number - GCell Grid"] = _design->get_layout()->get_gcell_grid_list()->get_gcell_grid_num();
  summary["Number - Cell Master"] = _design->get_layout()->get_cell_master_list()->get_cell_master_num();
  summary["Number - Via Rule"] = _design->get_layout()->get_via_rule_list()->get_num_via_rule_generate()
                                 + _design->get_layout()->get_via_list()->get_num_via()
                                 + _design->get_via_list()->get_num_via();
  summary["Number - IO Pin"] = _design->get_io_pin_list()->get_pin_num();
  summary["Number - Instance"] = _design->get_instance_list()->get_num();
  summary["Number - Blockage"] = _design->get_blockage_list()->get_num();
  summary["Number - Filler"] = _design->get_fill_list()->get_num_fill();
  summary["Number - Net"] = _design->get_net_list()->get_num();
  summary["Number - Special Net"] = _design->get_special_net_list()->get_num();
  root["Summary"] = summary;

  // step3: summary instance
  json summary_instance;
  json all_instance;
  all_instance["Number"] = _design->get_instance_list()->get_num();
  all_instance["Number Ratio"] = (double) (1);
  all_instance["Area"] = dmInst->instanceArea(IdbInstanceType::kMax);
  all_instance["Area Ratio"] = (double) (1);
  summary_instance["All Instances"] = all_instance;
  json netlist;
  netlist["Number"] = _design->get_instance_list()->get_num(IdbInstanceType::kNetlist);
  netlist["Number Ratio"] = ((double) _design->get_instance_list()->get_num(IdbInstanceType::kNetlist)) / _design->get_instance_list()->get_num();
  netlist["Area"] = dmInst->netlistInstArea();
  netlist["Area Ratio"] = ((double) dmInst->netlistInstArea()) / dmInst->instanceArea(IdbInstanceType::kMax);
  summary_instance["Netlist"] = netlist;
  json physical;
  physical["Number"] = _design->get_instance_list()->get_num(IdbInstanceType::kDist);
  physical["Number Ratio"] = ((double) _design->get_instance_list()->get_num(IdbInstanceType::kDist)) / _design->get_instance_list()->get_num();
  physical["Area"] = dmInst->distInstArea();
  physical["Area Ratio"] = ((double) dmInst->distInstArea()) / dmInst->instanceArea(IdbInstanceType::kMax);
  summary_instance["Physical"] = physical;
  json timing;
  timing["Number"] = _design->get_instance_list()->get_num(IdbInstanceType::kTiming);
  timing["Number Ratio"] = ((double) _design->get_instance_list()->get_num(IdbInstanceType::kTiming)) / _design->get_instance_list()->get_num();
  timing["Area"] = dmInst->timingInstArea();
  timing["Area Ratio"] = ((double) dmInst->timingInstArea()) / dmInst->instanceArea(IdbInstanceType::kMax);
  summary_instance["Timing"] = timing;
  json core;
  core["Number"] = _design->get_instance_list()->get_num_core();
  core["Number Ratio"] = ((double) _design->get_instance_list()->get_num_core()) / _design->get_instance_list()->get_num();
  core["Area"] = _design->get_instance_list()->get_area_core();
  core["Area Ratio"] = ((double) _design->get_instance_list()->get_area_core()) / dmInst->instanceArea(IdbInstanceType::kMax);
  summary_instance["Core"] = core;
  json core_logic;
  core_logic["Number"] = _design->get_instance_list()->get_num_core_logic();
  core_logic["Number Ratio"] = ((double) _design->get_instance_list()->get_num_core_logic()) / _design->get_instance_list()->get_num();
  core_logic["Area"] = _design->get_instance_list()->get_area_core_logic();
  core_logic["Area Ratio"] = ((double) _design->get_instance_list()->get_area_core_logic()) / dmInst->instanceArea(IdbInstanceType::kMax);
  summary_instance["Core - logic"] = core_logic;
  json pad;
  pad["Number"] = _design->get_instance_list()->get_num_pad();
  pad["Number Ratio"] = ((double) _design->get_instance_list()->get_num_pad()) / _design->get_instance_list()->get_num();
  pad["Area"] = _design->get_instance_list()->get_area_pad();
  pad["Area Ratio"] = ((double) _design->get_instance_list()->get_area_pad()) / dmInst->instanceArea(IdbInstanceType::kMax);
  summary_instance["Pad"] = pad;
  json block;
  block["Number"] = _design->get_instance_list()->get_num_block();
  block["Number Ratio"] = ((double) _design->get_instance_list()->get_num_block()) / _design->get_instance_list()->get_num();
  block["Area"] = _design->get_instance_list()->get_area_block();
  block["Area Ratio"] = ((double) _design->get_instance_list()->get_area_block()) / dmInst->instanceArea(IdbInstanceType::kMax);
  summary_instance["Block"] = block;
  json endcap;
  endcap["Number"] = _design->get_instance_list()->get_num_endcap();
  endcap["Number Ratio"] = ((double) _design->get_instance_list()->get_num_endcap()) / _design->get_instance_list()->get_num();
  endcap["Area"] = _design->get_instance_list()->get_area_endcap();
  endcap["Area Ratio"] = ((double) _design->get_instance_list()->get_area_endcap()) / dmInst->instanceArea(IdbInstanceType::kMax);
  summary_instance["Endcap"] = endcap;
  json cover;
  cover["Number"] = _design->get_instance_list()->get_num_cover();
  cover["Number Ratio"] = ((double) _design->get_instance_list()->get_num_cover()) / _design->get_instance_list()->get_num();
  cover["Area"] = _design->get_instance_list()->get_area_cover();
  cover["Area Ratio"] = ((double) _design->get_instance_list()->get_area_cover()) / dmInst->instanceArea(IdbInstanceType::kMax);
  summary_instance["Cover"] = cover;
  json ring;
  ring["Number"] = _design->get_instance_list()->get_num_ring();
  ring["Number Ratio"] = ((double) _design->get_instance_list()->get_num_ring()) / _design->get_instance_list()->get_num();
  ring["Area"] = _design->get_instance_list()->get_area_ring();
  ring["Area Ratio"] = ((double) _design->get_instance_list()->get_area_ring()) / dmInst->instanceArea(IdbInstanceType::kMax);
  summary_instance["Ring"] = ring;
  root["Summary - Instance"] = summary_instance;

  // step4: summary net
  json summary_net;
  json all_net;
  all_net["Number"] = _design->get_net_list()->get_num();
  all_net["Number Ratio"] = 1;
  all_net["Length"] = dmInst->allNetLength();
  all_net["Length Ratio"] = dmInst->allNetLength() == 0 ? 0 : 1;
  summary_net["All Nets"] = all_net;
  json signal;
  signal["Number"] = _design->get_net_list()->get_num_signal();
  signal["Number Ratio"] = ((double) _design->get_net_list()->get_num_signal()) / _design->get_net_list()->get_num();
  signal["Length"] = dmInst->getSignalNetListLength();
  signal["Length Ratio"] = dmInst->allNetLength() == 0 ? 0 : ((double) dmInst->getSignalNetListLength()) / dmInst->allNetLength();
  summary_net["Signal"] = signal;
  json clock;
  clock["Number"] = _design->get_net_list()->get_num_clock();
  clock["Number Ratio"] = ((double) _design->get_net_list()->get_num_clock()) / _design->get_net_list()->get_num();
  clock["Length"] = dmInst->getClockNetListLength();
  clock["Length Ratio"] = dmInst->allNetLength() == 0 ? 0 : ((double) dmInst->getClockNetListLength()) / dmInst->allNetLength();
  summary_net["Clock"] = clock;
  json pdn;
  pdn["Number"] = _design->get_net_list()->get_num_pdn();
  pdn["Number Ratio"] = ((double) _design->get_net_list()->get_num_pdn()) / _design->get_net_list()->get_num();
  pdn["Length"] = dmInst->getPdnNetListLength();
  pdn["Length Ratio"] = dmInst->allNetLength() == 0 ? 0 : ((double) dmInst->getPdnNetListLength()) / dmInst->allNetLength();
  summary_net["Power & Ground"] = pdn;
  root["Summary - Net"] = summary_net;

  // step5: summary layer
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
  };
  
  const int max_num = 34;
  const int max_fanout = 32;

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
  json temp_layer;
  for (int i = 0; i < layer_num; i++) {
    auto net_value = layer_net_value_list[i];
    auto special_net_value = layer_specialnet_value_list[i];
    temp_layer[net_value.layer_name]["Net - Wire Length"] = net_value.wire_len;
    temp_layer[net_value.layer_name]["Net - Wire Number"] = net_value.wire_num;
    temp_layer[net_value.layer_name]["Net - Via Number"] = net_value.via_num;
    temp_layer[net_value.layer_name]["Net - Patch Number"] = net_value.patch_num;
    temp_layer[net_value.layer_name]["Special Net - Wire Length"] = special_net_value.wire_len;
    temp_layer[net_value.layer_name]["Special Net - Wire Number"] = special_net_value.wire_num;
    temp_layer[net_value.layer_name]["Special Net - Via Number"] = special_net_value.via_num;
  }
  root["Summary - Layer"] = temp_layer;

  // step6: summary pin
  json summary_pin;
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
  for(int i = 0; i <= max_fanout; i++){
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
  root["Summary - Pin Distribution"] = summary_pin;
  
  file_stream << std::setw(4) << root;

  ieda::closeFileStream(file_stream);

  std::cout << std::endl << "Save feature json success, path = " << json_path << std::endl;
  return true;
}

}  // namespace idb
