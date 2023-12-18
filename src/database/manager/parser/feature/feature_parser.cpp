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

}  // namespace idb
