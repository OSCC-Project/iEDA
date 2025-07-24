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
 * @file PowerRouter.cpp
 * @author Jianrong Su
 * @brief
 * @version 0.1
 * @date 2025-06-23
 */

#include "PowerRouter.hh"

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace ipnp {

void PowerRouter::addPowerStripesToCore(idb::IdbSpecialNet* power_net, GridManager pnp_network)
{
  std::string net_name;
  if (power_net->is_vdd()) {
    net_name = "VDD";
  }
  else {
    net_name = "VSS";
  }

  idb::IdbSpecialWireList* wire_list = power_net->get_wire_list();

  auto grid_data = pnp_network.get_grid_data();
  auto template_data = pnp_network.get_template_data();
  auto power_layers = pnp_network.get_power_layers();

  for (int layer_idx = 0;layer_idx < pnp_network.get_layer_count();layer_idx++) {

    // Get the base coordinates of the core
    double x_base = pnp_network.get_core_llx();
    double y_base = pnp_network.get_core_lly();

    // Create a new routing layer
    idb::IdbLayer* layer = new idb::IdbLayer();
    layer->set_name("M" + std::to_string(power_layers[layer_idx]));
    layer->set_type(idb::IdbLayerType::kLayerRouting);

    // Begin: convert pnp_network to wire
    idb::IdbSpecialWire* wire = new idb::IdbSpecialWire();
    wire->set_wire_state(idb::IdbWiringStatement::kRouted);

    for (int i = 0; i < pnp_network.get_ho_region_num(); i++) {
      for (int j = 0; j < pnp_network.get_ver_region_num(); j++) {

        SingleTemplate& single_template = template_data[layer_idx][i][j];
        double width = single_template.get_width();
        double space = single_template.get_space();
        double offset = single_template.get_offset();
        double pg_offset = single_template.get_pg_offset();

        /**
         * @brief current_segment_outermost: the location of the outermost edge of the current segment.
         */
        double current_segment_outermost;
        if (net_name == "VSS") {
          current_segment_outermost = offset + width;
        }
        else {  // net_name == VDD
          current_segment_outermost = offset + width + pg_offset + width;
        }

        int segment_count = 0;

        // Convert single_template to segment
        if (single_template.get_direction() == StripeDirection::kHorizontal) {
          // Calculate the maximum number of segments that can be placed in the region
          int max_segment_count = (grid_data[layer_idx][i][j].get_height() - offset) / space;
          if (grid_data[layer_idx][i][j].get_height() - offset - max_segment_count * space > 2 * width + pg_offset) {
            max_segment_count++;
          }
          while (segment_count < max_segment_count) {
            // Create a new segment
            idb::IdbSpecialWireSegment* segment = new idb::IdbSpecialWireSegment();
            segment->set_layer(layer);
            segment->set_route_width((int)width);
            segment->set_shape_type(idb::IdbWireShapeType::kStripe);

            // Set the segment coordinates
            double stripe_x1 = x_base;
            double stripe_y1 = y_base + current_segment_outermost - 0.5 * width;
            double stripe_x2 = x_base + grid_data[layer_idx][i][j].get_width();
            double stripe_y2 = stripe_y1;
            segment->add_point((int)stripe_x1, (int)stripe_y1);
            segment->add_point((int)stripe_x2, (int)stripe_y2);

            // Add the segment to the wire
            wire->add_segment(segment);

            // Calculate the outermost position of the next segment
            current_segment_outermost += space;
            segment_count++;
          }
        }
        else {  // direction is vertical
          int max_segment_count = (grid_data[layer_idx][i][j].get_width() - offset) / space;
          if (grid_data[layer_idx][i][j].get_width() - offset - max_segment_count * space > 2 * width + pg_offset) {
            max_segment_count++;
          }
          while (segment_count < max_segment_count) {
            // Create a new segment
            idb::IdbSpecialWireSegment* segment = new idb::IdbSpecialWireSegment();
            segment->set_layer(layer);
            segment->set_route_width((int)width);
            segment->set_shape_type(idb::IdbWireShapeType::kStripe);

            // Set the segment coordinates
            double stripe_x1 = x_base + current_segment_outermost - 0.5 * width;
            double stripe_y1 = y_base;
            double stripe_x2 = stripe_x1;
            double stripe_y2 = y_base + grid_data[layer_idx][i][j].get_height();
            segment->add_point((int)stripe_x1, (int)stripe_y1);
            segment->add_point((int)stripe_x2, (int)stripe_y2);

            // Add the segment to the wire
            wire->add_segment(segment);

            // Calculate the outermost position of the next segment
            current_segment_outermost += space;
            segment_count++;
          }
        }

        x_base += grid_data[layer_idx][i][j].get_width();
      }
      x_base = pnp_network.get_core_llx();
      y_base += grid_data[layer_idx][i][0].get_height();
    }
    wire_list->add_wire(wire, idb::IdbWiringStatement::kRouted);
  }
}

void PowerRouter::addPowerStripesToDie(idb::IdbSpecialNet* power_net, GridManager pnp_network)
{
  std::string net_name;
  if (power_net->is_vdd()) {
    net_name = "VDD";
  }
  else {
    net_name = "VSS";
  }

  idb::IdbSpecialWireList* wire_list = power_net->get_wire_list();

  auto grid_data = pnp_network.get_grid_data();
  auto template_data = pnp_network.get_template_data();
  auto power_layers = pnp_network.get_power_layers();

  for (int layer_idx = 0;layer_idx < pnp_network.get_layer_count();layer_idx++) {

    // Get the base coordinates of the die
    double x_base = 0.0;
    double y_base = 0.0;

    // Create a new routing layer
    idb::IdbLayer* layer = new idb::IdbLayer();
    layer->set_name("M" + std::to_string(power_layers[layer_idx]));
    layer->set_type(idb::IdbLayerType::kLayerRouting);

    // Begin: convert pnp_network to wire
    idb::IdbSpecialWire* wire = new idb::IdbSpecialWire();
    wire->set_wire_state(idb::IdbWiringStatement::kRouted);

    for (int i = 0; i < pnp_network.get_ho_region_num(); i++) {
      for (int j = 0; j < pnp_network.get_ver_region_num(); j++) {

        SingleTemplate& single_template = template_data[layer_idx][i][j];
        double width = single_template.get_width();
        double space = single_template.get_space();
        double offset = single_template.get_offset();
        double pg_offset = single_template.get_pg_offset();

        /**
         * @brief current_segment_outermost: the location of the outermost edge of the current segment.
         */
        double current_segment_outermost;
        if (net_name == "VSS") {
          current_segment_outermost = offset + width;
        }
        else {  // net_name == VDD
          current_segment_outermost = offset + width + pg_offset + width;
        }

        int segment_count = 0;

        // Convert single_template to segment
        if (single_template.get_direction() == StripeDirection::kHorizontal) {
          // Calculate the maximum number of segments that can be placed in the region
          int max_segment_count = (grid_data[layer_idx][i][j].get_height() - offset) / space;
          if (grid_data[layer_idx][i][j].get_height() - offset - max_segment_count * space > 2 * width + pg_offset) {
            max_segment_count++;
          }
          while (segment_count < max_segment_count) {
            // Create a new segment
            idb::IdbSpecialWireSegment* segment = new idb::IdbSpecialWireSegment();
            segment->set_layer(layer);
            segment->set_route_width((int)width);
            segment->set_shape_type(idb::IdbWireShapeType::kStripe);

            // Set the segment coordinates
            double stripe_x1 = x_base;
            double stripe_y1 = y_base + current_segment_outermost - 0.5 * width;
            double stripe_x2 = x_base + grid_data[layer_idx][i][j].get_width();
            double stripe_y2 = stripe_y1;
            segment->add_point((int)stripe_x1, (int)stripe_y1);
            segment->add_point((int)stripe_x2, (int)stripe_y2);

            // Add the segment to the wire
            wire->add_segment(segment);

            // Calculate the outermost position of the next segment
            current_segment_outermost += space;
            segment_count++;
          }
        }
        else {  // direction is vertical
          int max_segment_count = (grid_data[layer_idx][i][j].get_width() - offset) / space;
          if (grid_data[layer_idx][i][j].get_width() - offset - max_segment_count * space > 2 * width + pg_offset) {
            max_segment_count++;
          }
          while (segment_count < max_segment_count) {
            // Create a new segment
            idb::IdbSpecialWireSegment* segment = new idb::IdbSpecialWireSegment();
            segment->set_layer(layer);
            segment->set_route_width((int)width);
            segment->set_shape_type(idb::IdbWireShapeType::kStripe);

            // Set the segment coordinates
            double stripe_x1 = x_base + current_segment_outermost - 0.5 * width;
            double stripe_y1 = y_base;
            double stripe_x2 = stripe_x1;
            double stripe_y2 = y_base + grid_data[layer_idx][i][j].get_height();
            segment->add_point((int)stripe_x1, (int)stripe_y1);
            segment->add_point((int)stripe_x2, (int)stripe_y2);

            // Add the segment to the wire
            wire->add_segment(segment);

            // Calculate the outermost position of the next segment
            current_segment_outermost += space;
            segment_count++;
          }
        }

        x_base += grid_data[layer_idx][i][j].get_width();
      }
      x_base = 0.0;
      y_base += grid_data[layer_idx][i][0].get_height();
    }
    wire_list->add_wire(wire, idb::IdbWiringStatement::kRouted);
  }

  int wire_need_to_add = power_layers[0] - pnp_network.get_layer_count() - 2;
  while (wire_need_to_add > 0) {
    idb::IdbSpecialWire* wire = new idb::IdbSpecialWire();
    wire->set_wire_state(idb::IdbWiringStatement::kRouted);
    wire_list->add_wire(wire, idb::IdbWiringStatement::kRouted);
    wire_need_to_add--;
  }
}

void PowerRouter::addPowerFollowPin(idb::IdbDesign* idb_design, idb::IdbSpecialNet* power_net)
{
  auto rows = idb_design->get_layout()->get_rows();
  auto wire_list = power_net->get_wire_list();
  auto row_list = rows->get_row_list();

  for (int layer_idx = 2; layer_idx > 0; layer_idx--) {
    // Create a new routing layer
    idb::IdbLayer* layer = new idb::IdbLayer();
    layer->set_name("M" + std::to_string(layer_idx));
    layer->set_type(idb::IdbLayerType::kLayerRouting);

    // Create a new wire
    idb::IdbSpecialWire* wire = new idb::IdbSpecialWire();
    wire->set_wire_state(idb::IdbWiringStatement::kRouted);

    int segment_idx;
    int row_num = rows->get_row_num();

    if (power_net->is_vdd()) {
      segment_idx = 0;
    }
    else {
      segment_idx = 1;
    }

    while(segment_idx < row_num) {
      // Create a new follow pin segment
      idb::IdbSpecialWireSegment* segment = new idb::IdbSpecialWireSegment();
      segment->set_layer(layer);
      segment->set_route_width(300.0);
      segment->set_shape_type(idb::IdbWireShapeType::kFollowPin);

      // Set the segment coordinates
      double stripe_x1 = row_list[segment_idx]->get_original_coordinate()->get_x();
      double stripe_y1 = row_list[segment_idx]->get_original_coordinate()->get_y();
      double stripe_x2 = stripe_x1 + idb_design->get_layout()->get_core()->get_bounding_box()->get_width();
      double stripe_y2 = stripe_y1;
      segment->add_point((int)stripe_x1, (int)stripe_y1);
      segment->add_point((int)stripe_x2, (int)stripe_y2);

      // Add the segment to the wire
      wire->add_segment(segment);

      segment_idx += 2;
    }

    if (row_num % 2 == 0 && power_net->is_vdd()) {
      // Create a new follow pin segment
      idb::IdbSpecialWireSegment* segment = new idb::IdbSpecialWireSegment();
      segment->set_layer(layer);
      segment->set_route_width(300.0);
      segment->set_shape_type(idb::IdbWireShapeType::kFollowPin);

      // Set the segment coordinates
      double stripe_x1 = row_list[row_num - 1]->get_bounding_box()->get_low_x();
      double stripe_y1 = row_list[row_num - 1]->get_bounding_box()->get_high_y();
      double stripe_x2 = stripe_x1 + idb_design->get_layout()->get_core()->get_bounding_box()->get_width();
      double stripe_y2 = stripe_y1;
      segment->add_point((int)stripe_x1, (int)stripe_y1);
      segment->add_point((int)stripe_x2, (int)stripe_y2);

      // Add the segment to the wire
      wire->add_segment(segment);
    }

    // Add the wire to the wire list
    wire_list->add_wire(wire, idb::IdbWiringStatement::kRouted);
  }
}

void PowerRouter::addPowerPort(idb::IdbDesign* idb_design, GridManager pnp_network, std::string pin_name, std::string layer_name)
{
  auto idb_layout = idb_design->get_layout();
  idb::IdbLayer* layer = idb_layout->get_layers()->find_layer(layer_name);

  idb::IdbPin* io_pin = idb_design->get_io_pin_list()->find_pin(pin_name);
  if (io_pin == nullptr) {
    LOG_INFO << "Can not find " << pin_name << " in io_pin_list.";
    LOG_INFO << "Create a new io pin.";

    // create a new io pin
    io_pin = new idb::IdbPin();
    io_pin->set_pin_name(pin_name);
    io_pin->set_net_name(pin_name);
    io_pin->set_term();
    idb_design->get_io_pin_list()->get_pin_list().emplace_back(io_pin);
  }

  // set term attribute
  idb::IdbTerm* term = io_pin->get_term();
  term->set_name(pin_name);
  term->set_direction(idb::IdbConnectDirection::kInOut);
  if (pin_name == "VSS") {
    term->set_type(idb::IdbConnectType::kGround);
  }
  else {
    term->set_type(idb::IdbConnectType::kPower);
  }
  term->set_special(true);
  term->set_placement_status_fix();
  term->set_has_port(true);

  auto template_data = pnp_network.get_template_data();
  auto grid_data = pnp_network.get_grid_data();
  double x_base = 0.0;
  double y_base = 0.0;

  if (template_data[2][0][0].get_direction() == StripeDirection::kHorizontal) {
    for (int i = 0; i < pnp_network.get_ho_region_num(); i++) {
      SingleTemplate& single_template = template_data[2][i][0];
      double width = single_template.get_width();
      double space = single_template.get_space();
      double offset = single_template.get_offset();
      double pg_offset = single_template.get_pg_offset();

      double current_outermost;
      if (pin_name == "VSS") {
        current_outermost = offset + width;
      }
      else {
        current_outermost = offset + width + pg_offset + width;
      }

      int port_count = 0;

      int max_port_count = (grid_data[2][i][0].get_height() - offset) / space;
      if (grid_data[2][i][0].get_height() - offset - max_port_count * space > 2 * width + pg_offset) {
        max_port_count++;
      }
      while (port_count < max_port_count) {
        // add port_1
        double port_x1 = x_base;
        double port_y1 = y_base + current_outermost - 0.5 * width;

        idb::IdbPort* port_1 = term->add_port();
        port_1->set_coordinate(int32_t(port_x1), int32_t(port_y1));
        port_1->set_placement_status_place();
        port_1->set_orient(idb::IdbOrient::kE_R270);

        idb::IdbLayerShape* layer_shape = port_1->add_layer_shape();
        layer_shape->set_layer(layer);
        layer_shape->add_rect(int32_t(-width / 2), 0, int32_t(width / 2), int32_t(width));

        // add port_2
        double port_x2 = port_x1 + idb_design->get_layout()->get_die()->get_width();
        double port_y2 = port_y1;

        idb::IdbPort* port_2 = term->add_port();
        port_2->set_coordinate(int32_t(port_x2), int32_t(port_y2));
        port_2->set_placement_status_place();
        port_2->set_orient(idb::IdbOrient::kW_R90);

        idb::IdbLayerShape* layer_shape_2 = port_2->add_layer_shape();
        layer_shape_2->set_layer(layer);
        layer_shape_2->add_rect(int32_t(-width / 2), 0, int32_t(width / 2), int32_t(width));

        // update current_outermost
        current_outermost += space;
        port_count++;
      }
      y_base += grid_data[2][i][0].get_height();
    }
  } // vertical
  else {
    for (int j = 0;j < pnp_network.get_ver_region_num();j++) {
      SingleTemplate& single_template = template_data[2][0][j];
      double width = single_template.get_width();
      double space = single_template.get_space();
      double offset = single_template.get_offset();
      double pg_offset = single_template.get_pg_offset();

      double current_outermost;
      if (pin_name == "VSS") {
        current_outermost = offset + width;
      }
      else {
        current_outermost = offset + width + pg_offset + width;
      }

      int port_count = 0;

      int max_port_count = (grid_data[2][0][j].get_width() - offset) / space;
      if (grid_data[2][0][j].get_width() - offset - max_port_count * space > 2 * width + pg_offset) {
        max_port_count++;
      }
      while (port_count < max_port_count) {
        // add port_1
        double port_x1 = x_base + current_outermost - 0.5 * width;
        double port_y1 = y_base;

        idb::IdbPort* port_1 = term->add_port();
        port_1->set_coordinate(int32_t(port_x1), int32_t(port_y1));
        port_1->set_placement_status_place();
        port_1->set_orient(idb::IdbOrient::kN_R0);

        idb::IdbLayerShape* layer_shape = port_1->add_layer_shape();
        layer_shape->set_layer(layer);
        layer_shape->add_rect(int32_t(-width / 2), 0, int32_t(width / 2), int32_t(width));

        // add port_2
        double port_x2 = port_x1;
        double port_y2 = port_y1 + idb_design->get_layout()->get_die()->get_height();

        idb::IdbPort* port_2 = term->add_port();
        port_2->set_coordinate(int32_t(port_x2), int32_t(port_y2));
        port_2->set_placement_status_place();
        port_2->set_orient(idb::IdbOrient::kS_R180);

        idb::IdbLayerShape* layer_shape_2 = port_2->add_layer_shape();
        layer_shape_2->set_layer(layer);
        layer_shape_2->add_rect(int32_t(-width / 2), 0, int32_t(width / 2), int32_t(width));

        // update current_outermost
        current_outermost += space;
        port_count++;
      }
      x_base += grid_data[2][0][j].get_width();
    }
  }
}

void PowerRouter::addVSSNet(idb::IdbDesign* idb_design, GridManager pnp_network)
{
  idb::IdbSpecialNet* vss_net = idb_design->get_special_net_list()->find_net("VSS");

  if (vss_net == nullptr) {
    idb::IdbSpecialNet* power_net = new idb::IdbSpecialNet();
    power_net->set_net_name("VSS");
    power_net->set_connect_type(idb::IdbConnectType::kGround);
    idb::IdbPin* io_pin = new idb::IdbPin();
    io_pin->set_pin_name("VSS");
    power_net->add_io_pin(io_pin);

    idb_design->get_special_net_list()->add_net(power_net);
    vss_net = power_net;
  }

  addPowerStripesToDie(vss_net, pnp_network);
  LOG_INFO << "Add VSS Power Stripes success.";

  addPowerFollowPin(idb_design, vss_net);
  LOG_INFO << "Add VSS Power Follow Pin success.";

  addPowerPort(idb_design, pnp_network, "VSS", "M7");
  LOG_INFO << "Add VSS Power Port success.";
}

void PowerRouter::addVDDNet(idb::IdbDesign* idb_design, GridManager pnp_network)
{
  idb::IdbSpecialNet* vdd_net = idb_design->get_special_net_list()->find_net("VDD");

  if (vdd_net == nullptr) {
    idb::IdbSpecialNet* power_net = new idb::IdbSpecialNet();
    power_net->set_net_name("VDD");
    power_net->set_connect_type(idb::IdbConnectType::kPower);
    idb::IdbPin* io_pin = new idb::IdbPin();
    io_pin->set_pin_name("VDD");
    power_net->add_io_pin(io_pin);

    idb_design->get_special_net_list()->add_net(power_net);
    vdd_net = power_net;
  }

  addPowerStripesToDie(vdd_net, pnp_network);
  LOG_INFO << "Add VDD Power Stripes success.";

  addPowerFollowPin(idb_design, vdd_net);
  LOG_INFO << "Add VDD Power Follow Pin success.";

  addPowerPort(idb_design, pnp_network, "VDD", "M7");
  LOG_INFO << "Add VDD Power Port success.";
}

void PowerRouter::addPowerNets(idb::IdbDesign* idb_design, GridManager pnp_network)
{
  if (!idb_design) {
    LOG_INFO << "Invalid IDB design object." << std::endl;
    return;
  }

  addVSSNet(idb_design, pnp_network);
  LOG_INFO << "Add VSS Power Nets success.";

  addVDDNet(idb_design, pnp_network);
  LOG_INFO << "Add VDD Power Nets success.";
}

}  // namespace ipnp