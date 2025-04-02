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
 * @date 2025-03-28
 */

#include "PowerRouter.hh"

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace ipnp {

void PowerRouter::addPowerStripes(idb::IdbSpecialNet* power_net, GridManager pnp_network)
{
  
  // 判断 power_net 的类型
  std::string net_name;
  if (power_net->is_vdd()) {
    net_name = "VDD";
  }
  else {
    net_name = "VSS";
  }

  // 获取 power_net 的 wire_list
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

    // 将 wire 添加到 wire_list 中
    wire_list->add_wire(wire, idb::IdbWiringStatement::kRouted);

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

void PowerRouter::addVSSNet(idb::IdbDesign* idb_design, GridManager pnp_network)
{
  idb::IdbSpecialNet* vss_net = idb_design->get_special_net_list()->find_net("VSS");

  if (vss_net == nullptr) {
    // 创建 VSS 网络
    idb::IdbSpecialNet* power_net = new idb::IdbSpecialNet();
    power_net->set_net_name("VSS");
    power_net->set_connect_type(idb::IdbConnectType::kGround);
    idb::IdbPin* io_pin = new idb::IdbPin();
    io_pin->set_pin_name("VSS");
    power_net->add_io_pin(io_pin);

    idb_design->get_special_net_list()->add_net(power_net);
    vss_net = power_net;
  }

  addPowerStripes(vss_net, pnp_network);
  std::cout << "[iPNP info]: Add VSS Power Stripes success." << std::endl;

  addPowerFollowPin(idb_design, vss_net);
  std::cout << "[iPNP info]: Add VSS Power Follow Pin success." << std::endl;
}

void PowerRouter::addVDDNet(idb::IdbDesign* idb_design, GridManager pnp_network)
{
  idb::IdbSpecialNet* vdd_net = idb_design->get_special_net_list()->find_net("VDD");

  if (vdd_net == nullptr) {
    // 创建 VDD 网络
    idb::IdbSpecialNet* power_net = new idb::IdbSpecialNet();
    power_net->set_net_name("VDD");
    power_net->set_connect_type(idb::IdbConnectType::kPower);
    idb::IdbPin* io_pin = new idb::IdbPin();
    io_pin->set_pin_name("VDD");
    power_net->add_io_pin(io_pin);

    idb_design->get_special_net_list()->add_net(power_net);
    vdd_net = power_net;
  }

  addPowerStripes(vdd_net, pnp_network);
  std::cout << "[iPNP info]: Add VDD Power Stripes success." << std::endl;

  addPowerFollowPin(idb_design, vdd_net);
  std::cout << "[iPNP info]: Add VDD Power Follow Pin success." << std::endl;
}

void PowerRouter::addPowerNets(idb::IdbDesign* idb_design, GridManager pnp_network)
{
  if (!idb_design) {
    std::cerr << "[iPNP error]: Invalid IDB design object." << std::endl;
    return;
  }

  addVSSNet(idb_design, pnp_network);
  std::cout << "[iPNP info]: Add VSS Power Nets success." << std::endl;

  addVDDNet(idb_design, pnp_network);
  std::cout << "[iPNP info]: Add VDD Power Nets success." << std::endl;
}

}  // namespace ipnp