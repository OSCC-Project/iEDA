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
 * @author Xinhao li
 * @brief
 * @version 0.1
 * @date 2024-07-15
 */

#include "PowerRouter.hh"

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace ipnp {

/**
 * @brief Convert pnp_network of type GridManager to iDB type.
 * @param net_type: Specify whether to write VDD or VSS to IdbSpecialNet.
 * @todo Add feature: power routing while avoiding Macro blocks.
 */
idb::IdbSpecialNet* PowerRouter::createNet(GridManager pnp_network, ipnp::PowerType net_type)
{
  std::string net_name = "VDD";
  bool is_power = true;  // is VDD or VSS
  if (net_type == ipnp::PowerType::kVDD) {
    net_name = "VDD";
    is_power = true;
  } else if (net_type == ipnp::PowerType::kVSS) {
    net_name = "VSS";
    is_power = false;
  } else {
    net_name = "OTHER";
    is_power = false;
  }
  std::cout << "[iPNP info]: add" << net_name << "net." << std::endl;

  idb::IdbSpecialNet* power_net = new idb::IdbSpecialNet();

  // Begin: convert pnp_network to power_net
  power_net->set_net_name(net_name);
  idb::IdbConnectType power_type = is_power ? idb::IdbConnectType::kPower : idb::IdbConnectType::kGround;
  power_net->set_connect_type(power_type);
  idb::IdbPin* io_pin = new idb::IdbPin();
  io_pin->set_pin_name(net_name);
  power_net->add_io_pin(io_pin);

  // Begin: convert pnp_network to wire_list
  idb::IdbSpecialWireList* wire_list = new idb::IdbSpecialWireList();

  // Begin: convert pnp_network to wire
  idb::IdbSpecialWire* wire = new idb::IdbSpecialWire();
  wire->set_wire_state(idb::IdbWiringStatement::kRouted);

  auto grid_data = pnp_network.get_grid_data();
  auto template_data = pnp_network.get_template_data();
  auto power_layers = pnp_network.get_power_layers();

  // @param x_base, y_base: base_coordinates. Should be updated after traversing each region.
  // @brief global coordinates = base_coordinates + coordinates within the region.
  double x_base = 0;
  double y_base = 0;

  for (int layer_idx = 0;layer_idx < pnp_network.get_layer_count();layer_idx++) {
    for (int i = 0; i < pnp_network.get_ho_region_num(); i++) {
      for (int j = 0; j < pnp_network.get_ver_region_num(); j++) {
    
        SingleTemplate& single_template = template_data[layer_idx][i][j];

        idb::IdbSpecialWireSegment* segment = new idb::IdbSpecialWireSegment();

        // Get the parameters of the template.
        double stripe_width = single_template.get_width();
        double stripe_space = single_template.get_space();
        double offset = single_template.get_offset();
        double pg_offset = single_template.get_pg_offset();
        ipnp::PowerType first_stripe_power_type = single_template.get_first_stripe_power_type();

        // is True if the first stripe in the region is of the opposite type of the net_type.
        int is_first_contrary_stripe_type;
        if (net_name == "VDD") {
          is_first_contrary_stripe_type = first_stripe_power_type == PowerType::kVSS ? 1 : 0;
        }
        else {  // net_name == VSS
          is_first_contrary_stripe_type = first_stripe_power_type == PowerType::kVDD ? 1 : 0;
        }

        int count = 0;
        /**
         * @brief The outermost position that can be reached by the current segment's edge when traversing each segment.
         */
        double current_segment_outermost
          = offset + (count + 1) * stripe_width + count * stripe_space
          + is_first_contrary_stripe_type * (pg_offset + stripe_width);

        // Convert single_layer_grid to segment
        if (single_template.get_direction() == StripeDirection::kHorizontal) {
          // Termination conditions: the (count+1)'th wire segment cannot be fully placed in the region.
          while (current_segment_outermost < y_base + grid_data[layer_idx][i][j].get_height()) {
            // Create a new routing layer
            idb::IdbLayer* layer = new idb::IdbLayer();

            // Set the layer name and type  
            layer->set_name("M" + std::to_string(power_layers[layer_idx]));
            layer->set_type(idb::IdbLayerType::kLayerRouting);

            // Set the segment parameters
            segment->set_layer(layer);
            segment->set_route_width((int)stripe_width);
            segment->set_shape_type(idb::IdbWireShapeType::kStripe);

            // Set the segment coordinates
            double stripe_x1 = x_base;
            double stripe_y1 = y_base + current_segment_outermost - 0.5 * stripe_width;
            double stripe_x2 = x_base + grid_data[layer_idx][i][j].get_width();
            double stripe_y2 = stripe_y1;

            // Add the segment points
            segment->add_point((int)stripe_x1, (int)stripe_y1);
            segment->add_point((int)stripe_x2, (int)stripe_y2);

            // Add the segment to the wire
            wire->add_segment(segment);

            // Calculate the outermost position of the next segment
            count++;
            current_segment_outermost
              = offset + (count + 1) * stripe_width + count * stripe_space + is_first_contrary_stripe_type * (pg_offset + stripe_width);
          }
        }
        else {  // direction is vertical
          // Termination conditions: the (count+1)'th wire segment cannot be fully placed in the region.
          while (current_segment_outermost < x_base + grid_data[layer_idx][i][j].get_width()) {
            // Create a new routing layer
            idb::IdbLayer* layer = new idb::IdbLayer();

            // Set the layer name and type
            layer->set_name("M" + std::to_string(power_layers[layer_idx]));
            layer->set_type(idb::IdbLayerType::kLayerRouting);

            // Set the segment parameters
            segment->set_layer(layer);
            segment->set_route_width((int)stripe_width);
            segment->set_shape_type(idb::IdbWireShapeType::kStripe);

            // Set the segment coordinates
            double stripe_x1 = x_base + current_segment_outermost - 0.5 * stripe_width;
            double stripe_y1 = y_base;
            double stripe_x2 = stripe_x1;
            double stripe_y2 = y_base + grid_data[layer_idx][i][j].get_height();

            // Add the segment points
            segment->add_point((int)stripe_x1, (int)stripe_y1);
            segment->add_point((int)stripe_x2, (int)stripe_y2);

            // Add the segment to the wire
            wire->add_segment(segment);

            // Calculate the outermost position of the next segment
            count++;
            current_segment_outermost
              = offset + (count + 1) * stripe_width + count * stripe_space + is_first_contrary_stripe_type * (pg_offset + stripe_width);
          }
        }

        x_base += grid_data[layer_idx][i][j].get_width();
      }
      x_base = 0;
      y_base += grid_data[layer_idx][i][0].get_height();
    }
    x_base = 0;
    y_base += grid_data[layer_idx][0][0].get_height();
  }
  // End: convert pnp_network to wire

  // Add the wire to the wire list
  wire_list->add_wire(wire, idb::IdbWiringStatement::kRouted);
  // End: convert pnp_network to wire_list

  // Add the wire list to the power net
  power_net->set_wire_list(wire_list);
  // End: convert pnp_network to power_net

  return power_net;
}

}  // namespace ipnp