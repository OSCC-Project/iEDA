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
 * @file iPNPIdbWrapper.cpp
 * @author Xinhao li
 * @brief
 * @version 0.1
 * @date 2024-07-15
 */

#include "iPNPIdbWrapper.hh"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "GridManager.hh"
#include "iPNPCommon.hh"
#include "idm.h"

namespace ipnp {

void iPNPIdbWrapper::readFromIdb()
{
  auto* idb_design = dmInst->get_idb_design();
  auto idb_layout = idb_design->get_layout();

  /**
   * @todo read die and macro infomation from iDB
   */
  IdbDie* idb_die = idb_layout->get_die();
  _input_die_llx = idb_die->get_llx();
  _input_die_lly = idb_die->get_lly();
  _input_die_urx = idb_die->get_urx();
  _input_die_ury = idb_die->get_ury();

  /**
   * @todo assign values to _input_db_pdn
   */
}

/**
 * @param net_type: Specify whether to write VDD or VSS to IdbSpecialNet.
 * @return idb::IdbSpecialNet*
 * @attention Will the objects created inside the function be automatically deleted after returning the pointer?
 */
unsigned iPNPIdbWrapper::createNet(GridManager pnp_network, ipnp::PowerType net_type)
{
  std::string net_name = "VDD";
  bool is_power = true;  // VDD or VSS
  if (net_type == ipnp::PowerType::VDD) {
    net_name = "VDD";
    is_power = true;
  } else if (net_type == ipnp::PowerType::VSS) {
    net_name = "VSS";
    is_power = false;
  } else if (net_type == ipnp::PowerType::GROUND) {
    net_name = "GROUND";
    is_power = false;
  } else {
    net_name = "OTHER";
    is_power = false;
  }
  std::cout << "[iPNP info]: add" << net_name << "net." << std::endl;

  idb::IdbSpecialNet* power_net = new idb::IdbSpecialNet();

  /**
   * @brief pnp_network --> power_net
   */
  //
  //
  power_net->set_net_name(net_name);
  idb::IdbConnectType power_type = is_power ? idb::IdbConnectType::kPower : idb::IdbConnectType::kGround;
  power_net->set_connect_type(power_type);
  idb::IdbPin* io_pin = new IdbPin();
  io_pin->set_pin_name(net_name);
  power_net->add_io_pin(io_pin);

  // pnp_network --> wire_list
  idb::IdbSpecialWireList* wire_list = new idb::IdbSpecialWireList();

  /**
   * @brief pnp_network --> wire
   */
  idb::IdbSpecialWire* wire = new idb::IdbSpecialWire();
  wire->set_wire_state(idb::IdbWiringStatement::kRouted);

  auto grid_data = pnp_network.get_grid_data();
  auto template_libs = pnp_network.get_template_libs();
  auto template_data = pnp_network.get_template_data();
  /**
   * @param x_base, y_base: global coordinates = base_coordinates + coordinates within the region.
   * @brief Update the base_coordinates after traversing each region.
   */
  double x_base = 0;
  double y_base = 0;
  for (int i = 0; i < pnp_network.get_ho_region_num(); i++) {
    for (int j = 0; j < pnp_network.get_ver_region_num(); j++) {
      PDNGridTemplate grid_template = template_libs[template_data[i][j]];
      std::vector<int> layers_occupied = grid_template.get_layers_occupied();
      auto grid_per_layer = grid_template.get_grid_per_layer();
      for (int k = 0; k < layers_occupied.size(); k++) {
        idb::IdbSpecialWireSegment* segment = new idb::IdbSpecialWireSegment();
        SingleLayerGrid single_layer_grid = grid_per_layer[layers_occupied[k]];
        double stripe_width = single_layer_grid.get_width();
        double stripe_space = single_layer_grid.get_space();
        double offset = single_layer_grid.get_offset();
        double pg_offset = single_layer_grid.get_pg_offset();
        ipnp::PowerType first_stripe_power_type = single_layer_grid.get_first_stripe_power_type();

        /**
         * @param is_first_contrary_stripe_type: is True if the first stripe in the region is of the opposite type of the net_type.
         */
        int is_first_contrary_stripe_type;
        if (net_name == "VDD") {
          is_first_contrary_stripe_type
              = (first_stripe_power_type == PowerType::VSS || first_stripe_power_type == PowerType::GROUND) ? 1 : 0;
        } else {  // net_name == VSS or GROUND
          is_first_contrary_stripe_type = first_stripe_power_type == PowerType::VDD ? 1 : 0;
        }
        int count = 0;
        if (single_layer_grid.get_direction() == StripeDirection::horizontal) {
          /**
           * @brief Termination conditions: the (count+1)'th wire segment cannot be fully placed in the region
           */
          while (offset + (count + 1) * stripe_width + count * stripe_space + is_first_contrary_stripe_type * (pg_offset + stripe_width)
                 < y_base + grid_data[i][j].get_height()) {
            /**
             * @brief single_layer_grid --> segment
             */
            idb::IdbLayer* layer = new idb::IdbLayer();
            layer->set_name("M" + std::to_string(layers_occupied[k]));
            layer->set_type(IdbLayerType::kLayerRouting);
            segment->set_layer(layer);

            segment->set_route_width((int) stripe_width);

            segment->set_shape_type(IdbWireShapeType::kStripe);

            double stripe_x1 = x_base;
            double stripe_y1 = y_base + offset + (count + 1) * stripe_width + count * stripe_space
                               + is_first_contrary_stripe_type * (pg_offset + stripe_width) - 0.5 * stripe_width;
            double stripe_x2 = x_base + grid_data[i][j].get_width();
            double stripe_y2 = stripe_y1;
            segment->add_point((int) stripe_x1, (int) stripe_y1);
            segment->add_point((int) stripe_x2, (int) stripe_y2);

            wire->add_segment(segment);
            count++;
          }

        } else {  // direction is vertical
          while (offset + (count + 1) * stripe_width + count * stripe_space + is_first_contrary_stripe_type * (pg_offset + stripe_width)
                 < x_base + grid_data[i][j].get_width()) {
            /**
             * @brief single_layer_grid --> segment
             */
            idb::IdbLayer* layer = new idb::IdbLayer();
            layer->set_name("M" + std::to_string(layers_occupied[k]));
            layer->set_type(IdbLayerType::kLayerRouting);
            segment->set_layer(layer);

            segment->set_route_width((int) stripe_width);

            segment->set_shape_type(IdbWireShapeType::kStripe);

            double stripe_x1 = x_base + offset + (count + 1) * stripe_width + count * stripe_space
                               + is_first_contrary_stripe_type * (pg_offset + stripe_width) - 0.5 * stripe_width;
            double stripe_y1 = y_base;
            double stripe_x2 = stripe_x1;
            double stripe_y2 = y_base + grid_data[i][j].get_height();
            segment->add_point((int) stripe_x1, (int) stripe_y1);
            segment->add_point((int) stripe_x2, (int) stripe_y2);

            wire->add_segment(segment);
            count++;
          }
        }
      }
      x_base += grid_data[i][j].get_width();
    }
    x_base = 0;
    y_base += grid_data[i][0].get_height();
  }

  wire_list->add_wire(wire, idb::IdbWiringStatement::kRouted);
  power_net->set_wire_list(wire_list);

  _idb_design->get_special_net_list()->add_net(power_net);

  return 1;
}

void iPNPIdbWrapper::writeToIdb(GridManager pnp_network)
{
  auto idb_design = dmInst->get_idb_design();

  createNet(pnp_network, ipnp::PowerType::VDD);
  createNet(pnp_network, ipnp::PowerType::VSS);

  std::cout << "[iPNP info]: Added iPNP net." << std::endl;
  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  /**
   * @brief save DEF
   */
  auto idb_builder = dmInst->get_idb_builder();
  idb_builder->saveDef("def file path", idb::DefWriteType::kChip);  // kChip?
}

}  // namespace ipnp
