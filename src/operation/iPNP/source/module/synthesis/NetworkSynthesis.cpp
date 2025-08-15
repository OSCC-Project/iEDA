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
 * @file NetworkSynthesis.cpp
 * @author Jianrong Su
 * @brief
 * @version 1.0
 * @date 2025-06-23
 */

#include "NetworkSynthesis.hh"

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "IdbLayer.h"
#include "IdbSpecialNet.h"
#include "IdbSpecialWire.h"
#include "IdbVias.h"
#include "Log.hh"
#include "idm.h"

namespace ipnp {

NetworkSynthesis::NetworkSynthesis(SysnType sysn_type, PNPGridManager grid_info)
    : _network_sys_type(sysn_type), _input_grid_info(grid_info), _synthesized_network(grid_info)
{
  // initialize the synthesized network
  _synthesized_network.set_power_layers(_input_grid_info.get_power_layers());
  _synthesized_network.set_ho_region_num(_input_grid_info.get_ho_region_num());
  _synthesized_network.set_ver_region_num(_input_grid_info.get_ver_region_num());
  _synthesized_network.set_core_width(_input_grid_info.get_core_width());
  _synthesized_network.set_core_height(_input_grid_info.get_core_height());
  _synthesized_network.set_grid_data(_input_grid_info.get_grid_data());
}

void NetworkSynthesis::synthesizeNetwork()
{
  switch (_network_sys_type) {
    case SysnType::kDefault:
      manualSetTemplates();
      break;
    case SysnType::kOptimizer:
      /**
       * @todo
       */
      break;
    case SysnType::kBest:
      /**
       * @todo
       */
      break;
    case SysnType::kWorst:
      /**
       * @todo
       */
      break;
    default:
      break;
  }
}

void NetworkSynthesis::manualSetTemplates()
{
  int layer_count = _input_grid_info.get_layer_count();
  int ho_region_num = _input_grid_info.get_ho_region_num();
  int ver_region_num = _input_grid_info.get_ver_region_num();
  auto power_layers = _input_grid_info.get_power_layers();

  auto horizontal_templates = _input_grid_info.get_horizontal_templates();
  auto vertical_templates = _input_grid_info.get_vertical_templates();
  auto layer_specific_templates = pnpConfig->get_layer_specific_templates();

  auto* idb_layers = dmInst->get_idb_layout()->get_layers();

  LOG_INFO << "Setting templates for " << layer_count << " power layers based on their actual directions";

  // distribute templates to each layer
  for (int layer_idx = 0; layer_idx < layer_count; ++layer_idx) {
    int power_layer_id = power_layers[layer_idx];
    
    auto* routing_layer = idb_layers->find_routing_layer(power_layer_id);
    auto* idb_layer_routing = dynamic_cast<idb::IdbLayerRouting*>(routing_layer);
    
    bool use_horizontal = idb_layer_routing->is_horizontal();
    std::string layer_name = routing_layer->get_name();
    
    LOG_INFO << "Layer " << layer_name << " (ID: " << power_layer_id << ") Using " 
             << (use_horizontal ? "horizontal" : "vertical") << " direction from layer properties";

    for (int i = 0; i < ho_region_num; ++i) {
      for (int j = 0; j < ver_region_num; ++j) {
        SingleTemplate template_to_use;
        
        auto it = layer_specific_templates.find(layer_name);
        if (it != layer_specific_templates.end()) {
          template_to_use.set_direction(it->second.direction == "horizontal" ? 
                                      StripeDirection::kHorizontal : StripeDirection::kVertical);
          template_to_use.set_width(it->second.width);
          template_to_use.set_pg_offset(it->second.pg_offset);
          template_to_use.set_space(it->second.space);
          template_to_use.set_offset(it->second.offset);
          
          LOG_INFO << "Using layer-specific template for " << layer_name;
        } else {
          if (use_horizontal) {
            template_to_use = horizontal_templates[0];
          } else {
            template_to_use = vertical_templates[0];
          }
          
          LOG_INFO << "Using default template for " << layer_name;
        }
        
        _synthesized_network.set_single_template(layer_idx, i, j, template_to_use);
      }
    }
  }

  LOG_INFO << "Manual template setting completed based on actual layer directions from power_layers configuration.";
}

}  // namespace ipnp
