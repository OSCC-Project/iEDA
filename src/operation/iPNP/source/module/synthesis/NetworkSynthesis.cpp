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
 * @author Xinhao li
 * @brief
 * @version 0.1
 * @date 2024-07-15
 */

#include "NetworkSynthesis.hh"

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "idm.h"

namespace ipnp {

NetworkSynthesis::NetworkSynthesis(SysnType sysn_type, GridManager grid_info)
  : _network_sys_type(sysn_type),
  _input_grid_info(grid_info),
  _synthesized_network(grid_info)
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
      // _synthesized_network.set_grid_data(_input_grid_info.get_grid_data());
      // _synthesized_network.set_template_data(_input_grid_info.get_template_data());
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
  
  // distribute templates to each layer
  for (int layer_idx = 0; layer_idx < layer_count; ++layer_idx) {
    int power_layer = power_layers[layer_idx];
    bool use_horizontal = (layer_idx % 2 == 1);
    
    std::cout << "Layer " << power_layer  << " Using "
              << (use_horizontal ? "horizontal" : "vertical") << " templates" << std::endl;
    
    for (int i = 0; i < ho_region_num; ++i) {
      for (int j = 0; j < ver_region_num; ++j) {
        if (use_horizontal) {
          // use horizontal template
          _synthesized_network.set_single_template(layer_idx, i, j, horizontal_templates[1]);
          if (layer_idx == 5 && i == 1 && j == 0) {
            SingleTemplate tmp;
            tmp.set_direction(StripeDirection::kHorizontal);
            tmp.set_width(8000.0);
            tmp.set_pg_offset(1600.0);
            tmp.set_space(38400.0);
            tmp.set_offset(27200.0);
            _synthesized_network.set_single_template(layer_idx, i, j, tmp);
          }
          if (layer_idx == 5 && i == 1 && j == 1) {
            SingleTemplate tmp;
            tmp.set_direction(StripeDirection::kHorizontal);
            tmp.set_width(8000.0);
            tmp.set_pg_offset(1600.0);
            tmp.set_space(38400.0);
            tmp.set_offset(27200.0);
            _synthesized_network.set_single_template(layer_idx, i, j, tmp);
          }
        }
        else {
          // use vertical template
          _synthesized_network.set_single_template(layer_idx, i, j, vertical_templates[1]);
          // M7层暂时特殊处理
          if (layer_idx == 2) {
            SingleTemplate template_m7;
            template_m7.set_direction(StripeDirection::kVertical);
            template_m7.set_width(900.0);
            template_m7.set_pg_offset(1600.0);
            template_m7.set_space(19200.0);
            template_m7.set_offset(8000.0);
            _synthesized_network.set_single_template(layer_idx, i, j, template_m7);
          }
          if (layer_idx == 6 && i == 1 && j == 2) {
            SingleTemplate tmp;
            tmp.set_direction(StripeDirection::kVertical);
            tmp.set_width(8000.0);
            tmp.set_pg_offset(1600.0);
            tmp.set_space(19200.0);
            tmp.set_offset(8000.0);
            _synthesized_network.set_single_template(layer_idx, i, j, tmp);
          }
          
        }
      }
    }
  }

  std::cout << "Manual template setting completed with alternating horizontal and vertical templates." << std::endl;
}

}  // namespace ipnp
