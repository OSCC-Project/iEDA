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
    : _nework_sys_type(sysn_type), _input_grid_info(grid_info), _synthesized_network(grid_info)
{
}

void NetworkSynthesis::synthesizeNetwork()
{
  switch (_nework_sys_type) {
    case SysnType::kDefault:
      randomSys();
      break;
    case SysnType::kOptimizer:
      _synthesized_network.set_grid_data(_input_grid_info.get_grid_data());
      _synthesized_network.set_template_data(_input_grid_info.get_template_data());
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

void NetworkSynthesis::randomSys()
{
  std::vector<std::vector<PDNRectanGridRegion>> grid_data;
  std::vector<std::vector<int>> template_data;
  PDNRectanGridRegion random_grid_region;

  std::srand(std::time(NULL));
  int ho_region_num = 3 + std::rand() % (9 - 3 + 1);
  int ver_region_num = 3 + std::rand() % (9 - 3 + 1);

  /**
   * @todo grid coordinate
   */
  // random_grid_region.set_width(_synthesized_network.get_chip_width() / ho_region_num);
  // random_grid_region.set_height(_synthesized_network.get_chip_height() / ver_region_num);
  _synthesized_network.set_ho_region_num(ho_region_num);
  _synthesized_network.set_ver_region_num(ver_region_num);

  for (int i = 0; i < _input_grid_info.get_ho_region_num(); i++) {
    for (int j = 0; j < _input_grid_info.get_ver_region_num(); j++) {
      grid_data[i][j] = random_grid_region;
      std::srand(std::time(NULL));
      template_data[i][j] = 1 + std::rand() % (_input_grid_info.get_template_libs().size() - 1);
    }
  }

  _synthesized_network.set_grid_data(grid_data);
  _synthesized_network.set_template_data(template_data);
}

}  // namespace ipnp
