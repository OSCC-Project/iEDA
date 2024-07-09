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
#include "NetworkSynthesis.hh"

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace idb {
class IdbRegularWire;
}

namespace ipnp {

NetworkSynthesis::NetworkSynthesis(std::string type, GridManager grid_info)
    : _nework_sys_type(type), _input_grid_info(grid_info), _synthesized_network(grid_info)  // list initial
{
}

NetworkSynthesis::~NetworkSynthesis()
{
}

void NetworkSynthesis::synthesizeNetwork()
{
  if (_nework_sys_type == "default") {
    randomSys();
  }
  if (_nework_sys_type == "optimizer") {
    // // std::vector<std::vector<PDNGridRegion>> grid_data;
    // std::vector<std::vector<PDNGridRegion>> input_grid_data = _input_grid_info.get_grid_data();
    // // std::vector<std::vector<int>> template_data;
    // std::vector<std::vector<int>> input_template_data = _input_grid_info.get_template_data();
    // // for (int i = 0; i < _input_grid_info.getHoRegionNum(); i++) {
    // //   for (int j = 0; j < _input_grid_info.getVerRegionNum(); j++) {
    // //     grid_data[i][j] = input_grid_data[i][j];
    // //     template_data[i][j] = input_template_data[i][j];
    // //   }
    // // }
    _synthesized_network.set_grid_data(_input_grid_info.get_grid_data());
    _synthesized_network.set_template_data(_input_grid_info.get_template_data());
  }
  if (_nework_sys_type == "best") {
    // TODO
  }
  if (_nework_sys_type == "worst") {
    // TODO
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
  random_grid_region.set_width(_synthesized_network.get_chip_width() / ho_region_num);
  random_grid_region.set_height(_synthesized_network.get_chip_height() / ver_region_num);
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

idb::IdbRegularWire* NetworkSynthesis::writeDef()
{
  // TODO: _synthesized_network --> def_file
  // Consider the situation that the region is irregular

  idb::IdbRegularWire* DEF;
  return DEF;
}

}  // namespace ipnp
