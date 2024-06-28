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

#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace ipnp {

NetworkSynthesis::NetworkSynthesis(std::string type, GridManager grid_info)
{
  _nework_sys_type = type;
  _input_grid_info = grid_info;
  _synthesized_network.write_ho_region_num(_input_grid_info.get_ho_region_num());
  _synthesized_network.write_ver_region_num(_input_grid_info.get_ver_region_num());
  _synthesized_network.write_template_libs(_input_grid_info.get_template_libs());
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
    // std::vector<std::vector<PDNGridRegion>> grid_data;
    std::vector<std::vector<PDNGridRegion>> input_grid_data = _input_grid_info.get_grid_data();
    // std::vector<std::vector<int>> template_data;
    std::vector<std::vector<int>> input_template_data = _input_grid_info.get_template_data();
    // for (int i = 0; i < _input_grid_info.getHoRegionNum(); i++) {
    //   for (int j = 0; j < _input_grid_info.getVerRegionNum(); j++) {
    //     grid_data[i][j] = input_grid_data[i][j];
    //     template_data[i][j] = input_template_data[i][j];
    //   }
    // }
    _synthesized_network.write_grid_data(input_grid_data);
    _synthesized_network.write_template_data(input_template_data);
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
  for (int i = 0; i < _input_grid_info.ho_region_num; i++) {
    for (int j = 0; j < _input_grid_info.ver_region_num; j++) {
      _synthesized_network._grid_data[i][j] = Random(PDNGridRegion);  // TODO
      _synthesized_network._template_data[i][j]
          = std::uniform_int_distribution<> distrib(0, _synthesized_network._template_libs.size() - 1);  // TODO: random Template. 
    }
  }
}

file* NetworkSynthesis::writeDef()
{
  // Consider the situation that the region is irregular
  def_file < -_synthesized_network;  // TODO
}

}  // namespace ipnp
