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
#pragma once

#include <string>
#include <vector>

namespace python_interface {

bool pdnAddIO(const std::string& pin_name, const std::string& net_name, const std::string& direction, bool is_power);
bool pdnGlobalConnect(const std::string& net_name, const std::string& instance_pin_name, bool is_power);
bool pdnPlacePort(const std::string& pin_name, const std::string& io_cell_name, int offset_x, int offset_y, int width, int height,
                  const std::string& layer);
bool pdnCreateGrid(const std::string& layer_name, const std::string& net_name_power, const std::string& net_name_ground, double width,
                   double offset);
bool pdnCreateStripe(const std::string& layer_name, const std::string& net_name_power, const std::string& net_name_ground, double width,
                     double pitch, double offset);
bool pdnConnectLayer(std::vector<std::string>& layers);
bool pdnConnectMacro(const std::string& pin_layer, const std::string& pdn_layer, std::vector<std::string>& power_pins,
                     std::vector<std::string>& ground_pins, const std::string& orient);
bool pdnConnectIOPin(std::vector<double>& point_list, const std::string& layer);
bool pdnConnectStripe(std::vector<double>& point_list, const std::string& net_name, const std::string& layer, int32_t width);
bool pdnAddSegmentStripe(const std::string& net_name, std::vector<double>& point_list, const std::string& layer, int32_t width,
                         std::vector<double>& point_begin, std::string& layer_start, std::vector<double>& point_end,
                         const std::string& layer_end, int32_t via_width, int32_t via_height);

bool pdnAddSegmentVia(const std::string& net_name, const std::string& layer, const std::string& top_layer, const std::string& bottom_layer,
                      int32_t offset_x, int32_t offset_y, int32_t width, int32_t height);
}  // namespace python_interface