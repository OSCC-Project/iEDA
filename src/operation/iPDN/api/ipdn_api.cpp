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
#include "ipdn_api.h"

#include "IdbDesign.h"
#include "builder.h"
#include "idm.h"
#include "pdn_plan.h"
#include "pdn_via.h"

namespace ipdn {
PdnApi* PdnApi::_instance = nullptr;

void PdnApi::addIOPin(std::string pin_name, std::string net_name, std::string direction, bool is_power)
{
  PdnPlan pdn_plan;

  pdn_plan.addIOPin(pin_name, net_name, direction, is_power);
}

void PdnApi::globalConnect(const std::string pdn_net_name, const std::string instance_pdn_pin_name, bool is_power)
{
  PdnPlan pdn_plan;

  pdn_plan.globalConnect(pdn_net_name, instance_pdn_pin_name, is_power);
}

void PdnApi::placePdnPort(std::string pin_name, std::string io_cell_name, int32_t offset_x, int32_t offset_y, int32_t width, int32_t height,
                          std::string layer_name)
{
  PdnPlan pdn_plan;

  pdn_plan.placePdnPort(pin_name, io_cell_name, offset_x, offset_y, width, height, layer_name);
}

void PdnApi::createGrid(std::string power_net_name, std::string ground_net_name, std::string layer_name, double route_width,
                        double route_offset)
{
  PdnPlan pdn_plan;

  pdn_plan.createGrid(power_net_name, ground_net_name, layer_name, route_width, route_offset);
}
/**
 * @brief 创建电源网络
 *
 * @param power_net_name
 * @param ground_net_name
 * @param layer_name
 * @param route_width
 * @param pitchh 电源线间距
 * @param route_offset 电源线起始偏移量
 */
void PdnApi::createStripe(std::string power_net_name, std::string ground_net_name, std::string layer_name, double route_width,
                          double pitchh, double route_offset)
{
  PdnPlan pdn_plan;

  pdn_plan.createStripe(power_net_name, ground_net_name, layer_name, route_width, pitchh, route_offset);
}

void PdnApi::connectLayerList(std::vector<std::string>& layer_list)
{
  int number = layer_list.size();
  if (number < 2) {
    std::cout << "Error : please connect at least 2 layers." << std::endl;
    return;
  }

  PdnPlan pdn_plan;
  pdn_plan.updateRouteMap();

  for (size_t i = 0; i < (layer_list.size() - 1); i += 2) {
    std::string layer_name_first = layer_list[i];
    std::string layer_name_second = layer_list[i + 1];

    pdn_plan.connectLayer(layer_name_first, layer_name_second);
  }
}

void PdnApi::connectMacroToPdnGrid(std::vector<std::string> power_name, std::vector<std::string> ground_name, std::string layer_name_first,
                                   std::string layer_name_second, std::string orient)
{
  PdnPlan pdn_plan;

  pdn_plan.connectMacroToPdnGrid(power_name, ground_name, layer_name_first, layer_name_second, orient);
}

void PdnApi::connectIOPinToPowerStripe(std::vector<double>& point_list, const std::string layer_name)
{
  PdnPlan pdn_plan;

  pdn_plan.connectIOPinToPowerStripe(point_list, layer_name);
}

void PdnApi::connectPowerStripe(std::vector<double>& point_list, const std::string& net_name, const std::string& layer_name, int32_t width)
{
  PdnPlan pdn_plan;

  pdn_plan.connectPowerStripe(point_list, net_name, layer_name, width);
}

bool PdnApi::addSegmentStripeList(std::vector<double>& point_list, std::string net_name, std::string layer_name, int32_t width)
{
  PdnPlan pdn_plan;

  return pdn_plan.addSegmentStripeList(point_list, net_name, layer_name, width);
}

bool PdnApi::addSegmentVia(std::string net_name, std::string top_metal, std::string bottom_metal, double coord_x, double coord_y,
                           int32_t width, int32_t height)
{
  PdnVia pdn_via;

  return pdn_via.addSegmentVia(net_name, top_metal, bottom_metal, coord_x, coord_y, width, height);
}

bool PdnApi::addSegmentVia(std::string net_name, std::string top_metal, std::string bottom_metal, int32_t coord_x, int32_t coord_y,
                           int32_t width, int32_t height)
{
  PdnVia pdn_via;

  return pdn_via.addSegmentVia(net_name, top_metal, bottom_metal, coord_x, coord_y, width, height);
}

bool PdnApi::addSegmentVia(std::string net_name, std::string cut_layer_name, int32_t coord_x, int32_t coord_y, int32_t width,
                           int32_t height)
{
  PdnVia pdn_via;

  return pdn_via.addSegmentVia(net_name, cut_layer_name, coord_x, coord_y, width, height);
}

}  // namespace ipdn
