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
#include "py_ipdn.h"

#include "ipdn_api.h"

namespace python_interface {

bool pdnAddIO(const std::string& pin_name, const std::string& net_name, const std::string& direction, bool is_power)
{
  const std::string pin = pin_name.empty() ? net_name : pin_name;
  pdnApiInst->addIOPin(pin, net_name, direction, is_power);
  return true;
}

bool pdnGlobalConnect(const std::string& net_name, const std::string& instance_pin_name, bool is_power)
{
  pdnApiInst->globalConnect(net_name, instance_pin_name, is_power);
  return true;
}

bool pdnPlacePort(const std::string& pin_name, const std::string& io_cell_name, int offset_x, int offset_y, int width, int height,
                  const std::string& layer)
{
  pdnApiInst->placePdnPort(pin_name, io_cell_name, offset_x, offset_y, width, height, layer);
  return true;
}

bool pdnCreateGrid(const std::string& layer_name, const std::string& net_name_power, const std::string& net_name_ground, double width,
                   double offset)
{
  pdnApiInst->createGrid(net_name_power, net_name_ground, layer_name, width, offset);
  return true;
}

bool pdnCreateStripe(const std::string& layer_name, const std::string& net_name_power, const std::string& net_name_ground, double width,
                     double pitch, double offset)
{
  pdnApiInst->createStripe(net_name_power, net_name_ground, layer_name, width, pitch, offset);
  return true;
}

bool pdnConnectLayer(std::vector<std::string>& layers)
{
  pdnApiInst->connectLayerList(layers);
  return true;
}

bool pdnConnectMacro(const std::string& pin_layer, const std::string& pdn_layer, std::vector<std::string>& power_pins,
                     std::vector<std::string>& ground_pins, const std::string& orient)
{
  pdnApiInst->connectMacroToPdnGrid(power_pins, ground_pins, pin_layer, pdn_layer, orient);
  return true;
}

bool pdnConnectIOPin(std::vector<double>& point_list, const std::string& layer)
{
  pdnApiInst->connectIOPinToPowerStripe(point_list, layer);
  return true;
}

bool pdnConnectStripe(std::vector<double>& point_list, const std::string& net_name, const std::string& layer, int32_t width)
{
  pdnApiInst->connectPowerStripe(point_list, net_name, layer, width);
  return true;
}

bool pdnAddSegmentStripe(const std::string& net_name, std::vector<double>& point_list, const std::string& layer, int32_t width,
                         std::vector<double>& point_begin, std::string& layer_start, std::vector<double>& point_end,
                         const std::string& layer_end, int32_t via_width, int32_t via_height)
{
  if (not point_list.empty()) {
    pdnApiInst->addSegmentStripeList(point_list, net_name, layer, width);
  }
  if (!(point_begin.empty() || point_end.empty())) {
    std::vector<double> points;
    points.insert(points.end(), point_begin.begin(), point_end.end());
    pdnApiInst->addSegmentStripeList(points, net_name, layer_start, width);
    pdnApiInst->addSegmentVia(net_name, layer_start, layer_end, point_end[0], point_end[1], via_width, via_height);
  }
  return true;
}

bool pdnAddSegmentVia(const std::string& net_name, const std::string& layer, const std::string& top_layer, const std::string& bottom_layer,
                      int32_t offset_x, int32_t offset_y, int32_t width, int32_t height)
{
  if (layer.empty() && (top_layer.empty() || bottom_layer.empty())) {
    return false;
  }
  // single cutlayer
  if (not layer.empty()) {
    pdnApiInst->addSegmentVia(net_name, layer, offset_x, offset_y, width, height);
  }
  if (!top_layer.empty() && !bottom_layer.empty()) {
    pdnApiInst->addSegmentVia(net_name, top_layer, bottom_layer, offset_x, offset_y, width, height);
  }
  return true;
}
}  // namespace python_interface