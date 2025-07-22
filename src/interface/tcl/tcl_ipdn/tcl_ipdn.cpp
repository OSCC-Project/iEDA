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
#include "tcl_ipdn.h"

#include "Str.hh"
// #include "iPNPApi.hh"
#include "idm.h"
#include "ipdn_api.h"
#include "tool_manager.h"

namespace tcl {

TclPdnAddIO::TclPdnAddIO(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* net_name = new TclStringOption("-net_name", 0);
  auto* pin_name = new TclStringOption("-pin_name", 0);
  auto* direction = new TclStringOption("-direction", 0);  // { INPUT | OUTPUT | INOUT | FEEDTHRU }
  // auto *use = new TclIntOption("-use", 0);
  auto* is_power = new TclIntOption("-is_power", 0);
  addOption(pin_name);
  addOption(net_name);
  addOption(direction);
  addOption(is_power);
  // addOption(use);
}

unsigned TclPdnAddIO::check()
{
  TclOption* pin_name = getOptionOrArg("-pin_name");
  TclOption* net_name = getOptionOrArg("-net_name");
  TclOption* direction = getOptionOrArg("-direction");
  TclOption* is_power = getOptionOrArg("-is_power");
  // TclOption *use = getOptionOrArg("-use");

  LOG_FATAL_IF(!pin_name);
  LOG_FATAL_IF(!net_name);
  LOG_FATAL_IF(!direction);
  LOG_FATAL_IF(!is_power);

  return 1;
}

unsigned TclPdnAddIO::exec()
{
  if (!check()) {
    return 0;
  }
  TclOption* net_name = getOptionOrArg("-net_name");
  TclOption* pin_name = getOptionOrArg("-pin_name");
  TclOption* direction = getOptionOrArg("-direction");
  TclOption* is_power = getOptionOrArg("-is_power");

  string net = net_name->getStringVal();
  string pin;
  if (pin_name->is_set_val() == 0) {
    pin = net;
  } else {
    pin = pin_name->getStringVal();
  }
  string dir = direction->getStringVal();
  bool power = is_power->getIntVal() == 0 ? false : true;

  pdnApiInst->addIOPin(pin, net, dir, power);
  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TclPdnGlobalConnect::TclPdnGlobalConnect(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* net_name = new TclStringOption("-net_name", 0, nullptr);
  auto* pin_name = new TclStringOption("-instance_pin_name", 0, nullptr);
  auto* is_power = new TclIntOption("-is_power", 0);

  addOption(net_name);
  addOption(pin_name);
  addOption(is_power);
}

unsigned TclPdnGlobalConnect::check()
{
  TclOption* net_name = getOptionOrArg("-net_name");
  TclOption* pin_name = getOptionOrArg("-instance_pin_name");
  TclOption* is_power = getOptionOrArg("-is_power");

  LOG_FATAL_IF(!net_name);
  LOG_FATAL_IF(!pin_name);
  LOG_FATAL_IF(!is_power);
  // LOG_FATAL_IF(!io_sitev);
  return 1;
}

unsigned TclPdnGlobalConnect::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* net_name = getOptionOrArg("-net_name");
  TclOption* pin_name = getOptionOrArg("-instance_pin_name");
  TclOption* is_power = getOptionOrArg("-is_power");

  string net = net_name->getStringVal();
  string pin = pin_name->getStringVal();
  int32_t is_p = is_power->getIntVal();

  pdnApiInst->globalConnect(net, pin, is_p);

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TclPdnPlacePort::TclPdnPlacePort(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* pin_name = new TclStringOption("-pin_name", 0);
  auto* io_cell_name = new TclStringOption("-io_cell_name", 0);
  auto* x = new TclIntOption("-offset_x", 0);
  auto* y = new TclIntOption("-offset_y", 0);
  auto* width = new TclIntOption("-width", 0);
  auto* height = new TclIntOption("-height", 0);
  auto* layer_name = new TclStringOption("-layer", 0);
  addOption(pin_name);
  addOption(io_cell_name);
  addOption(x);
  addOption(y);
  addOption(width);
  addOption(height);
  addOption(layer_name);
}

unsigned TclPdnPlacePort::check()
{
  TclOption* pin_name = getOptionOrArg("-pin_name");
  TclOption* io_cell_name = getOptionOrArg("-io_cell_name");
  TclOption* x = getOptionOrArg("-offset_x");
  TclOption* y = getOptionOrArg("-offset_y");
  TclOption* width = getOptionOrArg("-width");
  TclOption* height = getOptionOrArg("-height");
  TclOption* layer_namev = getOptionOrArg("-layer");
  LOG_FATAL_IF(!pin_name);
  LOG_FATAL_IF(!io_cell_name);
  LOG_FATAL_IF(!x);
  LOG_FATAL_IF(!y);
  LOG_FATAL_IF(!layer_namev);
  LOG_FATAL_IF(!width);
  LOG_FATAL_IF(!height);
  return 1;
  // return (pin_name && llx && lly && width && height && layer_namev);
}

unsigned TclPdnPlacePort::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* pin_name = getOptionOrArg("-pin_name");
  TclOption* io_cell_name = getOptionOrArg("-io_cell_name");
  TclOption* x = getOptionOrArg("-offset_x");
  TclOption* y = getOptionOrArg("-offset_y");
  TclOption* width = getOptionOrArg("-width");
  TclOption* height = getOptionOrArg("-height");
  TclOption* layer_namev = getOptionOrArg("-layer");

  auto value_pin = pin_name->getStringVal();
  auto value_io_cell = io_cell_name->getStringVal();

  auto value_offset_x = x->getIntVal();
  auto value_offset_y = y->getIntVal();
  auto value_width = width->getIntVal();
  auto value_height = height->getIntVal();
  auto value_layer_name = layer_namev->getStringVal();

  pdnApiInst->placePdnPort(value_pin, value_io_cell, value_offset_x, value_offset_y, value_width, value_height, value_layer_name);

  return 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TclPdnCreateGrid::TclPdnCreateGrid(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* layer_name = new TclStringOption("-layer_name", 0, nullptr);
  auto* power_net_name = new TclStringOption("-net_name_power", 0, nullptr);
  auto* ground_net_name = new TclStringOption("-net_name_ground", 0);
  auto* width = new TclDoubleOption("-width", 0, 0);
  auto* offset = new TclDoubleOption("-offset", 0, 0);

  addOption(layer_name);
  addOption(power_net_name);
  addOption(ground_net_name);
  addOption(width);
  addOption(offset);
}
unsigned TclPdnCreateGrid::check()
{
  TclOption* layer_name = getOptionOrArg("-layer_name");
  TclOption* power_net_name = getOptionOrArg("-net_name_power");
  TclOption* ground_net_name = getOptionOrArg("-net_name_ground");
  TclOption* width = getOptionOrArg("-width");
  TclOption* offset = getOptionOrArg("-offset");

  LOG_FATAL_IF(!layer_name);
  LOG_FATAL_IF(!power_net_name);
  LOG_FATAL_IF(!ground_net_name);
  LOG_FATAL_IF(!width);
  LOG_FATAL_IF(!offset);

  return 1;
}
unsigned TclPdnCreateGrid::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* layer_name = getOptionOrArg("-layer_name");
  TclOption* power_net_name = getOptionOrArg("-net_name_power");
  TclOption* ground_net_name = getOptionOrArg("-net_name_ground");
  TclOption* width_name = getOptionOrArg("-width");
  TclOption* offset_name = getOptionOrArg("-offset");

  string layer = layer_name->getStringVal();
  string net_power = power_net_name->getStringVal();
  string net_ground = ground_net_name->getStringVal();
  double width = width_name->getDoubleVal();
  double offset = offset_name->getDoubleVal();

  pdnApiInst->createGrid(net_power, net_ground, layer, width, offset);

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TclPdnCreateStripe::TclPdnCreateStripe(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* layer_name = new TclStringOption("-layer_name", 0, nullptr);
  auto* power_net_name = new TclStringOption("-net_name_power", 0);
  auto* ground_net_name = new TclStringOption("-net_name_ground", 0);
  auto* width = new TclDoubleOption("-width", 0, 0);
  auto* pitch = new TclDoubleOption("-pitch", 0, 0);
  auto* offset = new TclDoubleOption("-offset", 0, 0);

  addOption(layer_name);
  addOption(power_net_name);
  addOption(ground_net_name);
  addOption(width);
  addOption(pitch);
  addOption(offset);
}

unsigned TclPdnCreateStripe::check()
{
  TclOption* layer_name = getOptionOrArg("-layer_name");
  TclOption* power_net_name = getOptionOrArg("-net_name_power");
  TclOption* ground_net_name = getOptionOrArg("-net_name_ground");
  TclOption* width = getOptionOrArg("-width");
  TclOption* pitch = getOptionOrArg("-pitch");
  TclOption* offset = getOptionOrArg("-offset");

  LOG_FATAL_IF(!layer_name);
  LOG_FATAL_IF(!power_net_name);
  LOG_FATAL_IF(!ground_net_name);
  LOG_FATAL_IF(!width);
  LOG_FATAL_IF(!pitch);
  LOG_FATAL_IF(!offset);
  // LOG_FATAL_IF(!io_sitev);
  return 1;
}

unsigned TclPdnCreateStripe::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* layer_name = getOptionOrArg("-layer_name");
  TclOption* power_net_name = getOptionOrArg("-net_name_power");
  TclOption* ground_net_name = getOptionOrArg("-net_name_ground");
  TclOption* width = getOptionOrArg("-width");
  TclOption* pitch = getOptionOrArg("-pitch");
  TclOption* offset = getOptionOrArg("-offset");

  string layer = layer_name->getStringVal();
  string pow_pin = power_net_name->getStringVal();
  string grd_pin = ground_net_name->getStringVal();
  double wid = width->getDoubleVal();
  double pit = pitch->getDoubleVal();
  double off = offset->getDoubleVal();

  pdnApiInst->createStripe(pow_pin, grd_pin, layer, wid, pit, off);

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TclPdnConnectLayer::TclPdnConnectLayer(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* layers = new TclStringListOption("-layers", 0);
  // auto *layer_second = new TclStringOption("-second_layer_name", 0, nullptr);

  addOption(layers);
  // addOption(layer_second);
}

unsigned TclPdnConnectLayer::check()
{
  TclOption* layer_first = getOptionOrArg("-layers");
  // TclOption *layer_second = getOptionOrArg("-second_layer_name");

  LOG_FATAL_IF(!layer_first);
  // LOG_FATAL_IF(!layer_second);

  return 1;
}

unsigned TclPdnConnectLayer::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* layers = getOptionOrArg("-layers");

  std::vector<std::string> layer_list = layers->getStringList();

  pdnApiInst->connectLayerList(layer_list);

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TclPdnConnectMacro::TclPdnConnectMacro(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* pin_layer = new TclStringOption("-pin_layer", 0);
  auto* pdn_layer = new TclStringOption("-pdn_layer", 0);
  auto* power_pins = new TclStringOption("-power_pins", 0);
  auto* ground_pins = new TclStringOption("-ground_pins", 0);
  auto* orient = new TclStringOption("-orient", 0);

  addOption(pin_layer);
  addOption(pdn_layer);
  addOption(power_pins);
  addOption(ground_pins);
  addOption(orient);
}

unsigned TclPdnConnectMacro::check()
{
  TclOption* pin_layer = getOptionOrArg("-pin_layer");
  TclOption* pdn_layer = getOptionOrArg("-pdn_layer");
  TclOption* power_pins = getOptionOrArg("-power_pins");
  TclOption* ground_pins = getOptionOrArg("-ground_pins");
  TclOption* orient = getOptionOrArg("-orient");

  LOG_FATAL_IF(!pin_layer);
  LOG_FATAL_IF(!pdn_layer);
  LOG_FATAL_IF(!power_pins);
  LOG_FATAL_IF(!ground_pins);
  LOG_FATAL_IF(!orient);

  return 1;
}

unsigned TclPdnConnectMacro::exec()
{
  if (!check()) {
    return 0;
  }
  TclOption* tcl_pin_layer = getOptionOrArg("-pin_layer");
  TclOption* tcl_pdn_layer = getOptionOrArg("-pdn_layer");
  TclOption* tcl_power_pins = getOptionOrArg("-power_pins");
  TclOption* tcl_ground_pins = getOptionOrArg("-ground_pins");
  TclOption* tcl_orient = getOptionOrArg("-orient");

  ieda::Str str = ieda::Str();

  std::string pin_layer = tcl_pin_layer->getStringVal();
  std::string pdn_layer = tcl_pdn_layer->getStringVal();
  std::vector<std::string> power_list = str.split(tcl_power_pins->getStringVal(), " ");
  std::vector<std::string> ground_list = str.split(tcl_ground_pins->getStringVal(), " ");
  std::string orient = tcl_orient->getStringVal();

  pdnApiInst->connectMacroToPdnGrid(power_list, ground_list, pin_layer, pdn_layer, orient);

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TclPdnConnectIOPin::TclPdnConnectIOPin(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* point_list = new TclStringOption("-point_list", 0);
  auto* layer = new TclStringOption("-layer", 0);

  addOption(point_list);
  addOption(layer);
}

unsigned TclPdnConnectIOPin::check()
{
  TclOption* point_list = getOptionOrArg("-point_list");
  TclOption* layer = getOptionOrArg("-layer");

  LOG_FATAL_IF(!point_list);
  LOG_FATAL_IF(!layer);

  return 1;
}

unsigned TclPdnConnectIOPin::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* point_list = getOptionOrArg("-point_list");
  TclOption* layer_name = getOptionOrArg("-layer");
  ieda::Str str = ieda::Str();
  string layer = layer_name->getStringVal();
  vector<double> points = str.splitDouble(point_list->getStringVal(), " ");

  pdnApiInst->connectIOPinToPowerStripe(points, layer);

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TclPdnConnectStripe::TclPdnConnectStripe(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* point_list = new TclStringOption("-point_list", 0);
  auto* net = new TclStringOption("-net_name", 0);
  auto* layer = new TclStringOption("-layer", 0);
  auto* width_str = new TclIntOption("-width", 0);

  addOption(point_list);
  addOption(net);
  addOption(layer);
  addOption(width_str);
}

unsigned TclPdnConnectStripe::check()
{
  TclOption* point_list = getOptionOrArg("-point_list");
  TclOption* net = getOptionOrArg("-net_name");
  TclOption* layer = getOptionOrArg("-layer");
  // TclOption* width_str = getOptionOrArg("-width");

  LOG_FATAL_IF(!point_list);
  LOG_FATAL_IF(!layer);
  LOG_FATAL_IF(!net);
  //   LOG_FATAL_IF(!width_str);

  return 1;
}

unsigned TclPdnConnectStripe::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* tcl_points = getOptionOrArg("-point_list");
  TclOption* tcl_net = getOptionOrArg("-net_name");
  TclOption* tcl_layer = getOptionOrArg("-layer");
  TclOption* tcl_witdh = getOptionOrArg("-width");

  ieda::Str str = ieda::Str();

  std::vector<double> points = str.splitDouble(tcl_points->getStringVal(), " ");
  std::string net_name = tcl_net->getStringVal();
  std::string layer_name = tcl_layer->getStringVal();
  int32_t width = tcl_witdh->is_set_val() ? tcl_witdh->getIntVal() : -1;

  pdnApiInst->connectPowerStripe(points, net_name, layer_name, width);

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TclPdnAddSegmentStripe::TclPdnAddSegmentStripe(const char* cmd_name) : TclCmd(cmd_name)
{
  TclOption* tcl_net = new TclStringOption("-net_name", 0);
  TclOption* tcl_width = new TclIntOption("-width", 0);

  TclOption* tcl_point_list = new TclStringOption("-point_list", 0);
  TclOption* tcl_layer = new TclStringOption("-layer", 0);
  TclOption* tcl_pt_start = new TclStringOption("-point_begin", 0);
  TclOption* tcl_layer_start = new TclStringOption("-layer_start", 0);
  TclOption* tcl_pt_end = new TclStringOption("-point_end", 0);
  TclOption* tcl_layer_end = new TclStringOption("-layer_end", 0);
  TclOption* tcl_via_width = new TclIntOption("-via_width", 0);
  TclOption* tcl_via_height = new TclIntOption("-via_height", 0);

  addOption(tcl_net);
  addOption(tcl_width);
  addOption(tcl_point_list);
  addOption(tcl_layer);
  addOption(tcl_pt_start);
  addOption(tcl_layer_start);
  addOption(tcl_pt_end);
  addOption(tcl_layer_end);
  addOption(tcl_via_width);
  addOption(tcl_via_height);
}

unsigned TclPdnAddSegmentStripe::check()
{
  TclOption* tcl_net = getOptionOrArg("-net_name");

  TclOption* tcl_point_list = getOptionOrArg("-point_list");
  TclOption* tcl_layer = getOptionOrArg("-layer");
  TclOption* tcl_width = getOptionOrArg("-width");

  TclOption* tcl_pt_start = getOptionOrArg("-point_begin");
  TclOption* tcl_layer_start = getOptionOrArg("-layer_start");
  TclOption* tcl_pt_end = getOptionOrArg("-point_end");
  TclOption* tcl_layer_end = getOptionOrArg("-layer_end");
  TclOption* tcl_via_width = getOptionOrArg("-via_width");
  TclOption* tcl_via_height = getOptionOrArg("-via_height");

  if (tcl_point_list != nullptr) {
    LOG_FATAL_IF(!tcl_point_list);
    LOG_FATAL_IF(!tcl_layer);
    LOG_FATAL_IF(!tcl_net);
    LOG_FATAL_IF(!tcl_width);
  }

  if (tcl_pt_start != nullptr && tcl_pt_end != nullptr) {
    LOG_FATAL_IF(!tcl_net);
    LOG_FATAL_IF(!tcl_via_width);
    LOG_FATAL_IF(!tcl_via_height);
    LOG_FATAL_IF(!tcl_pt_start);
    LOG_FATAL_IF(!tcl_layer_start);
    LOG_FATAL_IF(!tcl_pt_end);
    LOG_FATAL_IF(!tcl_layer_end);
  }

  return 1;
}

unsigned TclPdnAddSegmentStripe::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* tcl_net = getOptionOrArg("-net_name");

  TclOption* tcl_point_list = getOptionOrArg("-point_list");
  TclOption* tcl_layer = getOptionOrArg("-layer");
  TclOption* tcl_width = getOptionOrArg("-width");

  TclOption* tcl_pt_start = getOptionOrArg("-point_begin");
  TclOption* tcl_layer_start = getOptionOrArg("-layer_start");
  TclOption* tcl_pt_end = getOptionOrArg("-point_end");
  TclOption* tcl_layer_end = getOptionOrArg("-layer_end");
  TclOption* tcl_via_width = getOptionOrArg("-via_width");
  TclOption* tcl_via_height = getOptionOrArg("-via_height");

  ieda::Str str = ieda::Str();

  auto point_list_str = tcl_point_list->getStringVal();
  if (point_list_str != nullptr) {
    std::vector<double> points = str.splitDouble(tcl_point_list->getStringVal(), " ");

    std::string net_name = tcl_net->getStringVal();
    std::string layer_name = tcl_layer->getStringVal();
    int32_t width = tcl_width->is_set_val() ? tcl_width->getIntVal() : -1;

    /// add stripe in layer start
    pdnApiInst->addSegmentStripeList(points, net_name, layer_name, width);
  }

  auto pt_start_str = tcl_pt_start->getStringVal();
  auto pt_end_str = tcl_pt_end->getStringVal();
  if (pt_start_str != nullptr && pt_end_str != nullptr) {
    std::vector<double> pt_start = str.splitDouble(tcl_pt_start->getStringVal(), " ");
    std::string layer_start = tcl_layer_start->getStringVal();

    std::vector<double> pt_end = str.splitDouble(tcl_pt_end->getStringVal(), " ");
    std::string layer_end = tcl_layer_end->getStringVal();
    string net_name = tcl_net->getStringVal();

    std::vector<double> points;
    points.insert(points.end(), pt_start.begin(), pt_start.end());
    points.insert(points.end(), pt_end.begin(), pt_end.end());

    int32_t wire_width = tcl_width->getIntVal();
    int32_t via_width = tcl_via_width->getIntVal();
    int32_t via_height = tcl_via_height->getIntVal();

    /// add stripe in layer start
    pdnApiInst->addSegmentStripeList(points, net_name, layer_start, wire_width);

    /// add via betweem layer start and layer end in point end
    pdnApiInst->addSegmentVia(net_name, layer_start, layer_end, pt_end[0], pt_end[1], via_width, via_height);
  }

  return 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TclPdnAddSegmentVia::TclPdnAddSegmentVia(const char* cmd_name) : TclCmd(cmd_name)
{
  TclOption* net_name = new TclStringOption("-net_name", 0);
  TclOption* layer = new TclStringOption("-layer", 0);
  TclOption* top_layer = new TclStringOption("-top_layer", 0);
  TclOption* bottom_layer = new TclStringOption("-bottom_layer", 0);
  TclOption* offset_x = new TclIntOption("-offset_x", 0);
  TclOption* offset_y = new TclIntOption("-offset_y", 0);
  TclOption* width = new TclIntOption("-width", 0);
  TclOption* height = new TclIntOption("-height", 0);

  addOption(net_name);
  addOption(layer);
  addOption(top_layer);
  addOption(bottom_layer);
  addOption(offset_x);
  addOption(offset_y);
  addOption(width);
  addOption(height);
}

unsigned TclPdnAddSegmentVia::check()
{
  TclOption* net_name = getOptionOrArg("-net_name");
  TclOption* layer_name = getOptionOrArg("-layer");
  TclOption* top_layer_name = getOptionOrArg("-top_layer");
  TclOption* bottom_layer_name = getOptionOrArg("-bottom_layer");
  TclOption* offset_x = getOptionOrArg("-offset_x");
  TclOption* offset_y = getOptionOrArg("-offset_y");
  TclOption* width = getOptionOrArg("-width");
  TclOption* height = getOptionOrArg("-height");

  LOG_FATAL_IF(!net_name);
  if (!top_layer_name || !bottom_layer_name) {
    LOG_FATAL_IF(!layer_name);
  }
  if (!layer_name) {
    LOG_FATAL_IF(!top_layer_name);
    LOG_FATAL_IF(!bottom_layer_name);
  }
  LOG_FATAL_IF(!offset_x);
  LOG_FATAL_IF(!offset_y);
  LOG_FATAL_IF(!width);
  LOG_FATAL_IF(!height);

  return 1;
}

unsigned TclPdnAddSegmentVia::exec()
{
  if (!check()) {
    return 0;
  }
  TclOption* tcl_net_name = getOptionOrArg("-net_name");
  TclOption* tcl_layer_name = getOptionOrArg("-layer");
  TclOption* tcl_top_layer_name = getOptionOrArg("-top_layer");
  TclOption* tcl_bottom_layer_name = getOptionOrArg("-bottom_layer");
  TclOption* tcl_offset_x = getOptionOrArg("-offset_x");
  TclOption* tcl_offset_y = getOptionOrArg("-offset_y");
  TclOption* tcl_width = getOptionOrArg("-width");
  TclOption* tcl_height = getOptionOrArg("-height");

  auto net_name = tcl_net_name->getStringVal();
  auto layer_name = tcl_layer_name->getStringVal();
  auto top_layer_name = tcl_top_layer_name->getStringVal();
  auto bottom_layer_name = tcl_bottom_layer_name->getStringVal();
  auto offset_x = tcl_offset_x->getIntVal();
  auto offset_y = tcl_offset_y->getIntVal();
  auto width = tcl_width->getIntVal();
  auto height = tcl_height->getIntVal();

  /// single cut layer
  if (layer_name != nullptr) {
    pdnApiInst->addSegmentVia(net_name, layer_name, offset_x, offset_y, width, height);
    return 1;
  }
  /// multi metal layers

  if (top_layer_name != nullptr && bottom_layer_name != nullptr) {
    pdnApiInst->addSegmentVia(net_name, top_layer_name, bottom_layer_name, offset_x, offset_y, width, height);
    return 1;
  }
  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TclRunPNP::TclRunPNP(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* config_file_name = new TclStringOption("-config_file_name", 0);
  addOption(config_file_name);
}

unsigned TclRunPNP::check()
{
  TclOption* config_file_name = getOptionOrArg("-config_file_name");

  LOG_FATAL_IF(!config_file_name);

  return 1;
}

unsigned TclRunPNP::exec()
{
  if (!check()) {
    return 0;
  }
  TclOption* tcl_config_file_name = getOptionOrArg("-config_file_name");

  auto config_file_name = tcl_config_file_name->getStringVal();

  // pnpApiInst->runiPNP(config_file_name);

  return 1;
}

}  // namespace tcl