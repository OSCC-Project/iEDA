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
#include "pdn_plan.h"

#include "idm.h"
#include "pdn_via.h"

namespace ipdn {
PdnPlan::PdnPlan()
{
  _cut_stripe = new CutStripe();
}

PdnPlan::~PdnPlan()
{
  if (_cut_stripe != nullptr) {
    delete _cut_stripe;
    _cut_stripe = nullptr;
  }
}

int32_t PdnPlan::transUnitDB(double value)
{
  auto idb_design = dmInst->get_idb_design(); //return IdbDesign* _idb_def_service->get_design()
  auto idb_layout = idb_design->get_layout(); //class IdbDefService -> get_layout(), return IdbLayout* _layout

  return idb_layout != nullptr ? idb_layout->transUnitDB(value) : -1; //IdbLayout -> transUnitDB(double value), 即_micron_dbu微米数*value
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PdnPlan::addIOPin(std::string pin_name, std::string net_name, std::string direction, bool is_power)
{
  auto idb_design = dmInst->get_idb_design();
  idb::IdbSpecialNet* power_net = idb_design->get_special_net_list()->find_net(net_name);

  if (power_net == nullptr) {
    std::cout << "[iPDN info]: add net = " << net_name << std::endl;
    power_net = new idb::IdbSpecialNet();
    power_net->set_net_name(net_name);
    idb::IdbConnectType power_type = is_power ? idb::IdbConnectType::kPower : idb::IdbConnectType::kGround;
    power_net->set_connect_type(power_type);
    idb_design->get_special_net_list()->add_net(power_net);
  }
  idb::IdbPin* io_pin = idb_design->get_io_pin_list()->find_pin(pin_name);
  if (io_pin == nullptr) {
    io_pin = new idb::IdbPin();
    idb_design->get_io_pin_list()->add_pin_list(io_pin);
  }
  idb::IdbTerm* io_term = io_pin->get_term();
  if (io_term == nullptr) {
    io_term = new IdbTerm();
  }
  io_pin->set_as_io();
  io_pin->set_special_net(power_net);
  io_pin->set_net_name(net_name);
  io_pin->set_term(io_term);
  io_pin->set_pin_name(pin_name);
  //   idb::IdbConnectDirection direct = transferStringToConnectDirection(direction);
  idb::IdbConnectDirection direct = idb::IdbEnum::GetInstance()->get_connect_property()->get_direction(direction);
  io_term->set_direction(direct);
  io_term->set_type(power_net->get_connect_type());
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PdnPlan::globalConnect(const std::string pdn_net_name, const std::string instance_pdn_pin_name, bool is_power)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_pdn_list = idb_design->get_special_net_list();

  idb::IdbSpecialNet* special_net = idb_pdn_list->find_net(pdn_net_name);
  if (special_net == nullptr) {
    special_net = new idb::IdbSpecialNet();
    idb_pdn_list->add_net(special_net);
    special_net->set_net_name(pdn_net_name);
  }
  special_net->add_pin_string(instance_pdn_pin_name);
  if (is_power) {
    special_net->set_connect_type(idb::IdbConnectType::kPower);
  } else {
    special_net->set_connect_type(idb::IdbConnectType::kGround);
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PdnPlan::placePdnPort(std::string pin_name, std::string io_cell_name, int32_t offset_x, int32_t offset_y, int32_t width,
                           int32_t height, std::string layer_name)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = idb_design->get_layout();

  idb::IdbLayer* layer = idb_layout->get_layers()->find_layer(layer_name);
  idb::IdbPin* io_pin = idb_design->get_io_pin_list()->find_pin(pin_name);
  idb::IdbInstance* io_cell = idb_design->get_instance_list()->find_instance(io_cell_name);
  if (io_pin == nullptr || io_cell == nullptr || layer == nullptr) {
    std::cout << "Error : value input error to placePdnPort." << std::endl;
    return;
  }

  /// calculate rect for port
  idb::IdbRect* rect = io_cell->get_bounding_box();
  int32_t llx = rect->get_low_x();
  int32_t lly = rect->get_low_y();

  int32_t rect_llx = llx + offset_x;
  int32_t rect_lly = lly + offset_y;
  int32_t rect_urx = rect_llx + width;
  int32_t rect_ury = rect_lly + height;

  /// pin coordinate
  int32_t pin_x = (rect_llx + rect_urx) / 2;
  int32_t pin_y = (rect_lly + rect_ury) / 2;

  io_pin->set_average_coordinate(pin_x, pin_y);
  io_pin->set_orient();
  io_pin->set_location(pin_x, pin_y);

  /// set term attribute
  idb::IdbTerm* term = io_pin->get_term();
  if (term == nullptr) {
    std::cout << "Error : can not find IO Term." << std::endl;
    return;
  }
  term->set_special(true);
  term->set_average_position(offset_x / 2, offset_y / 2);
  term->set_placement_status_fix();
  term->set_has_port(term->get_port_list().size() > 0 ? true : false);

  /// set port
  idb::IdbPort* port = term->add_port();
  port->set_coordinate(pin_x, pin_y);
  port->set_placement_status_place();
  port->set_orient(IdbOrient::kN_R0);

  idb::IdbLayerShape* layer_shape = port->add_layer_shape();
  layer_shape->set_layer(layer);
  layer_shape->add_rect(-width / 2, -height / 2, width - width / 2, height - height / 2);

  /// adjust the pin port coordinate
  io_pin->set_bounding_box();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PdnPlan::createGrid(std::string power_net_name, std::string ground_net_name, std::string layer_name, double route_width,
                         double route_offset)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = idb_design->get_layout();
  auto idb_core = idb_layout->get_core();
  auto idb_layer_list = idb_layout->get_layers();
  auto idb_row_list = idb_layout->get_rows();
  auto idb_blockage_list = idb_design->get_blockage_list();

  int32_t x_start = idb_core->get_bounding_box()->get_low_x();
  int32_t y_start = idb_core->get_bounding_box()->get_low_y();
  int32_t x_end = idb_core->get_bounding_box()->get_high_x();
  int32_t y_end = idb_core->get_bounding_box()->get_high_y();

  auto layer = idb_layer_list->find_layer(layer_name);
  /// get the default layer pitch in layer metal 1
  int32_t pitch = (dynamic_cast<IdbLayerRouting*>(layer))->get_pitch_prefer();
  int32_t width = transUnitDB(route_width);
  int32_t offset = transUnitDB(route_offset);

  /// 存储电源线信息
  RouteInfo rt = RouteInfo();
  rt.set_width_pitch_offset(width, pitch, offset);
  _layer_power_route_info_map.insert(make_pair(layer_name, rt));

  /// get special wire
  auto special_wire_power = idb_design->get_special_net_list()->generateWire(power_net_name);
  auto special_wire_ground = idb_design->get_special_net_list()->generateWire(ground_net_name);
  if (special_wire_power == nullptr || special_wire_ground == nullptr) {
    std::cout << "Error : Net name error." << std::endl;
    return;
  }

  /// cover rows
  int32_t index = 0;
  int32_t top_y = INT32_MIN;
  for (auto row : idb_row_list->get_row_list()) {
    int32_t coordinate_y = row->get_original_coordinate()->get_y();

    auto special_wire_segment
        = createSpecialWireSegment(layer, width, IdbWireShapeType::kFollowPin, x_start, coordinate_y, x_end, coordinate_y);

    top_y = std::max(top_y, coordinate_y);
    std::vector<IdbSpecialWireSegment*> blk_result = createSpecialWireSegmentWithInBlockage(special_wire_segment, idb_blockage_list);
    if (index % 2 == 0) {
      for (IdbSpecialWireSegment* seg : blk_result) {
        special_wire_power->add_segment(seg);
      }
    } else {
      for (IdbSpecialWireSegment* seg : blk_result) {
        special_wire_ground->add_segment(seg);
      }
    }
    ++index;
  }

  /// cover the top edge
  top_y += idb_row_list->get_row_height();
  auto segment_top = createSpecialWireSegment(layer, width, IdbWireShapeType::kFollowPin, x_start, top_y, x_end, top_y);
  auto blk_result_top = createSpecialWireSegmentWithInBlockage(segment_top, idb_blockage_list);
  if (index % 2 == 0) {
    for (auto seg : blk_result_top) {
      special_wire_power->add_segment(seg);
    }

  } else {
    for (auto seg : blk_result_top) {
      special_wire_ground->add_segment(seg);
    }
  }

  idb_core->get_bounding_box()->set_low_y(y_start - width / 2);
  idb_core->get_bounding_box()->set_high_y(y_end + width / 2);
  std::cout << "Core ll coordinate change to:" << idb_core->get_bounding_box()->get_low_x() << " , "
            << idb_core->get_bounding_box()->get_low_y() << std::endl;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
idb::IdbSpecialWireSegment* PdnPlan::createSpecialWireSegment(idb::IdbLayer* layer, int32_t route_width,
                                                              idb::IdbWireShapeType wire_shape_type, int32_t x_start, int32_t y_start,
                                                              int32_t x_end, int32_t y_end)
{
  idb::IdbSpecialWireSegment* special_wire_segment = new idb::IdbSpecialWireSegment();
  special_wire_segment->set_layer_as_new();
  special_wire_segment->set_layer(layer);
  special_wire_segment->set_route_width(route_width);
  special_wire_segment->set_shape_type(wire_shape_type);
  special_wire_segment->add_point(x_start, y_start);
  special_wire_segment->add_point(x_end, y_end);

  special_wire_segment->set_bounding_box();
  return special_wire_segment;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<idb::IdbSpecialWireSegment*> PdnPlan::createSpecialWireSegmentWithInBlockage(idb::IdbSpecialWireSegment* wire_segment,
                                                                                         idb::IdbBlockageList* blockage_list)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = idb_design->get_layout();
  auto idb_core = idb_layout->get_core();

  std::vector<idb::IdbSpecialWireSegment*> result;
  std::vector<idb::IdbBlockage*> layer_blockage;
  idb::IdbLayer* wire_layer = wire_segment->get_layer();
  idb::IdbRect* segment_shape = wire_segment->get_bounding_box();
  int32_t seg_llx = segment_shape->get_low_x();
  int32_t seg_lly = segment_shape->get_low_y();
  int32_t seg_urx = segment_shape->get_high_x();
  int32_t seg_ury = segment_shape->get_high_y();
  int32_t route_width = wire_segment->get_route_width();
  idb::IdbWireShapeType wire_type = wire_segment->get_shape_type();

  // IdbRect core = _floorplandb->get_core()->get_bounding_box();
  int32_t core_llx = idb_core->get_bounding_box()->get_low_x();
  int32_t core_lly = idb_core->get_bounding_box()->get_low_y();
  int32_t core_urx = idb_core->get_bounding_box()->get_high_x();
  int32_t core_ury = idb_core->get_bounding_box()->get_high_y();

  int32_t blk_llx;
  int32_t blk_lly;
  int32_t blk_urx;
  int32_t blk_ury;

  /// obtain all routing blockage on this layer
  for (auto blk : blockage_list->get_blockage_list()) {
    if (blk->get_type() == idb::IdbBlockage::IdbBlockageType::kRoutingBlockage) {
      if (dynamic_cast<idb::IdbRoutingBlockage*>(blk)->get_layer()->get_name() == wire_layer->get_name()) {
        layer_blockage.push_back(blk);
      }
    }
  }

  std::vector<int32_t> overlap_blockage_edge_temp;
  std::vector<int32_t> overlap_blockage_edge;

  for (auto rt_blk : layer_blockage) {
    std::vector<idb::IdbRect*> rect_list = rt_blk->get_rect_list();
    // Now, by default, a blockage has only one rect
    idb::IdbRect* blk_shape = rt_blk->get_rect_list()[0];
    blk_llx = blk_shape->get_low_x();
    blk_lly = blk_shape->get_low_y();
    blk_urx = blk_shape->get_high_x();
    blk_ury = blk_shape->get_high_y();
    if (wire_segment->is_horizontal()) {
      if ((seg_lly < blk_ury && seg_lly > blk_lly) || (seg_ury < blk_ury && seg_ury > blk_lly)) {
        overlap_blockage_edge_temp.push_back(blk_llx);
        overlap_blockage_edge_temp.push_back(blk_urx);
      }
    } else if (wire_segment->is_vertical()) {
      if ((seg_llx < blk_urx && seg_llx > blk_llx) || (seg_urx < blk_urx && seg_urx > blk_llx)) {
        overlap_blockage_edge_temp.push_back(blk_lly);
        overlap_blockage_edge_temp.push_back(blk_ury);
      }
    }
  }

  if (overlap_blockage_edge_temp.size() <= 0) {
    result.push_back(wire_segment);
    return result;
  } else if (wire_segment->is_horizontal()) {
    overlap_blockage_edge_temp.push_back(seg_llx);
    overlap_blockage_edge_temp.push_back(seg_urx);
  } else if (wire_segment->is_vertical()) {
    overlap_blockage_edge_temp.push_back(seg_lly);
    overlap_blockage_edge_temp.push_back(seg_ury);
  }

  /// sort overlap blockage
  sort(overlap_blockage_edge_temp.begin(), overlap_blockage_edge_temp.end(), [](int32_t a, int32_t b) { return a < b; });
  for (size_t i = 0; i < overlap_blockage_edge_temp.size(); ++i) {
    if (wire_segment->is_horizontal()) {
      if (overlap_blockage_edge_temp[i] < core_llx) {
        ++i;
        continue;
      } else if (overlap_blockage_edge_temp[i] > core_urx) {
        overlap_blockage_edge.pop_back();
        break;
      } else {
        overlap_blockage_edge.push_back(overlap_blockage_edge_temp[i]);
      }
    } else if (wire_segment->is_vertical()) {
      if (overlap_blockage_edge_temp[i] < core_lly) {
        ++i;
        continue;
      } else if (overlap_blockage_edge_temp[i] > core_ury) {
        overlap_blockage_edge.pop_back();
        break;
      } else {
        overlap_blockage_edge.push_back(overlap_blockage_edge_temp[i]);
      }
    }
  }

  int real_size = overlap_blockage_edge.size();
  if (real_size < 2) {
    result.push_back(wire_segment);
    return result;
  }

  // cut old wire special wire segment
  for (int i = 0; i < real_size - 1; i += 2) {
    if (overlap_blockage_edge[i] == overlap_blockage_edge[i + 1]) {
      continue;
    }
    if (wire_segment->is_horizontal()) {
      if (overlap_blockage_edge[i] <= core_llx && overlap_blockage_edge[i + 1] <= core_llx) {
        continue;
      } else {
        auto new_seg = createSpecialWireSegment(wire_layer, route_width, wire_type, overlap_blockage_edge[i], seg_lly + route_width / 2,
                                                overlap_blockage_edge[i + 1], seg_lly + route_width / 2);
        result.push_back(new_seg);
      }
    } else if (wire_segment->is_vertical()) {
      if (overlap_blockage_edge[i] >= core_ury && overlap_blockage_edge[i + 1] >= core_ury) {
        continue;
      } else {
        auto new_seg = createSpecialWireSegment(wire_layer, route_width, wire_type, seg_llx + route_width / 2, overlap_blockage_edge[i],
                                                seg_llx + route_width / 2, overlap_blockage_edge[i + 1]);
        result.push_back(new_seg);
      }
    }
  }

  return result;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
void PdnPlan::createStripe(std::string power_net_name, std::string ground_net_name, std::string layer_name, double route_width,
                           double pitchh, double route_offset)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = idb_design->get_layout();
  auto idb_core = idb_layout->get_core();
  auto idb_layer_list = idb_layout->get_layers();
  // auto idb_row_list = idb_layout->get_rows();
  // auto idb_blockage_list = idb_design->get_blockage_list();
  auto idb_pdn_list = idb_design->get_special_net_list();

  idb::IdbLayer* layer = idb_layer_list->find_layer(layer_name);
  if (layer == nullptr) {
    std::cout << "Error : can not find layer." << std::endl;
  }

  /// hardcode
  // string  routing_layer_name = routing_layers[0]->get_name();
  /// 110nm
  std::string routing_layer_name = "METAL1";
  /// sky 130
  // string  routing_layer_name = "met1";
  auto core_boundingbox = idb_core->get_bounding_box();
  int32_t rail_half_route_width = _layer_power_route_info_map[routing_layer_name].get_route_width() / 2;
  core_boundingbox->set_low_y(core_boundingbox->get_low_y() - rail_half_route_width);
  core_boundingbox->set_high_y(core_boundingbox->get_high_y() + rail_half_route_width);

  int32_t x_start = core_boundingbox->get_low_x();
  int32_t y_start = core_boundingbox->get_low_y();
  int32_t x_end = core_boundingbox->get_high_x();
  int32_t y_end = core_boundingbox->get_high_y();
  int32_t pitch = transUnitDB(pitchh);
  int32_t half_pitch = pitch / 2;
  int32_t width = transUnitDB(route_width);
  int32_t offset = transUnitDB(route_offset);

  /// get power route info
  RouteInfo rt = RouteInfo();
  rt.set_width_pitch_offset(width, pitch, offset);
  _layer_power_route_info_map.insert(make_pair(layer_name, rt));

  /// 找到该绕线层的track信息，电源线需要根据track生成
  int32_t track_offset = (dynamic_cast<idb::IdbLayerRouting*>(layer))->get_offset_prefer();
  int32_t track_pitch = (dynamic_cast<idb::IdbLayerRouting*>(layer))->get_pitch_prefer();

  /// find special wire
  auto special_wire_power = idb_pdn_list->generateWire(power_net_name);
  auto special_wire_ground = idb_pdn_list->generateWire(ground_net_name);

  ////
  bool isHorizontal = dynamic_cast<idb::IdbLayerRouting*>(layer)->is_horizontal();

  idb::IdbSpecialWireSegment* special_wire_segment = nullptr;
  int32_t start = (isHorizontal ? y_start : x_start) + offset + width / 2;
  int32_t end = isHorizontal ? y_end : x_end;

  for (int i = start; i <= end; i += pitch) {
    if (width <= track_pitch) {
      i = (i - track_offset) / track_pitch * track_pitch + track_offset;
    }

    if (isHorizontal) {
      special_wire_segment = createSpecialWireSegment(layer, width, idb::IdbWireShapeType::kStripe, x_start, i, x_end, i);
    } else {
      special_wire_segment = createSpecialWireSegment(layer, width, idb::IdbWireShapeType::kStripe, i, y_start, i, y_end);
    }
    addSpecialWireSegmentWithInBlockage(special_wire_power, special_wire_segment);

    if ((i + half_pitch + width / 2) <= end) {
      i += half_pitch;
      if (isHorizontal) {
        special_wire_segment = createSpecialWireSegment(layer, width, IdbWireShapeType::kStripe, x_start, i, x_end, i);
      } else {
        special_wire_segment = createSpecialWireSegment(layer, width, IdbWireShapeType::kStripe, i, y_start, i, y_end);
      }
      addSpecialWireSegmentWithInBlockage(special_wire_ground, special_wire_segment);
      // special_wire_ground->add_segment(special_wire_segment);
      i -= half_pitch;
    }
  }
}

/**
 * @brief Enables the creation of power line segments with wiring obstacles
 *
 * @param sp_wire
 * @param sp_wire_segment
 */
void PdnPlan::addSpecialWireSegmentWithInBlockage(idb::IdbSpecialWire* sp_wire, idb::IdbSpecialWireSegment* sp_wire_segment)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_blockage_list = idb_design->get_blockage_list();

  std::vector<idb::IdbSpecialWireSegment*> blk_result = createSpecialWireSegmentWithInBlockage(sp_wire_segment, idb_blockage_list);
  for (auto seg : blk_result) {
    sp_wire->add_segment(seg);
  }
}
/**
 * @brief  connect special net between the specified two routing layers by using
 * vias
 * @param  layer_name_first
 * @param  layer_name_second
 */
void PdnPlan::connectLayer(std::string layer_name_first, std::string layer_name_second)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = dmInst->get_idb_layout();
  auto idb_layer_list = idb_layout->get_layers();
  auto idb_pdn_list = idb_design->get_special_net_list();

  /// find top cut bottom layer
  IdbLayerRouting* layer_bottom = dynamic_cast<IdbLayerRouting*>(idb_layer_list->find_layer(layer_name_first));
  IdbLayerRouting* layer_top = dynamic_cast<IdbLayerRouting*>(idb_layer_list->find_layer(layer_name_second));

  if (layer_bottom == nullptr || layer_top == nullptr || layer_bottom == layer_top) {
    std::cout << "Error : layers not exist." << layer_top->get_name() << " & " << layer_bottom->get_name() << std::endl;
    return;
  }
  /// ensure bottom is smaller than top
  if (layer_top->get_order() < layer_bottom->get_order()) {
    std::swap(layer_top, layer_bottom);
  }

  /// do not sopport 2 layer with the same direction
  if ((layer_top->is_horizontal() && layer_bottom->is_horizontal()) || (layer_top->is_vertical() && layer_bottom->is_vertical())) {
    std::cout << "Error : layers have the same direction." << layer_top->get_name() << " & " << layer_bottom->get_name() << std::endl;
    return;
  }

  /// connect each net
  for (IdbSpecialNet* net : idb_pdn_list->get_net_list()) {
    /// get the routing width and height to find via
    /// attention : use the given layer top and layer bottom width
    // int32_t width_top = net->get_layer_width(layer_top->get_name());
    // int32_t width_bottom = net->get_layer_width(layer_bottom->get_name());
    // int width = layer_top->is_vertical() ? width_top : width_bottom;
    // int height = layer_bottom->is_horizontal() ? width_bottom : width_top;

    /// find via list that fit the 2 layers for each net
    std::vector<IdbVia*> via_find_list;

    connectTwoLayerForNet(net, via_find_list, layer_top, layer_bottom);
    std::cout << "Success : ConnectTwoLayerForNet " << net->get_net_name() << std::endl;
  }

  std::cout << "Success : connectTwoLayer " << layer_name_first << " & " << layer_name_second << std::endl;
}

void PdnPlan::updateRouteMap()
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_pdn_list = idb_design->get_special_net_list();

  string layer_name;

  for (auto special_net : idb_pdn_list->get_net_list()) {
    if (special_net->get_net_name() == "VDD") {
      for (auto special_wire : special_net->get_wire_list()->get_wire_list()) {
        for (auto wire_segment : special_wire->get_segment_list()) {
          layer_name = wire_segment->get_layer()->get_name();
          _layer_power_route_info_map[layer_name].add_special_wire_segment("VDD", wire_segment);
        }
      }
    } else if (special_net->get_net_name() == "VSS") {
      for (auto special_wire : special_net->get_wire_list()->get_wire_list()) {
        for (auto wire_segment : special_wire->get_segment_list()) {
          layer_name = wire_segment->get_layer()->get_name();
          _layer_power_route_info_map[layer_name].add_special_wire_segment("VSS", wire_segment);
        }
      }
    }
  }
}

/**
 * @brief Connect power lines on different layers of the same specialnet
 *
 * @param net
 * @param via_list
 * @param layer_top
 * @param layer_bottom
 */
void PdnPlan::connectTwoLayerForNet(idb::IdbSpecialNet* net, std::vector<idb::IdbVia*>& via_list, idb::IdbLayerRouting* layer_top,
                                    idb::IdbLayerRouting* layer_bottom)
{
  /// find wire list which this net has
  idb::IdbSpecialWireList* wire_list = net->get_wire_list();
  if (wire_list == nullptr) {
    std::cout << "Error : not wire in Special net " << net->get_net_name() << std::endl;
    return;
  }
  /// find all wire segment belong to layer_top and layer_bottom
  for (idb::IdbSpecialWire* wire : wire_list->get_wire_list()) {
    std::vector<idb::IdbSpecialWireSegment*> segment_list_top;
    std::vector<idb::IdbSpecialWireSegment*> segment_list_bottom;
    for (idb::IdbSpecialWireSegment* segment : wire->get_segment_list()) {
      if (segment->is_tripe() || segment->is_follow_pin()) {
        if (segment->get_layer()->compareLayer(layer_top)) {
          segment_list_top.emplace_back(segment);
        }

        if (segment->get_layer()->compareLayer(layer_bottom)) {
          segment_list_bottom.emplace_back(segment);
        }
      }
    }

    std::cout << net->get_net_name() << " Finish construct segment list" << std::endl;

    connectTwoLayerForWire(wire, via_list, segment_list_top, segment_list_bottom);
  }
}

/**
 * @brief Connect power line segments of the same wire on different layers
 *
 * @param wire
 * @param via_list
 * @param segment_list_top
 * @param segment_list_bottom
 */
void PdnPlan::connectTwoLayerForWire(idb::IdbSpecialWire* wire, std::vector<idb::IdbVia*>& via_list,
                                     std::vector<idb::IdbSpecialWireSegment*>& segment_list_top,
                                     std::vector<idb::IdbSpecialWireSegment*>& segment_list_bottom)
{
  if (segment_list_top.size() <= 0 || segment_list_bottom.size() <= 0) {
    return;
  }

  auto idb_layout = dmInst->get_idb_layout();
  auto idb_layer_list = idb_layout->get_layers();
  PdnVia pdn_via;
  int number = 0;

  idb::IdbLayerRouting* layer_bottom_routing
      = dynamic_cast<idb::IdbLayerRouting*>(idb_layer_list->find_layer(segment_list_bottom[0]->get_layer()->get_name()));
  idb::IdbLayerRouting* layer_top_routing
      = dynamic_cast<idb::IdbLayerRouting*>(idb_layer_list->find_layer(segment_list_top[0]->get_layer()->get_name()));
  for (idb::IdbSpecialWireSegment* segment_top : segment_list_top) {
    for (idb::IdbSpecialWireSegment* segment_bottom : segment_list_bottom) {
      /// calculate intersection between layer stripe
      idb::IdbRect coordinate;
      // if (segment_top->get_intersect_coordinate(segment_bottom, coordinate))
      // {
      if (_cut_stripe->get_intersect_coordinate(segment_top, segment_bottom, coordinate)) {
        // for (IdbVia *via : via_list) {
        // IdbLayer *             layer_top =
        // via->get_top_layer_shape().get_layer();
        for (int32_t layer_order = layer_bottom_routing->get_order(); layer_order <= (layer_top_routing->get_order() - 2);) {
          // int32_t layer_order_bottom = layer_bottom_routing->get_order();
          idb::IdbLayerCut* layer_cut_find = dynamic_cast<idb::IdbLayerCut*>(idb_layer_list->find_layer_by_order(layer_order + 1));
          if (layer_cut_find == nullptr) {
            std::cout << "Error : layer input illegal." << std::endl;
            return;
          }
          //   IdbVia *via_find = via_listt->find_via_generate(
          //       layer_cut_find, coordinate.get_width(),
          //       coordinate.get_height());
          idb::IdbVia* via_find = pdn_via.findVia(layer_cut_find, coordinate.get_width(), coordinate.get_height());
          if (via_find == nullptr) {
            std::cout << "Error : can not find VIA matchs." << std::endl;
            continue;
          }
          idb::IdbLayer* layer_top = via_find->get_top_layer_shape().get_layer();
          idb::IdbCoordinate<int32_t> middle = coordinate.get_middle_point();
          idb::IdbSpecialWireSegment* segment_via
              = pdn_via.createSpecialWireVia(layer_top, 0, idb::IdbWireShapeType::kStripe, &middle, via_find);
          wire->add_segment(segment_via);
          number++;

          if (number % 10000 == 0) {
            std::cout << "-";
          }
          layer_order += 2;
          // }
        }
      }
    }
  }
  std::cout << std::endl << "Finish add_segment number = " << number << std::endl;
}

/**
 * @brief  Connect io pin to the power network in core
 * @param  point_list
 * @param  layer_name
 */
void PdnPlan::connectIOPinToPowerStripe(std::vector<double>& point_list, const std::string layer_name)
{
  std::vector<idb::IdbCoordinate<int32_t>*> result;
  for (size_t i = 0; i < point_list.size() - 1; i += 2) {
    idb::IdbCoordinate<int32_t>* p = new idb::IdbCoordinate<int32_t>();
    p->set_xy(transUnitDB(point_list[i]), transUnitDB(point_list[i + 1]));
    result.push_back(p);
  }

  auto idb_layout = dmInst->get_idb_layout();
  auto idb_layer_list = idb_layout->get_layers();
  auto layer = idb_layer_list->find_layer(layer_name);

  _cut_stripe->initEdge();
  _cut_stripe->connectIOPinToPowerStripe(result, layer);

  /// release
  for (idb::IdbCoordinate<int32_t>* coordinate : result) {
    if (coordinate != nullptr) {
      delete coordinate;
      coordinate = nullptr;
    }
  }

  std::vector<idb::IdbCoordinate<int32_t>*>().swap(result);
}

/**
 * @brief
 *
 * @param point_list
 * @param net_name
 * @param layer_name
 * @param width
 */
void PdnPlan::connectPowerStripe(std::vector<double>& point_list, const std::string& net_name, const std::string& layer_name, int32_t width)
{
  std::vector<idb::IdbCoordinate<int32_t>*> result;
  for (size_t i = 0; i < point_list.size() - 1; i += 2) {
    idb::IdbCoordinate<int32_t>* p = new idb::IdbCoordinate<int32_t>();
    p->set_xy(transUnitDB(point_list[i]), transUnitDB(point_list[i + 1]));
    result.push_back(p);
  }

  _cut_stripe->connectPowerStripe(result, net_name, layer_name, width);

  /// release
  for (IdbCoordinate<int32_t>* coordinate : result) {
    if (coordinate != nullptr) {
      delete coordinate;
      coordinate = nullptr;
    }
  }

  vector<IdbCoordinate<int32_t>*>().swap(result);
}

bool PdnPlan::addSegmentStripeList(std::vector<double>& point_list, std::string net_name, std::string layer_name, int32_t width)
{
  std::vector<idb::IdbCoordinate<int32_t>*> result;
  for (size_t i = 0; i < point_list.size() - 1; i += 2) {
    idb::IdbCoordinate<int32_t>* p = new idb::IdbCoordinate<int32_t>();
    p->set_xy(transUnitDB(point_list[i]), transUnitDB(point_list[i + 1]));
    result.push_back(p);
  }

  bool b_result = addSegmentStripeList(result, net_name, layer_name, width);

  /// release
  for (IdbCoordinate<int32_t>* coordinate : result) {
    if (coordinate != nullptr) {
      delete coordinate;
      coordinate = nullptr;
    }
  }

  vector<IdbCoordinate<int32_t>*>().swap(result);

  return b_result;
}

bool PdnPlan::addSegmentStripeList(std::vector<idb::IdbCoordinate<int32_t>*> point_list, std::string net_name, std::string layer_name,
                                   int32_t width)
{
  if (point_list.size() >= 2) {
    for (size_t i = 0; i < point_list.size() - 1; i++) {
      addSegmentStripe(point_list[i]->get_x(), point_list[i]->get_y(), point_list[i + 1]->get_x(), point_list[i + 1]->get_y(), net_name,
                       layer_name, width);
    }

    return true;
  }

  return false;
}

bool PdnPlan::addSegmentStripe(int32_t x_start, int32_t y_start, int32_t x_end, int32_t y_end, std::string net_name, std::string layer_name,
                               int32_t width)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = dmInst->get_idb_layout();
  auto idb_layer_list = idb_layout->get_layers();
  auto idb_pdn_list = idb_design->get_special_net_list();

  /// find all the cut layer betweem top_metal and bottom_metal
  IdbLayer* routing_layer = idb_layer_list->find_layer(layer_name);
  if (routing_layer == nullptr) {
    return false;
  }

  IdbSpecialNet* net = idb_pdn_list->find_net(net_name);
  if (net == nullptr) {
    std::cout << "Error : can't find the net. " << std::endl;
    return false;
  }

  IdbSpecialWire* wire = net->get_wire_list()->get_num() > 0 ? net->get_wire_list()->find_wire(0) : net->get_wire_list()->add_wire(nullptr);
  if (wire == nullptr) {
    std::cout << "Error : can't get the wire." << std::endl;
    return false;
  }

  IdbSpecialWireSegment* segment
      = createSpecialWireSegment(routing_layer, width, IdbWireShapeType::kStripe, x_start, y_start, x_end, y_end);
  wire->add_segment(segment);

  return segment != nullptr ? true : false;
}

}  // namespace ipdn