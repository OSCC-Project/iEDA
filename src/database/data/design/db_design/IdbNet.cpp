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
 * @project		iDB
 * @file		IdbNet.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Defines netlist connectivity and regular-routes for nets containing regular pins
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbNet.h"

#include <algorithm>

#include "IdbInstance.h"

namespace idb {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IdbNet::IdbNet()
{
  _net_name = "";
  _xtalk = 0;
  _connect_type = IdbConnectType::kNone;
  _source_type = IdbInstanceType::kNone;
  _weight = 0;

  _fix_bump = false;
  _frequency = -1;
  _average_coordinate = nullptr;

  _io_pin_list = new IdbPins();
  _instance_pin_list = new IdbPins();
  _instance_list = new IdbInstanceList();

  _wire_list = new IdbRegularWireList();
}

IdbNet::~IdbNet()
{
  if (_io_pin_list != nullptr) {
    delete _io_pin_list;
    _io_pin_list = nullptr;
  }

  if (_instance_pin_list != nullptr) {
    delete _instance_pin_list;
    _instance_pin_list = nullptr;
  }

  _instance_list->reset(false);
  if (_instance_list != nullptr) {
    delete _instance_list;
    _instance_list = nullptr;
  }

  if (_wire_list != nullptr) {
    delete _wire_list;
    _wire_list = nullptr;
  }

  if (_average_coordinate != nullptr) {
    delete _average_coordinate;
    _average_coordinate = nullptr;
  }
}

void IdbNet::clear_wire_list()
{
  _wire_list->clear();
}

void IdbNet::set_connect_type(string type)
{
  set_connect_type(IdbEnum::GetInstance()->get_connect_property()->get_type(type));
}

void IdbNet::set_source_type(string type)
{
  _source_type = IdbEnum::GetInstance()->get_instance_property()->get_type(type);
}

void IdbNet::set_average_coordinate(IdbCoordinate<int32_t>* average_coordinate)
{
  if (_average_coordinate == average_coordinate) {
    return;
  }
  if (_average_coordinate != nullptr) {
    delete _average_coordinate;
  }
  _average_coordinate = average_coordinate;
}

// If pin's direction is OUTPUT, the pin is a driven pin
// If pin's direction is INOUT(Generally IO Pin), the driven pin depends on the first connected pin's direction
// case 1 : pin1[OUTPUT], Pin2[INPUT], pin3[INPUT]... : driven pin = pin1

// to be discuss
// case 2 : pin1[INOUT], pin2[OUT]                    : driven pin = pin2 ?? Not sure if the situation is exist???
// case 3 : pin1[INOUT], {pin2[INOUT]<--Instance-->pin3[OUTPUT]}, pin4[INPUT]...  : driven pin = pin1
//          ------------>---------------------------------------->--------------
// case 4 : pin1[INOUT], {pin2[INOUT]<--Instance-->pin3[INPUT]}, pin4[OUTPUT]...  : driven pin = pin2
//          ------------<----------------------------------------<--------------
//          both pin1 & pin2 are INOUT direction, the direction depends on direction of pin3 which is in the same
//          instance
// IdbPin* IdbNet::findDrivingPin()
IdbPin* IdbNet::get_driving_pin()
{
  /// 1st step : check if exist instance output pin, if existed, it is driving pin
  for (IdbPin* pin : _instance_pin_list->get_pin_list()) {
    if (pin->get_term()->get_direction() == IdbConnectDirection::kOutput
        || pin->get_term()->get_direction() == IdbConnectDirection::kOutputTriState) {
      return pin;
    }
  }

  /// 2nd step : if instance output pin not exist, find io pin with input direction
  for (IdbPin* io_pin : _io_pin_list->get_pin_list()) {
    // case 2
    if (io_pin->get_term()->get_direction() == IdbConnectDirection::kInput) {
      return io_pin;
    }
  }

  /// 3nd step, if io input pin not exist, find io pin with inout direction
  for (IdbPin* io_pin : _io_pin_list->get_pin_list()) {
    // case 2
    if (io_pin->get_term()->get_direction() == IdbConnectDirection::kInOut) {
      return io_pin;
    }
  }
  //     // case 3 or case 4
  // IdbInstance* instance      = instance_pin->get_instance();
  // IdbPins* instance_pin_list = instance->get_pin_list();
  // for (IdbPin* pin : instance_pin_list->get_pin_list()) {
  //   if (pin->get_pin_name() != instance_pin->get_pin_name()) {
  //     return pin->get_term()->get_direction() == IdbConnectDirection::kOutput ? _io_pin : instance_pin;
  //   }
  //   }

  // 临时修复，可能会有问题，zzs
  if (!_io_pin_list->get_pin_list().empty()) {
    return _io_pin_list->get_pin_list().front();
  }
  if (!_instance_pin_list->get_pin_list().empty()) {
    return _instance_pin_list->get_pin_list().front();
  }

  std::cout << "Error : No driver pin exist..." << std::endl;
  return nullptr;
}

vector<IdbPin*> IdbNet::get_load_pins()
{
  vector<IdbPin*> pin_list;

  auto* driver_pin = get_driving_pin();

  std::copy(_io_pin_list->get_pin_list().begin(), _io_pin_list->get_pin_list().end(), std::back_inserter(pin_list));
  std::copy(_instance_pin_list->get_pin_list().begin(), _instance_pin_list->get_pin_list().end(), std::back_inserter(pin_list));

  pin_list.erase(std::remove(pin_list.begin(), pin_list.end(), driver_pin), pin_list.end());

  return pin_list;
}

IdbRect* IdbNet::get_bounding_box()
{
  int32_t min_lx = INT32_MAX;
  int32_t min_ly = INT32_MAX;
  int32_t max_ux = INT32_MIN;
  int32_t max_uy = INT32_MIN;
  auto* idb_driving_pin = get_driving_pin();
  if (idb_driving_pin != nullptr) {
    min_lx = idb_driving_pin->get_average_coordinate()->get_x();
    min_ly = idb_driving_pin->get_average_coordinate()->get_y();
    max_ux = idb_driving_pin->get_average_coordinate()->get_x();
    max_uy = idb_driving_pin->get_average_coordinate()->get_y();
  }
  for (auto* idb_load_pin : get_load_pins()) {
    min_lx = std::min(min_lx, idb_load_pin->get_average_coordinate()->get_x());
    min_ly = std::min(min_ly, idb_load_pin->get_average_coordinate()->get_y());
    max_ux = std::max(max_ux, idb_load_pin->get_average_coordinate()->get_x());
    max_uy = std::max(max_uy, idb_load_pin->get_average_coordinate()->get_y());
  }
  return new IdbRect(min_lx, min_ly, max_ux, max_uy);
}

bool IdbNet::set_bounding_box()
{
  // IdbRect* rect = get_bounding_box();

  // int32_t ll_x = _average_coordinate->get_x() - _io_term->get_bounding_box()->get_width()/2;
  // int32_t ll_y = _average_coordinate->get_y() - _io_term->get_bounding_box()->get_height()/2;
  // int32_t ur_x = _average_coordinate->get_x() + _io_term->get_bounding_box()->get_width()/2;
  // int32_t ur_y = _average_coordinate->get_y() + _io_term->get_bounding_box()->get_height()/2;
  //  rect->set_rect(ll_x, ll_y, ur_x, ur_y);
  return false;
}

bool IdbNet::has_io_pin(IdbPin* io_pin)
{
  return _io_pin_list != nullptr && _io_pin_list->contains(io_pin);
}

bool IdbNet::has_instance_pin(IdbPin* inst_pin)
{
  return _instance_pin_list != nullptr && _instance_pin_list->contains(inst_pin);
}

bool IdbNet::has_instance(IdbInstance* instance)
{
  return _instance_list != nullptr && _instance_list->contains(instance);
}

IdbPin* IdbNet::add_io_pin_unique(IdbPin* io_pin)
{
  if (_io_pin_list == nullptr || io_pin == nullptr) {
    return nullptr;
  }

  return _io_pin_list->add_pin_ref_unique(io_pin);
}

IdbPin* IdbNet::add_instance_pin_unique(IdbPin* inst_pin)
{
  if (_instance_pin_list == nullptr || inst_pin == nullptr) {
    return nullptr;
  }

  return _instance_pin_list->add_pin_ref_unique(inst_pin);
}

bool IdbNet::erase_pin_ref(IdbPin* pin)
{
  bool erased = false;
  if (_io_pin_list != nullptr) {
    erased |= _io_pin_list->erase_pin_ref(pin);
  }
  if (_instance_pin_list != nullptr) {
    erased |= _instance_pin_list->erase_pin_ref(pin);
  }

  return erased;
}

bool IdbNet::erase_instance_ref(IdbInstance* instance)
{
  if (_instance_list == nullptr || instance == nullptr) {
    return false;
  }

  return _instance_list->erase_instance_ref(instance);
}

void IdbNet::remove_pin(IdbPin* pin)
{
  erase_pin_ref(pin);
  if (pin != nullptr && pin->get_net() == this) {
    pin->remove_net();
  }
}

void IdbNet::remove_segment(IdbRegularWireSegment* seg_del)
{
  for (auto* wire : _wire_list->get_wire_list()) {
    if (wire->delete_seg(seg_del)) {
      return;
    }
  }
}

bool IdbNet::checkConnection()
{
  bool b_result = false;

  // /// build pin
  // for (IdbPin* pin : _instance_pin_list->get_pin_list()) {
  // }

  // /// build wire
  // for (IdbRegularWire* wire : _wire_list->get_wire_list()) {
  //   for (IdbRegularWireSegment* segment : wire->get_segment_list()) {
  //   }
  // }

  if (!b_result) {
    std::cout << "[IdbNetList Error] Net connected failed. Net name = " << get_net_name() << std::endl;
  }

  return b_result;
}

uint64_t IdbNet::wireLength()
{
  return _wire_list->wireLength();
}

uint64_t IdbNet::get_via_number()
{
  uint64_t number = 0;
  for (auto wire : _wire_list->get_wire_list()) {
    for (auto segment : wire->get_segment_list()) {
      number += segment->get_via_list().size();
    }
  }

  return number;
}

void IdbNet::mergeWireSegments()
{
  if (_wire_list == nullptr) {
    return;
  }

  auto split_segment = [](IdbRegularWireSegment* segment, bool keep_via) -> IdbRegularWireSegment* {
    if (segment == nullptr) {
      return nullptr;
    }

    if (keep_via) {
      segment->clearPoints();
      return nullptr;
    }

    IdbRegularWireSegment* via_seg = new IdbRegularWireSegment();
    via_seg->set_layer_status(true);
    via_seg->set_layer_name(segment->get_layer_name());
    via_seg->set_layer(segment->get_layer());
    via_seg->set_via_list(segment->take_via_list());
    via_seg->set_is_via(true);
    segment->set_is_via(false);
    return via_seg;
  };

  auto merge = [&](std::map<int, std::vector<IdbRegularWireSegment*>>& segment_map,
                   bool b_horizontal) -> std::pair<std::vector<IdbRegularWireSegment*>, std::vector<IdbRegularWireSegment*>> {
    std::vector<IdbRegularWireSegment*> new_segments;
    std::vector<IdbRegularWireSegment*> delete_segments;

    for (auto& [coordinate, segments] : segment_map) {
      if (segments.size() < 2) {
        continue;
      }

      std::sort(segments.begin(), segments.end(), [&](IdbRegularWireSegment* a, IdbRegularWireSegment* b) {
        return b_horizontal ? a->get_point_start()->get_x() < b->get_point_start()->get_x()
                            : a->get_point_start()->get_y() < b->get_point_start()->get_y();
      });

      auto it_1 = segments.begin();
      auto it_2 = segments.begin() + 1;
      for (; it_2 != segments.end(); it_2++) {
        auto coord_1_begin = b_horizontal ? (*it_1)->get_point_start()->get_x() : (*it_1)->get_point_start()->get_y();
        auto coord_1_end = b_horizontal ? (*it_1)->get_point_second()->get_x() : (*it_1)->get_point_second()->get_y();
        auto coord_2_begin = b_horizontal ? (*it_2)->get_point_start()->get_x() : (*it_2)->get_point_start()->get_y();
        auto coord_2_end = b_horizontal ? (*it_2)->get_point_second()->get_x() : (*it_2)->get_point_second()->get_y();
        if (coord_1_begin > coord_1_end) {
          std::swap(coord_1_begin, coord_1_end);
        }
        if (coord_2_begin > coord_2_end) {
          std::swap(coord_2_begin, coord_2_end);
        }

        if (coord_1_end >= coord_2_begin) {
          auto end_coord = std::max(coord_1_end, coord_2_end);
          b_horizontal ? (*it_1)->get_point_second()->set_x(end_coord) : (*it_1)->get_point_second()->set_y(end_coord);

          if ((*it_1)->is_via()) {
            IdbRegularWireSegment* via_seg = split_segment((*it_1), false);
            if (via_seg != nullptr) {
              new_segments.push_back(via_seg);
            }
          }

          if ((*it_2)->is_via()) {
            split_segment((*it_2), true);
          } else {
            delete_segments.push_back((*it_2));
          }
        } else {
          it_1 = it_2;
        }
      }
    }

    return std::make_pair(new_segments, delete_segments);
  };

  auto update_net = [&](std::vector<IdbRegularWireSegment*>& new_segments, std::vector<IdbRegularWireSegment*>& delete_segments) {
    for (auto* del_seg : delete_segments) {
      remove_segment(del_seg);
    }

    if (!new_segments.empty()) {
      if (_wire_list->get_wire_list().empty()) {
        _wire_list->add_wire();
      }
      auto& wire_segs = _wire_list->get_wire_list()[0]->get_segment_list();
      wire_segs.insert(wire_segs.end(), new_segments.begin(), new_segments.end());
    }
  };

  struct LayerData
  {
    std::map<int, std::vector<IdbRegularWireSegment*>> horizontal_map;
    std::map<int, std::vector<IdbRegularWireSegment*>> vertical_map;
  };
  std::map<int, LayerData> layer_map;

  for (auto* wire : _wire_list->get_wire_list()) {
    if (wire == nullptr) {
      continue;
    }
    for (auto* segment : wire->get_segment_list()) {
      if (segment == nullptr || segment->get_layer() == nullptr || !segment->is_wire()) {
        continue;
      }

      auto& layer_data = layer_map[segment->get_layer()->get_order()];
      auto* point_start = segment->get_point_start();
      auto* point_end = segment->get_point_end();
      if (point_start == nullptr || point_end == nullptr) {
        continue;
      }

      if (point_start->get_y() == point_end->get_y()) {
        layer_data.horizontal_map[point_start->get_y()].emplace_back(segment);
      } else {
        layer_data.vertical_map[point_start->get_x()].emplace_back(segment);
      }
    }
  }

  for (auto& [layer_id, layer_data] : layer_map) {
    auto [new_segs_h, del_segs_h] = merge(layer_data.horizontal_map, true);
    auto [new_segs_v, del_segs_v] = merge(layer_data.vertical_map, false);
    update_net(new_segs_h, del_segs_h);
    update_net(new_segs_v, del_segs_v);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbNetList::IdbNetList()
{
}

IdbNetList::~IdbNetList()
{
  for (auto& net : _net_list) {
    if (net != nullptr) {
      delete net;
      net = nullptr;
    }
  }

  _net_list.clear();
}

IdbNet* IdbNetList::find_net(string name)
{
  //   for (IdbNet* net : _net_list) {
  //     if (net->get_net_name() == name) {
  //       return net;
  //     }
  //   }

  //   return nullptr;
  auto net_pair = _net_map.find(name);
  if (net_pair != _net_map.end()) {
    return net_pair->second;
  }

  return nullptr;
}

IdbNet* IdbNetList::find_net(size_t index)
{
  if (_net_list.size() > index) {
    return _net_list.at(index);
  }

  return nullptr;
}

bool IdbNetList::rename_net(IdbNet* net, string new_name)
{
  if (net == nullptr || new_name.empty()) {
    return false;
  }

  if (find_net(new_name) != nullptr && find_net(new_name) != net) {
    return false;
  }

  const string old_name = net->get_net_name();
  if (!old_name.empty()) {
    auto map_iter = _net_map.find(old_name);
    if (map_iter != _net_map.end() && map_iter->second == net) {
      _net_map.erase(map_iter);
    }
  }

  net->set_net_name(new_name);
  _net_map[new_name] = net;

  std::vector<IdbPin*> pin_list;
  if (net->get_io_pins() != nullptr) {
    auto& io_pins = net->get_io_pins()->get_pin_list();
    pin_list.insert(pin_list.end(), io_pins.begin(), io_pins.end());
  }
  if (net->get_instance_pin_list() != nullptr) {
    auto& inst_pins = net->get_instance_pin_list()->get_pin_list();
    pin_list.insert(pin_list.end(), inst_pins.begin(), inst_pins.end());
  }
  for (auto* pin : pin_list) {
    if (pin != nullptr && pin->get_net() == net) {
      pin->set_net_name(new_name);
    }
  }

  return true;
}

IdbNet* IdbNetList::add_net(IdbNet* net)
{
  IdbNet* pNet = net;
  if (pNet == nullptr) {
    pNet = new IdbNet();
  }

  if (!pNet->get_net_name().empty()) {
    IdbNet* existed_net = find_net(pNet->get_net_name());
    if (existed_net != nullptr) {
      return existed_net;
    }
  }

  pNet->set_id(_mutex_index++);
  _net_list.emplace_back(pNet);
  if (!pNet->get_net_name().empty()) {
    _net_map[pNet->get_net_name()] = pNet;
  }

  return pNet;
}

IdbNet* IdbNetList::add_net(string name, IdbConnectType type)
{
  IdbNet* existed_net = find_net(name);
  if (existed_net != nullptr) {
    return existed_net;
  }

  IdbNet* pNet = new IdbNet();
  pNet->set_id(_mutex_index++);
  pNet->set_net_name(name);
  pNet->set_connect_type(type);
  _net_map[name] = pNet;
  _net_list.emplace_back(pNet);

  return pNet;
}

/**
 * @Brief : remove net in net list
 * @param  name
 * @return true
 * @return false
 */
bool IdbNetList::remove_net(string name)
{
  /// remove net from net map
  auto it_map = _net_map.find(name);
  if (it_map != _net_map.end()) {
    it_map = _net_map.erase(it_map);
  }

  /// remove net from netlist
  auto it = std::find_if(_net_list.begin(), _net_list.end(), [name](auto net) { return name == net->get_net_name(); });
  if (it == _net_list.end()) {
    return false;
  }

  if (*it != nullptr) {
    std::vector<IdbPin*> pin_list;
    auto* net = *it;
    if (net->get_io_pins() != nullptr) {
      auto& io_pins = net->get_io_pins()->get_pin_list();
      pin_list.insert(pin_list.end(), io_pins.begin(), io_pins.end());
    }
    if (net->get_instance_pin_list() != nullptr) {
      auto& inst_pins = net->get_instance_pin_list()->get_pin_list();
      pin_list.insert(pin_list.end(), inst_pins.begin(), inst_pins.end());
    }

    for (auto* pin : pin_list) {
      if (pin != nullptr && pin->get_net() == net) {
        pin->remove_net();
      }
    }
  }

  /// delete net & release resource
  delete *it;
  *it = nullptr;
  _net_list.erase(it);

  return true;
}

/**
 * @Brief : remove all the wire in the net
 */
void IdbNetList::clear_wire_list()
{
  for (IdbNet* net : _net_list) {
    net->clear_wire_list();
  }
}

/**
 * @Brief : check the connection for all the nets
 * @return true
 * @return false
 */
bool IdbNetList::checkConnection()
{
  bool b_result = false;
  for (IdbNet* net : _net_list) {
    if (!net->checkConnection()) {
    }
  }

  return b_result;
}

uint64_t IdbNetList::maxFanout()
{
  uint64_t net_len = 0;
  for (auto net : _net_list) {
    net_len += net->wireLength();
  }
  return net_len;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool IdbNetChecker::checkNetConnection(IdbNet* net)
{
  bool b_result = false;

  // /// build pin
  // for (IdbPin* pin : net->get_instance_pin_list()->get_pin_list()) {
  // }

  // /// build wire
  // for (IdbRegularWire* wire : net->get_wire_list()->get_wire_list()) {
  //   for (IdbRegularWireSegment* segment : wire->get_segment_list()) {
  //   }
  // }

  if (!b_result) {
    std::cout << "[IdbNetList Error] Net connected failed. Net name = " << net->get_net_name() << std::endl;
  }

  return b_result;
}

}  // namespace idb
