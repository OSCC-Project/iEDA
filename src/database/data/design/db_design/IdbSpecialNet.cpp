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
 * @file		IdbSpecialNet.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Defines netlist connectivity and special-routes for nets containing special pins.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbSpecialNet.h"

#include <algorithm>

namespace idb {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IdbSpecialNet::IdbSpecialNet()
{
  _net_name = "";
  _connect_type = IdbConnectType::kNone;
  _source_type = IdbInstanceType::kNone;
  _weight = 0;

  _io_pin_list = new IdbPins();
  _instance_pin_list = new IdbPins();
  _instance_list = new IdbInstanceList();
  _wire_list = new IdbSpecialWireList();
}

IdbSpecialNet::~IdbSpecialNet()
{
  if (_io_pin_list) {
    delete _io_pin_list;
    _io_pin_list = nullptr;
  }

  if (_instance_pin_list) {
    delete _instance_pin_list;
    _instance_pin_list = nullptr;
  }

  _instance_list->reset(false);
  if (_instance_list) {
    delete _instance_list;
    _instance_list = nullptr;
  }

  if (_wire_list) {
    delete _wire_list;
    _wire_list = nullptr;
  }

  _pin_string_list.clear();
  std::vector<std::string>().swap(_pin_string_list);
}

void IdbSpecialNet::set_connect_type(string type)
{
  set_connect_type(IdbEnum::GetInstance()->get_connect_property()->get_type(type));
}

void IdbSpecialNet::set_source_type(string type)
{
  _source_type = IdbEnum::GetInstance()->get_instance_property()->get_type(type);
}

void IdbSpecialNet::add_io_pin(IdbPin* io_pin)
{
  _io_pin_list->add_pin_ref_unique(io_pin);
}

void IdbSpecialNet::add_instance_pin(IdbPin* inst_pin)
{
  _instance_pin_list->add_pin_ref_unique(inst_pin);
}

void IdbSpecialNet::add_instance(IdbInstance* instance)
{
  _instance_list->add_instance_ref(instance);
}

void IdbSpecialNet::add_wildcard_instance_pin(const string& term_name)
{
  if (term_name.empty() || matches_wildcard_instance_pin(term_name)) {
    return;
  }

  _pin_string_list.emplace_back(term_name);
}

bool IdbSpecialNet::matches_wildcard_instance_pin(const string& term_name) const
{
  return std::find(_pin_string_list.begin(), _pin_string_list.end(), term_name) != _pin_string_list.end();
}

bool IdbSpecialNet::has_io_pin(IdbPin* io_pin)
{
  return _io_pin_list != nullptr && _io_pin_list->contains(io_pin);
}

bool IdbSpecialNet::has_instance_pin(IdbPin* inst_pin)
{
  return _instance_pin_list != nullptr && _instance_pin_list->contains(inst_pin);
}

bool IdbSpecialNet::has_instance(IdbInstance* instance)
{
  return _instance_list != nullptr && _instance_list->contains(instance);
}

bool IdbSpecialNet::erase_pin_ref(IdbPin* pin)
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

bool IdbSpecialNet::erase_instance_ref(IdbInstance* instance)
{
  if (_instance_list == nullptr || instance == nullptr) {
    return false;
  }

  return _instance_list->erase_instance_ref(instance);
}

/// get the width for this net in layer "layer name"
int32_t IdbSpecialNet::get_layer_width(string layer_name)
{
  if (_wire_list->get_num() <= 0) {
    return -1;
  }

  for (IdbSpecialWire* wire : _wire_list->get_wire_list()) {
    for (IdbSpecialWireSegment* segment : wire->get_segment_list()) {
      /// find the first segment that is type of stripe and on the layer name
      if ((segment->is_line()) && segment->get_layer()->get_name() == layer_name) {
        return segment->get_route_width();
      }
    }
  }

  return -1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbSpecialNetList::IdbSpecialNetList()
{
}

IdbSpecialNetList::~IdbSpecialNetList()
{
  for (auto* net : _net_list) {
    if (nullptr != net) {
      delete net;
      net = nullptr;
    }
  }

  _net_list.clear();
  std::vector<IdbSpecialNet*>().swap(_net_list);

  for (auto* edge : _edge_segment_list) {
    if (nullptr != edge) {
      delete edge;
      edge = nullptr;
    }
  }

  _edge_segment_list.clear();
  std::vector<IdbSpecialNetEdgeSegmenArray*>().swap(_edge_segment_list);
}

/// init all edge list for each layer
void IdbSpecialNetList::initEdge(IdbLayers* layers)
{
  /// celar edge list
  clear_edge_list();
  /// init all layer edge
  for (IdbLayer* layer : layers->get_routing_layers()) {
    IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(layer);
    add_edge_segment_array_for_layer(routing_layer);
  }

  /// construct all edge segment of points
  for (IdbSpecialNet* net : _net_list) {
    SegmentType type = SegmentType ::kNone;

    if (net->is_vdd()) {
      type = SegmentType::kVDD;
    } else if (net->is_vss()) {
      type = SegmentType::kVSS;
    } else {
      continue;
    }

    for (IdbSpecialWire* wire : net->get_wire_list()->get_wire_list()) {
      for (IdbSpecialWireSegment* segment : wire->get_segment_list()) {
        /// only use the stripe wire to calculate the coordinate
        if (segment->get_point_list().size() < _POINT_MAX_ || !(segment->is_tripe() || segment->is_follow_pin())) {
          continue;
        }

        IdbLayer* layer = segment->get_layer();
        IdbSpecialNetEdgeSegmenArray* edge_array = find_edge_segment_array_by_layer(layer);
        if (edge_array != nullptr) {
          edge_array->updateSegment(segment, wire, type);
        }
      }
    }
  }

  std::cout << "Init Special Net Edge." << std::endl;
}

IdbSpecialNet* IdbSpecialNetList::find_net(string name)
{
  for (IdbSpecialNet* net : _net_list) {
    if (net->get_net_name() == name) {
      return net;
    }
  }

  return nullptr;
}

IdbSpecialNet* IdbSpecialNetList::find_net(size_t index)
{
  if (_net_list.size() > index) {
    return _net_list.at(index);
  }

  return nullptr;
}

IdbSpecialNet* IdbSpecialNetList::add_net(IdbSpecialNet* net)
{
  IdbSpecialNet* pNet = net;
  if (pNet == nullptr) {
    pNet = new IdbSpecialNet();
  }

  if (!pNet->get_net_name().empty()) {
    IdbSpecialNet* existed_net = find_net(pNet->get_net_name());
    if (existed_net != nullptr) {
      return existed_net;
    }
  }

  _net_list.emplace_back(pNet);

  return pNet;
}

IdbSpecialNet* IdbSpecialNetList::add_net(string name)
{
  IdbSpecialNet* existed_net = find_net(name);
  if (existed_net != nullptr) {
    return existed_net;
  }

  IdbSpecialNet* pNet = new IdbSpecialNet();
  pNet->set_net_name(name);
  _net_list.emplace_back(pNet);

  return pNet;
}

bool IdbSpecialNetList::remove_net(string name)
{
  auto it = std::find_if(_net_list.begin(), _net_list.end(), [name](auto net) { return net != nullptr && name == net->get_net_name(); });
  if (it == _net_list.end()) {
    return false;
  }

  if (*it != nullptr) {
    std::vector<IdbPin*> pin_list;
    auto* net = *it;
    if (net->get_io_pin_list() != nullptr) {
      auto& io_pins = net->get_io_pin_list()->get_pin_list();
      pin_list.insert(pin_list.end(), io_pins.begin(), io_pins.end());
    }
    if (net->get_instance_pin_list() != nullptr) {
      auto& inst_pins = net->get_instance_pin_list()->get_pin_list();
      pin_list.insert(pin_list.end(), inst_pins.begin(), inst_pins.end());
    }

    for (auto* pin : pin_list) {
      if (pin != nullptr && pin->get_special_net() == net) {
        pin->remove_special_net();
      }
    }
  }

  delete *it;
  *it = nullptr;
  _net_list.erase(it);
  return true;
}

IdbSpecialWire* IdbSpecialNetList::generateWire(string net_name)
{
  IdbSpecialWire* wire = nullptr;

  IdbSpecialNet* net = find_net(net_name);
  if (net != nullptr && net->get_wire_list() != nullptr) {
    IdbSpecialWireList* wire_list = net->get_wire_list();
    wire = wire_list->get_num() > 0 ? wire_list->find_wire(0) : wire_list->add_wire(nullptr);
    wire->set_wire_state(IdbWiringStatement::kRouted);
  }

  return wire;
}

///////////operate manually

IdbSpecialNetEdgeSegmenArray* IdbSpecialNetList::find_edge_segment_array_by_layer(IdbLayer* layer)
{
  if (layer == nullptr) {
    return nullptr;
  }

  for (IdbSpecialNetEdgeSegmenArray* segment_array : _edge_segment_list) {
    if (segment_array->get_layer()->compareLayer(layer->get_name())) {
      return segment_array;
    }
  }

  return nullptr;
}

IdbSpecialNetEdgeSegmenArray* IdbSpecialNetList::add_edge_segment_array(IdbSpecialNetEdgeSegmenArray* edge_segment)
{
  IdbSpecialNetEdgeSegmenArray* pSegment = edge_segment;
  if (pSegment == nullptr) {
    pSegment = new IdbSpecialNetEdgeSegmenArray();
  }

  _edge_segment_list.emplace_back(pSegment);

  return pSegment;
}

IdbSpecialNetEdgeSegmenArray* IdbSpecialNetList::add_edge_segment_array_for_layer(IdbLayerRouting* layer)
{
  IdbSpecialNetEdgeSegmenArray* pSegment = find_edge_segment_array_by_layer(layer);
  if (pSegment == nullptr) {
    pSegment = new IdbSpecialNetEdgeSegmenArray();
    pSegment->set_layer(layer);
  }

  _edge_segment_list.emplace_back(pSegment);

  return pSegment;
}

void IdbSpecialNetList::clear_edge_list()
{
  for (IdbSpecialNetEdgeSegmenArray* edge_segment_array : _edge_segment_list) {
    if (edge_segment_array != nullptr) {
      delete edge_segment_array;
      edge_segment_array = nullptr;
    }
  }

  _edge_segment_list.clear();
  vector<IdbSpecialNetEdgeSegmenArray*>().swap(_edge_segment_list);
}

bool IdbSpecialNetList::connectIO(vector<IdbCoordinate<int32_t>*>& point_list, IdbLayer* layer)
{
  if (point_list.size() < _POINT_MAX_ || layer == nullptr) {
    return false;
  }
  /// find segment array of layer
  IdbSpecialNetEdgeSegmenArray* layer_segment_array = find_edge_segment_array_by_layer(layer);
  if (layer_segment_array == nullptr) {
    std::cout << "Error : can not find edge." << std::endl;
    return false;
  }

  /// find the segment connected and adjust the point list by segment
  return layer_segment_array->addSegmentByCoordinateList(point_list);
}

bool IdbSpecialNetList::addPowerStripe(vector<IdbCoordinate<int32_t>*>& point_list, string net_name, string layer_name)
{
  IdbSpecialNet* net = find_net(net_name);
  if (net == nullptr) {
    std::cout << "Error : can't find the net. " << std::endl;
    return false;
  }

  IdbSpecialWire* wire = net->get_wire_list()->get_num() > 0 ? net->get_wire_list()->find_wire(0) : net->get_wire_list()->add_wire(nullptr);
  if (wire == nullptr) {
    std::cout << "Error : can't get the wire." << std::endl;
    return false;
  }

  IdbSpecialWireSegment* segment = wire->get_layer_segment(layer_name);
  if (segment == nullptr) {
    std::cout << "Error : can't find any power stripe in the net." << std::endl;
    return false;
  }

  return wire->add_segment_list(point_list, segment);
}

}  // namespace idb
