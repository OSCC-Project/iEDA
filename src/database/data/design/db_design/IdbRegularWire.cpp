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
 * @file		IdbRegularwire.h
 * @date		25/05/2021
 * @version		0.1
 * @description
 Regular

        Defines regular wire connectivity and special-routes for nets containing special pins.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbRegularWire.h"

#include <algorithm>

namespace idb {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IdbRegularWireSegment::IdbRegularWireSegment()
{
  _is_new_layer = false;
  _layer_name = "";
  _layer = nullptr;
  // _route_width = -1;
  // _shape_type = IdbWireShapeType::kNone;
  // _style = -1;
  _is_via = false;

  _is_rect = false;
  _delta_rect = nullptr;
}

IdbRegularWireSegment::~IdbRegularWireSegment()
{
  clear();

  if (_delta_rect != nullptr) {
    delete _delta_rect;
    _delta_rect = nullptr;
  }
}

void IdbRegularWireSegment::clearPoints()
{
  for (auto* point : _point_list) {
    if (point != nullptr) {
      delete point;
      point = nullptr;
    }
  }
  _point_list.clear();
  std::vector<IdbCoordinate<int32_t>*>().swap(_point_list);
}

void IdbRegularWireSegment::clear()
{
  clearPoints();

  if (_via_list.size() > 0) {
    for (auto* via : _via_list) {
      if (via != nullptr) {
        delete via;
        via = nullptr;
      }
    }
  }
  _via_list.clear();
  vector<IdbVia*>().swap(_via_list);

  _virtual_points.clear();
}

// void IdbRegularWireSegment::set_shape_type(string type)
// {
//     set_shape_type(IdbEnum::GetInstance()->get_connect_property()->get_wire_shape(type));
// }

IdbCoordinate<int32_t>* IdbRegularWireSegment::get_point(size_t index)
{
  IdbCoordinate<int32_t>* point = nullptr;
  if ((_point_list.size() > index) && (index >= 0)) {
    point = _point_list.at(index);
  }

  return point;
}

IdbCoordinate<int32_t>* IdbRegularWireSegment::get_point_start()
{
  if (_point_list.size() > _POINT_START_) {
    return get_point(_POINT_START_);
  }

  return nullptr;
}

IdbCoordinate<int32_t>* IdbRegularWireSegment::get_point_second()
{
  if (_point_list.size() > _POINT_SECOND_) {
    return get_point(_POINT_SECOND_);
  }

  return nullptr;
}

IdbCoordinate<int32_t>* IdbRegularWireSegment::get_point_end()
{
  int size = _point_list.size();

  return size > 0 ? get_point(size - 1) : nullptr;
}
/**
 * get rect shape for wire & delta rect
 */
idb::IdbRect IdbRegularWireSegment::get_segment_rect()
{
  if (is_rect()) {
    return get_delta_rect();
  } else {
    int32_t routing_width = dynamic_cast<IdbLayerRouting*>(_layer)->get_width();
    IdbCoordinate<int32_t>* point_1 = get_point_start();
    IdbCoordinate<int32_t>* point_2 = get_point_second();

    int32_t ll_x = 0;
    int32_t ll_y = 0;
    int32_t ur_x = 0;
    int32_t ur_y = 0;
    if (point_1->get_y() == point_2->get_y()) {
      // horizontal
      ll_x = std::min(point_1->get_x(), point_2->get_x()) - routing_width / 2;
      ll_y = std::min(point_1->get_y(), point_2->get_y()) - routing_width / 2;
      ur_x = std::max(point_1->get_x(), point_2->get_x()) + routing_width / 2;
      ur_y = ll_y + routing_width;
    } else {
      // vertical
      ll_x = std::min(point_1->get_x(), point_2->get_x()) - routing_width / 2;
      ll_y = std::min(point_1->get_y(), point_2->get_y()) - routing_width / 2;
      ur_x = ll_x + routing_width;
      ur_y = std::max(point_1->get_y(), point_2->get_y()) + routing_width / 2;
    }

    return idb::IdbRect(ll_x, ll_y, ur_x, ur_y);
  }
}

IdbCoordinate<int32_t>* IdbRegularWireSegment::add_point(int32_t x, int32_t y)
{
  IdbCoordinate<int32_t>* point = new IdbCoordinate<int32_t>(x, y);
  _point_list.emplace_back(point);

  return point;
}
IdbCoordinate<int32_t>* IdbRegularWireSegment::add_virtual_point(int32_t x, int32_t y)
{
  return *_virtual_points.insert(add_point(x, y)).first;
}

IdbVia* IdbRegularWireSegment::copy_via(IdbVia* via)
{
  if (via != nullptr) {
    IdbVia* via_new = via->clone();

    _via_list.emplace_back(via_new);

    return via_new;
  }

  return nullptr;
}

void IdbRegularWireSegment::set_via_list(vector<IdbVia*> via_list)
{
  if (_via_list.size() > 0) {
    set_is_via(true);
  } else {
    set_is_via(false);
  }
  _via_list = via_list;
}

void IdbRegularWireSegment::set_delta_rect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y)
{
  _delta_rect = new IdbRect(ll_x, ll_y, ur_x, ur_y);
}

uint64_t IdbRegularWireSegment::length()
{
  if (is_via()) {
    /// thickness
    // return 0;
  }
  if (is_rect()) {
    /// do nothing
  } else {
    if (_point_list.size() == _POINT_MAX_) {
      IdbCoordinate<int32_t>* pt1 = get_point_start();
      IdbCoordinate<int32_t>* pt2 = get_point_second();

      if (pt1->get_y() == pt2->get_y()) {
        /// horizontal
        return std::abs(pt1->get_x() - pt2->get_x());
      } else if (pt1->get_x() == pt2->get_x()) {
        /// vertical
        return std::abs(pt1->get_y() - pt2->get_y());
      } else {
        std::cout << "[Idb Error} Net segment error." << std::endl;
      }
    }
  }
  return 0;
}

bool IdbRegularWireSegment::isIntersection(IdbLayerShape* layer_shape)
{
  if (layer_shape == nullptr) {
    return false;
  }

  auto layer = dynamic_cast<IdbLayerRouting*>(_layer);

  if (is_via()) {
    /// must be on the same layer
    for (auto seg_via : get_via_list()) {
      IdbLayerShape bottom_shape = seg_via->get_bottom_layer_shape();
      IdbLayerShape top_shape = seg_via->get_top_layer_shape();

      IdbLayerShape* connect_seg_shape = nullptr;
      /// top
      if (layer_shape->get_layer()->get_order() == bottom_shape.get_layer()->get_order()) {
        connect_seg_shape = &bottom_shape;
      }

      if (layer_shape->get_layer()->get_order() == top_shape.get_layer()->get_order()) {
        connect_seg_shape = &top_shape;
      }

      /// no intersected layer
      if (connect_seg_shape == nullptr) {
        return false;
      }

      /// check connection
      for (auto rect : layer_shape->get_rect_list()) {
        for (auto via_rect : connect_seg_shape->get_rect_list()) {
          if (rect->isIntersection(via_rect)) {
            return true;
          }
        }
      }
    }
  }

  if (is_rect()) {
    /// must be on the same layer
    if (_layer->get_order() != layer_shape->get_layer()->get_order()) {
      return false;
    }

    for (auto rect : layer_shape->get_rect_list()) {
      if (_delta_rect->isIntersection(rect)) {
        return true;
      }
    }
  }

  if (is_wire()) {
    /// must be on the same layer
    if (_layer->get_order() != layer_shape->get_layer()->get_order()) {
      return false;
    }

    IdbRect this_rect(get_point_start(), get_point_second(), layer->get_width());
    for (auto rect : layer_shape->get_rect_list()) {
      if (this_rect.isIntersection(rect)) {
        return true;
      }
    }
  }

  return false;
}

bool IdbRegularWireSegment::isIntersection(IdbRegularWireSegment* segment)
{
  if (segment == nullptr) {
    return false;
  }

  bool b_conneted = false;

  if (is_wire() && segment->is_wire()) {
    b_conneted |= isConnectWireToWire(segment);
  }

  if (is_wire() && segment->is_via()) {
    b_conneted |= isConnectWireToVia(segment);
  }

  if (is_via() && segment->is_wire()) {
    b_conneted |= segment->isConnectWireToVia(this);
  }

  if (is_via() && segment->is_via()) {
    b_conneted |= isConnectViaToVia(segment);
  }

  if (is_wire() && segment->is_rect()) {
    b_conneted |= isConnectWireToDeltaRect(segment);
  }

  if (is_rect() && segment->is_wire()) {
    b_conneted |= segment->isConnectWireToDeltaRect(this);
  }

  if (is_rect() && segment->is_via()) {
    b_conneted |= isConnectRectToVia(segment);
  }

  if (is_via() && segment->is_rect()) {
    b_conneted |= segment->isConnectRectToVia(this);
  }

  if (is_rect() && segment->is_rect()) {
    b_conneted |= isConnectRectToRect(segment);
  }

  return b_conneted;
}
/**
 * @brief check connection between two wire segment in a routing layer
 *
 * @param segment target segment
 * @return true connected
 * @return false unconnected
 */
bool IdbRegularWireSegment::isConnectWireToWire(IdbRegularWireSegment* segment)
{
  if (segment == nullptr) {
    return false;
  }

  /// must be on the same layer
  if (_layer->get_order() != segment->get_layer()->get_order()) {
    return false;
  }

  auto layer = dynamic_cast<IdbLayerRouting*>(_layer);

  IdbRect this_rect(get_point_start(), get_point_second(), layer->get_width());
  IdbRect seg_rect(segment->get_point_start(), segment->get_point_second(), layer->get_width());

  return this_rect.isIntersection(seg_rect);

  //   /// if intersected
  //   auto this_coord_1 = get_point_start();
  //   auto this_coord_2 = get_point_second();

  //   auto seg_coord_1 = segment->get_point_start();
  //   auto seg_coord_2 = segment->get_point_second();

  //   // check intersection
  //   int min_node_1_x = std::min(this_coord_1->get_x(), this_coord_2->get_x());
  //   int min_node_1_y = std::min(this_coord_1->get_y(), this_coord_2->get_y());
  //   int max_node_1_x = std::max(this_coord_1->get_x(), this_coord_2->get_x());
  //   int max_node_1_y = std::max(this_coord_1->get_y(), this_coord_2->get_y());

  //   int min_node_2_x = std::min(seg_coord_1->get_x(), seg_coord_2->get_x());
  //   int min_node_2_y = std::min(seg_coord_1->get_y(), seg_coord_2->get_y());
  //   int max_node_2_x = std::max(seg_coord_1->get_x(), seg_coord_2->get_x());
  //   int max_node_2_y = std::max(seg_coord_1->get_y(), seg_coord_2->get_y());

  //   // no intersection
  //   if (min_node_1_x > max_node_2_x || max_node_1_x < min_node_2_x || min_node_1_y > max_node_2_y || max_node_1_y < min_node_2_y) {
  //     return false;
  //   }
  //
  //   return true;
}

bool IdbRegularWireSegment::isConnectWireToDeltaRect(IdbRegularWireSegment* segment)
{
  if (segment == nullptr) {
    return false;
  }

  /// must be on the same layer
  if (_layer->get_order() != segment->get_layer()->get_order()) {
    return false;
  }

  auto layer = dynamic_cast<IdbLayerRouting*>(_layer);

  IdbRect this_rect(get_point_start(), get_point_second(), layer->get_width());

  return this_rect.isIntersection(segment->get_delta_rect());
}

bool IdbRegularWireSegment::isConnectWireToVia(IdbRegularWireSegment* segment)
{
  if (segment == nullptr) {
    return false;
  }

  /// must be on the same layer
  for (auto seg_via : segment->get_via_list()) {
    IdbLayerShape bottom_shape = seg_via->get_bottom_layer_shape();
    IdbLayerShape top_shape = seg_via->get_top_layer_shape();

    IdbLayerShape* connect_seg_shape = nullptr;
    /// top
    if (_layer->get_order() == bottom_shape.get_layer()->get_order()) {
      connect_seg_shape = &bottom_shape;
    }

    if (_layer->get_order() == top_shape.get_layer()->get_order()) {
      connect_seg_shape = &top_shape;
    }

    /// no intersected layer
    if (connect_seg_shape == nullptr) {
      return false;
    }

    /// check connection
    auto layer = dynamic_cast<IdbLayerRouting*>(_layer);
    IdbRect this_rect(get_point_start(), get_point_second(), layer->get_width());
    for (auto seg_rect : connect_seg_shape->get_rect_list()) {
      if (this_rect.isIntersection(seg_rect)) {
        return true;
      }
    }
  }

  return false;
}

bool IdbRegularWireSegment::isConnectViaToVia(IdbRegularWireSegment* segment)
{
  if (segment == nullptr) {
    return false;
  }

  for (auto this_via : _via_list) {
    auto this_shape_bottom = this_via->get_bottom_layer_shape();
    auto this_shape_top = this_via->get_top_layer_shape();
    for (auto seg_via : segment->get_via_list()) {
      auto seg_shape_bottom = seg_via->get_bottom_layer_shape();
      auto seg_shape_top = seg_via->get_top_layer_shape();
      /// if two via connected, top layer of one via must be connected to bottom layer of the other via

      IdbLayerShape* layer_shape_1 = nullptr;
      IdbLayerShape* layer_shape_2 = nullptr;
      if (this_shape_bottom.get_layer()->get_order() == seg_shape_top.get_layer()->get_order()) {
        layer_shape_1 = &this_shape_bottom;
        layer_shape_2 = &seg_shape_top;
      }

      if (this_shape_top.get_layer()->get_order() == seg_shape_bottom.get_layer()->get_order()) {
        layer_shape_1 = &this_shape_top;
        layer_shape_2 = &seg_shape_bottom;
      }
      /// if layer shape not exist, igonore
      if (layer_shape_1 == nullptr || layer_shape_2 == nullptr) {
        continue;
      }

      for (auto this_rect : layer_shape_1->get_rect_list()) {
        for (auto seg_rect : layer_shape_2->get_rect_list()) {
          if (this_rect->isIntersection(seg_rect)) {
            return true;
          }
        }
      }
    }
  }

  return false;
}

bool IdbRegularWireSegment::isConnectRectToVia(IdbRegularWireSegment* segment)
{
  if (segment == nullptr) {
    return false;
  }

  /// must be on the same layer
  for (auto seg_via : segment->get_via_list()) {
    IdbLayerShape bottom_shape = seg_via->get_bottom_layer_shape();
    IdbLayerShape top_shape = seg_via->get_top_layer_shape();

    IdbLayerShape* connect_seg_shape = nullptr;
    /// top
    if (_layer->get_order() == bottom_shape.get_layer()->get_order()) {
      connect_seg_shape = &bottom_shape;
    }

    if (_layer->get_order() == top_shape.get_layer()->get_order()) {
      connect_seg_shape = &top_shape;
    }

    /// no intersected layer
    if (connect_seg_shape == nullptr) {
      return false;
    }

    /// check connection
    for (auto seg_rect : connect_seg_shape->get_rect_list()) {
      if (_delta_rect->isIntersection(seg_rect)) {
        return true;
      }
    }
  }

  return false;
}

bool IdbRegularWireSegment::isConnectRectToRect(IdbRegularWireSegment* segment)
{
  if (segment == nullptr) {
    return false;
  }

  /// must be on the same layer
  if (_layer->get_order() != segment->get_layer()->get_order()) {
    return false;
  }

  return _delta_rect->isIntersection(segment->get_delta_rect());
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbRegularWire::IdbRegularWire()
{
  _wire_state = IdbWiringStatement::kNone;
}

IdbRegularWire::~IdbRegularWire()
{
  for (auto* segment : _segment_list) {
    if (segment != nullptr) {
      delete segment;
      segment = nullptr;
    }
  }

  _segment_list.clear();
  std::vector<IdbRegularWireSegment*>().swap(_segment_list);
}

uint IdbRegularWire::get_via_num()
{
  uint number = 0;
  for (auto& segment : _segment_list) {
    if (segment->is_via()) {
      number += segment->get_via_list().size();
    }
  }

  return number;
}

void IdbRegularWire::set_wire_state(string state)
{
  set_wire_state(IdbEnum::GetInstance()->get_connect_property()->get_wiring_state(state));

  //----------------------tbd---------------------
  // wireShield not support
}

IdbRegularWireSegment* IdbRegularWire::add_segment(IdbRegularWireSegment* segment)
{
  IdbRegularWireSegment* psegment = segment;
  if (psegment == nullptr) {
    psegment = new IdbRegularWireSegment();
  }
  _segment_list.emplace_back(psegment);

  return psegment;
}

IdbRegularWireSegment* IdbRegularWire::add_segment(string layer_name)
{
  IdbRegularWireSegment* segment = new IdbRegularWireSegment();
  segment->set_layer_name(layer_name);
  _segment_list.emplace_back(segment);

  return segment;
}

void IdbRegularWire::delete_seg(IdbRegularWireSegment* seg_del)
{
  _segment_list.erase(std::find(_segment_list.begin(), _segment_list.end(), seg_del));
  delete seg_del;
  seg_del = nullptr;
}

void IdbRegularWire::clear_segment()
{
  for (auto& segment : _segment_list) {
    if (segment != nullptr) {
      segment->clear();
      delete segment;
      segment = nullptr;
    }
  }

  _segment_list.clear();
  vector<IdbRegularWireSegment*>().swap(_segment_list);
}

uint64_t IdbRegularWire::wireLength()
{
  uint64_t wire_len = 0;
  for (auto seg : _segment_list) {
    wire_len += seg->length();
  }
  return wire_len;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbRegularWireList::IdbRegularWireList()
{
}

IdbRegularWireList::~IdbRegularWireList()
{
  clear();
}

void IdbRegularWireList::clear()
{
  for (auto* wire : _wire_list) {
    if (wire != nullptr) {
      wire->clear_segment();
      delete wire;
      wire = nullptr;
    }
  }
  _wire_list.clear();
  vector<IdbRegularWire*>().swap(_wire_list);
}

// IdbSpecialWire* IdbSpecialWireList::find_wire(string name)
// {
//     for(IdbSpecialWire* wire : _wire_list)
//     {
//         if(wire->get_name() == name)
//         {
//             return net;
//         }
//     }

//     return nullptr;
// }

// IdbSpecialWire* IdbSpecialWireList::find_wire(size_t index)
// {
//     if(_num > index)
//     {
//         return _wire_list.at(index);
//     }

//     return nullptr;
// }

IdbRegularWire* IdbRegularWireList::add_wire(IdbRegularWire* wire)
{
  IdbRegularWire* pWire = wire;
  if (pWire == nullptr) {
    pWire = new IdbRegularWire();
  }
  _wire_list.emplace_back(pWire);

  return pWire;
}

uint64_t IdbRegularWireList::wireLength()
{
  uint64_t total_len = 0;
  for (auto wire : _wire_list) {
    total_len += wire->wireLength();
  }
  return total_len;
}

}  // namespace idb
