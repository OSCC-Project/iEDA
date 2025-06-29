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
#include "IdbSpecialWire.h"

#include <algorithm>

namespace idb {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IdbSpecialWireSegment::IdbSpecialWireSegment()
{
  _is_new_layer = false;
  // _layer_name = "";
  _layer = nullptr;
  _route_width = -1;
  _shape_type = IdbWireShapeType::kNone;
  _style = -1;
  _is_via = false;
  _is_rect = false;
  _via = nullptr;  // new IdbVia();
  _delta_rect = nullptr;
}

IdbSpecialWireSegment::~IdbSpecialWireSegment()
{
  for (IdbCoordinate<int32_t>* point : _point_list) {
    if (point) {
      delete point;
      point = nullptr;
    }
  }

  if (_via) {
    delete _via;
    _via = nullptr;
  }
}

void IdbSpecialWireSegment::set_shape_type(string type)
{
  set_shape_type(IdbEnum::GetInstance()->get_connect_property()->get_wire_shape(type));
}

IdbCoordinate<int32_t>* IdbSpecialWireSegment::get_point(uint32_t index)
{
  return _point_list.size() > index ? _point_list.at(index) : nullptr;
}

IdbCoordinate<int32_t>* IdbSpecialWireSegment::get_point_start()
{
  if (_point_list.size() > 0) {
    return get_point(_POINT_START_);
  }

  return nullptr;
}

IdbCoordinate<int32_t>* IdbSpecialWireSegment::get_point_second()
{
  if (_point_list.size() > 1) {
    return get_point(_POINT_SECOND_);
  }

  return nullptr;
}

IdbCoordinate<int32_t>* IdbSpecialWireSegment::add_point(int32_t x, int32_t y)
{
  IdbCoordinate<int32_t>* point = new IdbCoordinate<int32_t>(x, y);
  _point_list.emplace_back(point);

  return point;
}

void IdbSpecialWireSegment::set_delta_rect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y)
{
  _delta_rect = new IdbRect(ll_x, ll_y, ur_x, ur_y);
}

IdbVia* IdbSpecialWireSegment::copy_via(IdbVia* via)
{
  if (via != nullptr) {
    if (_via != nullptr) {
      delete _via;
    }

    _via = via->clone();

    return _via;
  }

  return nullptr;
}

/// only copy segment information of stripe, not including via, point list
IdbSpecialWireSegment* IdbSpecialWireSegment::copy()
{
  if (is_via()) {
    return nullptr;
  }

  IdbSpecialWireSegment* segment_new = new IdbSpecialWireSegment();
  segment_new->_layer = _layer;
  segment_new->_route_width = _route_width;
  segment_new->_style = _style;
  segment_new->_shape_type = _shape_type;
  segment_new->_is_new_layer = true;

  return segment_new;
}

bool IdbSpecialWireSegment::set_bounding_box()
{
  if (is_via() && _via != nullptr) {
    IdbObject::set_bounding_box(_via->get_cut_bounding_box());
    return true;
  } else {
    if (_point_list.size() >= 2) {
      // ensure there are 2 point in a segment
      //   IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(_layer);
      int32_t routing_width = _route_width == 0 ? _route_width : _route_width;

      IdbCoordinate<int32_t>* point_1 = get_point_start();
      IdbCoordinate<int32_t>* point_2 = get_point_second();

      int32_t ll_x = 0;
      int32_t ll_y = 0;
      int32_t ur_x = 0;
      int32_t ur_y = 0;
      int nn = 0;
      if (point_1->get_y() == point_2->get_y()) {
        // Horizontal
        ll_x = std::min(point_1->get_x(), point_2->get_x());
        ll_y = point_1->get_y() - routing_width / 2;
        ur_x = std::max(point_1->get_x(), point_2->get_x());
        ur_y = ll_y + routing_width;
        nn++;

      } else {
        // Vertical
        ll_x = point_1->get_x() - routing_width / 2;
        ll_y = std::min(point_1->get_y(), point_2->get_y());
        ur_x = ll_x + routing_width;
        ur_y = std::max(point_1->get_y(), point_2->get_y());
      }

      return IdbObject::set_bounding_box(ll_x, ll_y, ur_x, ur_y);
    } else {
      return IdbObject::set_bounding_box(0, 0, 0, 0);
    }
  }
}

void IdbSpecialWireSegment::adjustStripe(IdbCoordinate<int32_t>* start, IdbCoordinate<int32_t>* end)
{
  IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(_layer);
  IdbRect* bouding_box = get_bounding_box();
  if (routing_layer->is_horizontal()) {
    // horizontal
    start->set_y(bouding_box->get_middle_point_y());
    end->set_y(bouding_box->get_middle_point_y());
  } else {
    /// vertical
    start->set_x(bouding_box->get_middle_point_x());
    end->set_x(bouding_box->get_middle_point_x());
  }
}

bool IdbSpecialWireSegment::containLine(IdbCoordinate<int32_t>* start, IdbCoordinate<int32_t>* end)
{
  if (start == nullptr || end == nullptr) {
    return false;
  }

  if (start->get_x() != end->get_x() && start->get_y() != end->get_y()) {
    return false;
  }

  IdbRect* bouding_box = get_bounding_box();
  if (bouding_box == nullptr) {
    return false;
  }

  if (bouding_box->containPoint(start) || bouding_box->containPoint(end)) {
    /// points inside the rect
    return true;
  } else {
    if (start->get_y() == end->get_y()) {
      /// horizontal
      if (start->get_y() < bouding_box->get_low_y() || start->get_y() > bouding_box->get_high_y()) {
        return false;
      }

      int32_t start_x = std::min(start->get_x(), end->get_x());
      int32_t end_x = std::max(start->get_x(), end->get_x());
      //   return (start_x < bouding_box->get_low_x() && end_x > bouding_box->get_high_x()) ? true : false;
      if (start_x < bouding_box->get_low_x() && end_x > bouding_box->get_high_x())
        return true;
      else
        return false;
    } else {
      /// vertical
      if (start->get_x() < bouding_box->get_low_x() || start->get_x() > bouding_box->get_high_x()) {
        return false;
      }

      int32_t start_y = std::min(start->get_y(), end->get_y());
      int32_t end_y = std::max(start->get_y(), end->get_y());
      if (start_y < bouding_box->get_low_y() && end_y > bouding_box->get_high_y())
        return true;
      else
        return false;
      //   return (start_y < bouding_box->get_low_y() && end_y > bouding_box->get_high_y()) ? true : false;
    }
  }

  return false;
}

bool IdbSpecialWireSegment::get_intersect_coordinate(IdbSpecialWireSegment* segment, IdbCoordinate<int32_t>& intersect_coordinate)
{
  if (!(segment->is_line() && is_line())) {
    return false;
  }

  /// if intersect
  IdbCoordinate<int32_t>* start = segment->get_point_start();
  IdbCoordinate<int32_t>* end = segment->get_point_second();
  if (containLine(start, end)) {
    /// segment compare is horizontal
    if (start->get_y() == end->get_y()) {
      intersect_coordinate.set_x(this->get_point_start()->get_x());
      intersect_coordinate.set_y(start->get_y());
    } else {
      /// segment compare is vertical
      intersect_coordinate.set_x(start->get_x());
      intersect_coordinate.set_y(this->get_point_start()->get_y());
    }

    return true;
  }

  return false;
}

IdbCoordinate<int32_t>* IdbSpecialWireSegment::get_point_left()
{
  int32_t left_pt = INT32_MAX;
  IdbCoordinate<int32_t>* point = nullptr;
  for (IdbCoordinate<int32_t>* coordinate : _point_list) {
    if (coordinate->get_x() < left_pt) {
      left_pt = coordinate->get_x();
      point = coordinate;
    }
  }

  return point;
}

IdbCoordinate<int32_t>* IdbSpecialWireSegment::get_point_right()
{
  int32_t right_pt = 0;
  IdbCoordinate<int32_t>* point = nullptr;
  for (IdbCoordinate<int32_t>* coordinate : _point_list) {
    if (coordinate->get_x() > right_pt) {
      right_pt = coordinate->get_x();
      point = coordinate;
    }
  }

  return point;
}

IdbCoordinate<int32_t>* IdbSpecialWireSegment::get_point_bottom()
{
  int32_t bottom_pt = INT32_MAX;
  IdbCoordinate<int32_t>* point = nullptr;
  for (IdbCoordinate<int32_t>* coordinate : _point_list) {
    if (coordinate->get_y() < bottom_pt) {
      bottom_pt = coordinate->get_y();
      point = coordinate;
    }
  }

  return point;
}

IdbCoordinate<int32_t>* IdbSpecialWireSegment::get_point_top()
{
  int32_t top_pt = 0;
  IdbCoordinate<int32_t>* point = nullptr;
  for (IdbCoordinate<int32_t>* coordinate : _point_list) {
    if (coordinate->get_y() > top_pt) {
      top_pt = coordinate->get_y();
      point = coordinate;
    }
  }

  return point;
}

bool IdbSpecialWireSegment::is_horizontal()
{
  if (!is_tripe() && !is_follow_pin()) {
    return false;
  } else {
    IdbCoordinate<int32_t>* point_start = get_point_start();
    IdbCoordinate<int32_t>* point_end = get_point_second();
    if (point_start->get_y() == point_end->get_y()) {
      return true;
    }
  }
  return false;
}

bool IdbSpecialWireSegment::is_vertical()
{
  if (!is_tripe() && !is_follow_pin()) {
    return false;
  } else {
    IdbCoordinate<int32_t>* point_start = get_point_start();
    IdbCoordinate<int32_t>* point_end = get_point_second();
    if (point_start->get_x() == point_end->get_x()) {
      return true;
    }
  }
  return false;
}

int32_t IdbSpecialWireSegment::length()
{
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

  if (is_via()) {
    /// thickness
    return 0;
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbSpecialWire::IdbSpecialWire()
{
  _wire_state = IdbWiringStatement::kNone;
  _num = 0;
}

IdbSpecialWire::~IdbSpecialWire()
{
  for (auto& segment : _segment_list) {
    if (nullptr != segment) {
      delete segment;
      segment = nullptr;
    }
  }

  _segment_list.clear();
}

uint IdbSpecialWire::get_via_num()
{
  uint number = 0;
  for (auto segment : _segment_list) {
    if (segment->is_via()) {
      number++;
    }
  }

  return number;
}

void IdbSpecialWire::set_wire_state(string state)
{
  set_wire_state(IdbEnum::GetInstance()->get_connect_property()->get_wiring_state(state));

  //----------------------tbd---------------------
  // wireShield not support
}

IdbSpecialWireSegment* IdbSpecialWire::add_segment(IdbSpecialWireSegment* segment)
{
  IdbSpecialWireSegment* pSegment = segment;
  if (pSegment == nullptr) {
    pSegment = new IdbSpecialWireSegment();
  }

  if (_segment_list.size() > 0) {
    pSegment->set_layer_as_new();
  }

  _segment_list.emplace_back(pSegment);
  ++_num;

  return pSegment;
}

/// add a segment connect to the segment_connect
IdbSpecialWireSegment* IdbSpecialWire::add_segment_stripe(IdbCoordinate<int32_t>* start, IdbCoordinate<int32_t>* end,
                                                          IdbSpecialWireSegment* segment_connected, int32_t width)
{
  if (segment_connected == nullptr) {
    return nullptr;
  }
  /// direction must be horizontal or vertical, otherwise value is illegal
  if (start->get_x() != end->get_x() && start->get_y() != end->get_y()) {
    return nullptr;
  }

  IdbSpecialWireSegment* segment_new = segment_connected->copy();
  if (width >= 0) {
    segment_new->set_route_width(width);
  }

  if (start->get_x() > end->get_x() || start->get_y() > end->get_y()) {
    segment_new->add_point(end->get_x(), end->get_y());
    segment_new->add_point(start->get_x(), start->get_y());
    segment_new->set_bounding_box();
  } else {
    segment_new->add_point(start->get_x(), start->get_y());
    segment_new->add_point(end->get_x(), end->get_y());
    segment_new->set_bounding_box();
  }

  return add_segment(segment_new);
}

// IdbSpecialWireSegment* IdbSpecialWire::addSegmentVia(int32_t coord_x, int32_t coord_y, IdbVia* via) {
//   IdbSpecialWireSegment* segment = add_segment();

//   IdbCoordinate<int32_t>* coordinate = new IdbCoordinate<int32_t>(coord_x, coord_y);
//   IdbVia* via_new                    = segment->copy_via(via);
//   if (via_new != nullptr) {
//     via_new->set_coordinate(coordinate);
//   }

//   segment->set_is_via(true);

//   return segment;
// }

/// point_list : should be more than 2 points
/// segment_connected: segment that connect to the new segment generated
int32_t IdbSpecialWire::add_segment_list(vector<IdbCoordinate<int32_t>*>& point_list, IdbSpecialWireSegment* segment_connected,
                                         int32_t width)
{
  if (segment_connected == nullptr || point_list.size() < _POINT_MAX_) {
    return -1;
  }

  int32_t number = 0;
  int32_t seg_width = width == -1 ? segment_connected->get_route_width() : width;
  /// add stripe segment
  for (size_t i = 0; i < (point_list.size() - 1); ++i) {
    IdbCoordinate<int32_t>* coordinate_1 = new IdbCoordinate<int32_t>(point_list[i]->get_x(), point_list[i]->get_y());
    IdbCoordinate<int32_t>* coordinate_2 = new IdbCoordinate<int32_t>(point_list[i + 1]->get_x(), point_list[i + 1]->get_y());
    /// adjust the coordinate between start and end
    if (point_list.size() > 2) {
      if (i == 0) {
        adjustCoordinate(coordinate_1, coordinate_2, seg_width, CoordinatePosition::kEnd);
      } else if (i == point_list.size() - 2) {
        adjustCoordinate(coordinate_1, coordinate_2, seg_width, CoordinatePosition::kStart);
      } else {
        adjustCoordinate(coordinate_1, coordinate_2, seg_width, CoordinatePosition::kBoth);
      }
    }

    /// add segment
    if (seg_width == segment_connected->get_route_width()) {
      add_segment_stripe(coordinate_1, coordinate_2, segment_connected);
    } else if (seg_width == width) {
      add_segment_stripe(coordinate_1, coordinate_2, segment_connected, seg_width);
    }
    number++;

    delete coordinate_1;
    delete coordinate_2;
  }

  return number;
}

void IdbSpecialWire::removeViaInBoundingBox(IdbRect rect, IdbLayer* layer)
{
  int i = 0;
  for (auto segment = _segment_list.begin(); segment != _segment_list.end();) {
    if ((*segment)->is_via() && (*segment)->get_via()->isIntersection(rect, layer)) {
      segment = _segment_list.erase(std::begin(_segment_list) + i);
      std::cout << "Success : remove via = " << (*segment)->get_via()->get_name() << std::endl;
    } else {
      ++segment;
      ++i;
    }
  }
}

void adjustCoordinate(IdbCoordinate<int32_t>* start, IdbCoordinate<int32_t>* end, int32_t width, CoordinatePosition coor_post)
{
  switch (coor_post) {
      /// adjust start
    case CoordinatePosition::kStart: {
      if (start->get_y() == end->get_y()) {
        /// hotizontal
        int32_t new_x = start->get_x() > end->get_x() ? (start->get_x() + width / 2) : (start->get_x() - width / 2);
        start->set_x(new_x);
      } else if (start->get_x() == end->get_x()) {
        /// vertical
        int32_t new_y = start->get_y() > end->get_y() ? (start->get_y() + width / 2) : (start->get_y() - width / 2);
        start->set_y(new_y);
      } else {
        /// do nothing
      }

      break;
    }
      /// adjust end
    case CoordinatePosition::kEnd: {
      if (start->get_y() == end->get_y()) {
        /// hotizontal
        int32_t new_x = end->get_x() > start->get_x() ? (end->get_x() + width / 2) : (end->get_x() - width / 2);
        end->set_x(new_x);
      } else if (start->get_x() == end->get_x()) {
        /// vertical
        int32_t new_y = end->get_y() > start->get_y() ? (end->get_y() + width / 2) : (end->get_y() - width / 2);
        end->set_y(new_y);
      } else {
        /// do nothing
      }
      break;
    }
      /// adjust both
    case CoordinatePosition::kBoth: {
      if (start->get_y() == end->get_y()) {
        /// hotizontal
        if (start->get_x() > end->get_x()) {
          start->set_x(start->get_x() + width / 2);
          end->set_x(end->get_x() - width / 2);
        } else {
          start->set_x(start->get_x() - width / 2);
          end->set_x(end->get_x() + width / 2);
        }
      } else if (start->get_x() == end->get_x()) {
        /// vertical
        if (start->get_y() > end->get_y()) {
          start->set_y(start->get_y() + width / 2);
          end->set_y(end->get_y() - width / 2);
        } else {
          start->set_y(start->get_y() - width / 2);
          end->set_y(end->get_y() + width / 2);
        }
      } else {
        /// do nothing
      }
      break;
    }
    default:
      // do nothing
      break;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbSpecialWireList::IdbSpecialWireList()
{
  _num = 0;
}

IdbSpecialWireList::~IdbSpecialWireList()
{
  for (auto& wire : _wire_list) {
    if (nullptr != wire) {
      delete wire;
      wire = nullptr;
    }
  }

  _wire_list.clear();
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

IdbSpecialWire* IdbSpecialWireList::add_wire(IdbSpecialWire* wire, IdbWiringStatement state)
{
  IdbSpecialWire* pWire = wire;
  if (pWire == nullptr) {
    pWire = new IdbSpecialWire();
    pWire->set_wire_state(state);
  }
  _wire_list.emplace_back(pWire);
  ++_num;

  return pWire;
}

void IdbSpecialWireList::reset()
{
  for (auto& wire : _wire_list) {
    if (wire != nullptr) {
      delete wire;
      wire = nullptr;
    }
  }
  _wire_list.clear();
  _num = 0;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IdbSpecialWireSegment* IdbSpecialNetEdgeSegment::isStripeCrossLine(IdbCoordinate<int32_t>* start, IdbCoordinate<int32_t>* end)
{
  for (IdbSpecialWireSegment* segment : _segment_list) {
    if (segment->containLine(start, end)) {
      return segment;
    }
  }

  return nullptr;
}

/// @return return the new segment create by cutting stripe
IdbSpecialWireSegment* IdbSpecialNetEdgeSegment::cutStripe(IdbSpecialNetEdgeSegment* edge_segment_connected, IdbCoordinate<int32_t>* start,
                                                           IdbCoordinate<int32_t>* end)
{
  ///
  std::cout << "Segment == " << std::endl;
  for (IdbSpecialWireSegment* st : _segment_list) {
    std::cout << " ( " << st->get_point(0)->get_x() << " , " << st->get_point(0)->get_y() << " ) , ( " << st->get_point(1)->get_x() << " , "
              << st->get_point(1)->get_y() << " ) " << std::endl;
  }

  std::cout << "_coordinate_x_y = " << _coordinate_x_y << " edge_segment_connected = " << edge_segment_connected->get_coordinate()
            << std::endl;

  /// exclude the connected stripe
  if (edge_segment_connected->get_coordinate() != _coordinate_x_y) {
    /// judge if overlap
    IdbSpecialWireSegment* segment_cut = isStripeCrossLine(start, end);
    if (segment_cut != nullptr) {
      IdbLayerRouting* layer_routing = dynamic_cast<IdbLayerRouting*>(segment_cut->get_layer());
      int32_t width = segment_cut->get_route_width();
      int32_t space = layer_routing->get_spacing_list()->get_spacing(width);

      if (start->get_y() == end->get_y()) {
        /// horizontal
        /// get original coordinate
        IdbCoordinate<int32_t>* top_point = segment_cut->get_point_top();
        // IdbCoordinate<int32_t>* bottom_point = segment_cut->get_point_bottom();

        /// calculate new y coordinate
        int32_t base_y = end->get_y();
        int32_t top_y = base_y + width + space;
        int32_t bottom_y = base_y - width - space;

        /// add to wire
        IdbCoordinate<int32_t>* coordinate_1 = new IdbCoordinate<int32_t>(top_point->get_x(), top_y);
        IdbCoordinate<int32_t>* coordinate_2 = new IdbCoordinate<int32_t>(top_point->get_x(), top_point->get_y());
        IdbSpecialWireSegment* new_segment = _wire->add_segment_stripe(coordinate_1, coordinate_2, segment_cut);
        delete coordinate_1;
        coordinate_1 = nullptr;
        delete coordinate_2;
        coordinate_2 = nullptr;
        // add to edge stripe
        _segment_list.emplace_back(new_segment);

        /// set original segment as below segment, so reset the top point
        top_point->set_y(bottom_y);
        /// update bounding box
        segment_cut->set_bounding_box();

        /// remove via which is cut by the stripe
        IdbRect rect_cut(top_point->get_x() - width / 2, bottom_y, top_point->get_x() + width / 2, top_y);
        _wire->removeViaInBoundingBox(rect_cut, segment_cut->get_layer());

        std::cout << "Success : cutStripe. " << std::endl;
        ///
        std::cout << "Segment == " << std::endl;
        for (IdbSpecialWireSegment* st : _segment_list) {
          std::cout << " ( " << st->get_point(0)->get_x() << " , " << st->get_point(0)->get_y() << " ) , ( " << st->get_point(1)->get_x()
                    << " , " << st->get_point(1)->get_y() << " ) " << std::endl;
        }

        return new_segment;

      } else {
        /// vertical cut
        /// get original coordinate
        // IdbCoordinate<int32_t>* left_point  = segment_cut->get_point_left();
        IdbCoordinate<int32_t>* right_point = segment_cut->get_point_right();

        /// calculate new y coordinate
        int32_t base_x = end->get_x();
        int32_t right_x = base_x + width + space;
        int32_t left_x = base_x - width - space;

        /// add to wire
        IdbCoordinate<int32_t>* coordinate_1 = new IdbCoordinate<int32_t>(right_x, right_point->get_y());
        IdbCoordinate<int32_t>* coordinate_2 = new IdbCoordinate<int32_t>(right_point->get_x(), right_point->get_y());
        IdbSpecialWireSegment* new_segment = _wire->add_segment_stripe(coordinate_1, coordinate_2, segment_cut);
        delete coordinate_1;
        coordinate_1 = nullptr;
        delete coordinate_2;
        coordinate_2 = nullptr;

        // add to edge stripe
        _segment_list.emplace_back(new_segment);

        /// set original segment as left segment, so reset the right point
        /// update bounding box
        right_point->set_x(left_x);
        segment_cut->set_bounding_box();

        /// remove via which is cut by the stripe
        IdbRect rect_cut(left_x, right_point->get_y() - width / 2, right_x, right_point->get_y() + width / 2);
        _wire->removeViaInBoundingBox(rect_cut, segment_cut->get_layer());

        std::cout << "Success : cutStripe. " << std::endl;
        ///
        std::cout << "Segment == " << std::endl;
        for (IdbSpecialWireSegment* st : _segment_list) {
          std::cout << " ( " << st->get_point(0)->get_x() << " , " << st->get_point(0)->get_y() << " ) , ( " << st->get_point(1)->get_x()
                    << " , " << st->get_point(1)->get_y() << " ) " << std::endl;
        }

        return new_segment;
      }
    } else {
      std::cout << "CutStripe : no intersection. " << std::endl;
      return nullptr;
    }
  }
  std::cout << "Same Coordinate. " << std::endl;

  return nullptr;
}

// bool IdbSpecialNetEdgeSegment::cutStripe(IdbCoordinate<int32_t>* start, IdbCoordinate<int32_t>* end) {
//   /// judge if overlap
//   IdbSpecialWireSegment* segment_cut = isStripeCrossLine(start, end);
//   if (segment_cut != nullptr) {
//     IdbLayerRouting* layer_routing = dynamic_cast<IdbLayerRouting*>(segment_cut->get_layer());
//     int32_t width                  = segment_cut->get_route_width();
//     int32_t space                  = layer_routing->get_spacing_list()->get_spacing(width);

//     if (start->get_y() == end->get_y()) {
//       /// horizontal
//       /// get original coordinate
//       IdbCoordinate<int32_t>* top_point = segment_cut->get_point_top();
//       // IdbCoordinate<int32_t>* bottom_point = segment_cut->get_point_bottom();

//       /// calculate new y coordinate
//       int32_t base_y   = end->get_y();
//       int32_t top_y    = base_y + width + space;
//       int32_t bottom_y = base_y - width - space;

//       /// copy a new segment as above
//       IdbCoordinate<int32_t>* coordinate_1 = new IdbCoordinate<int32_t>(top_point->get_x(), top_y);
//       IdbCoordinate<int32_t>* coordinate_2 = new IdbCoordinate<int32_t>(top_point->get_x(), top_point->get_y());
//       IdbSpecialWireSegment* new_segment   = _wire->add_segment_stripe(coordinate_1, coordinate_2, segment_cut);
//       // add to edge stripe
//       _segment_list.emplace_back(new_segment);

//       /// set original segment as below segment, so reset the top point
//       top_point->set_y(bottom_y);
//       segment_cut->set_bounding_box();

//       std::cout << "Success : cutStripe by ( " << start->get_x() << " , " << start->get_y() << " ) , ( " <<
//       end->get_x()
//                 << " , " << end->get_y() << " ) " << std::endl;

//       return true;

//     } else {
//       /// vertical cut
//       /// get original coordinate
//       // IdbCoordinate<int32_t>* left_point  = segment_cut->get_point_left();
//       IdbCoordinate<int32_t>* right_point = segment_cut->get_point_right();

//       /// calculate new y coordinate
//       int32_t base_x  = end->get_x();
//       int32_t right_x = base_x + width + space;
//       int32_t left_x  = base_x - width - space;

//       /// copy a new segment as right
//       IdbCoordinate<int32_t>* coordinate_1 = new IdbCoordinate<int32_t>(right_x, right_point->get_y());
//       IdbCoordinate<int32_t>* coordinate_2 = new IdbCoordinate<int32_t>(right_point->get_x(), right_point->get_y());
//       IdbSpecialWireSegment* new_segment   = _wire->add_segment_stripe(coordinate_1, coordinate_2, segment_cut);

//       // add to edge stripe
//       _segment_list.emplace_back(new_segment);

//       /// set original segment as left segment, so reset the right point
//       right_point->set_x(left_x);
//       segment_cut->set_bounding_box();

//       std::cout << "Success : cutStripe by ( " << start->get_x() << " , " << start->get_y() << " ) , ( " <<
//       end->get_x()
//                 << " , " << end->get_y() << " ) " << std::endl;

//       return true;
//     }
//   }
// }

void IdbSpecialNetEdgeSegment::add_segment_list_by_coordinate(vector<IdbCoordinate<int32_t>*>& point_list)
{
  /// add segment
  if (_segment_list.size() <= 0 || _wire == nullptr) {
    return;
  }

  IdbSpecialWireSegment* wire_segment = _segment_list[0];
  _wire->add_segment_list(point_list, wire_segment);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void IdbSpecialNetEdgeSegmenArray::updateSegment(IdbSpecialWireSegment* segment, IdbSpecialWire* wire, SegmentType type)
{
  if (segment->get_point_list().size() < _POINT_MAX_ || !(segment->is_tripe() || segment->is_follow_pin()))
    return;

  updateSegmentEdgePoints(segment, wire, type);
  updateSegmentArray(segment, wire, type);
}

void IdbSpecialNetEdgeSegmenArray::updateSegmentEdgePoints(IdbSpecialWireSegment* segment, IdbSpecialWire* wire, SegmentType type)
{
  int32_t coordinate_value = -1;
  /// /// horizontal
  if (_layer->is_horizontal()) {
    if (segment->get_point_start()->get_y() == segment->get_point_second()->get_y()) {
      coordinate_value = segment->get_point_start()->get_y();
    }
  } else {
    /// vertical
    if (segment->get_point_start()->get_x() == segment->get_point_second()->get_x()) {
      coordinate_value = segment->get_point_start()->get_x();
    }
  }

  if (coordinate_value == -1) {
    return;
  }

  if (type == SegmentType::kVDD) {
    // vdd_1
    if (_segment_vdd_1->get_coordinate() > coordinate_value || _segment_vdd_1->get_coordinate() == -1) {
      _segment_vdd_1->set_coordinate(coordinate_value);
      _segment_vdd_1->reset_segment_list(segment);
      _segment_vdd_1->set_wire(wire);
    } else if (_segment_vdd_1->get_coordinate() == coordinate_value) {
      _segment_vdd_1->add_segment(segment);
    } else {
      /// do nothing
    }
    // vdd_2
    if (_segment_vdd_2->get_coordinate() < coordinate_value || _segment_vdd_2->get_coordinate() == -1) {
      _segment_vdd_2->set_coordinate(coordinate_value);
      _segment_vdd_2->reset_segment_list(segment);
      _segment_vdd_2->set_wire(wire);

    } else if (_segment_vdd_2->get_coordinate() == coordinate_value) {
      _segment_vdd_2->add_segment(segment);
    } else {
      /// do nothing
    }

  } else if (type == SegmentType::kVSS) {
    // vdd_1
    if (_segment_vss_1->get_coordinate() > coordinate_value || _segment_vss_1->get_coordinate() == -1) {
      _segment_vss_1->set_coordinate(coordinate_value);
      _segment_vss_1->reset_segment_list(segment);
      _segment_vss_1->set_wire(wire);
    } else if (_segment_vss_1->get_coordinate() == coordinate_value) {
      _segment_vss_1->add_segment(segment);
    } else {
      /// do nothing
    }
    // vdd_2
    if (_segment_vss_2->get_coordinate() < coordinate_value || _segment_vss_2->get_coordinate() == -1) {
      _segment_vss_2->set_coordinate(coordinate_value);
      _segment_vss_2->reset_segment_list(segment);
      _segment_vss_2->set_wire(wire);

    } else if (_segment_vss_2->get_coordinate() == coordinate_value) {
      _segment_vss_2->add_segment(segment);
    } else {
      /// do nothing
    }
  } else {
    /// error
    std::cout << "Error : segment type error." << std::endl;
  }
}

void IdbSpecialNetEdgeSegmenArray::updateSegmentArray(IdbSpecialWireSegment* segment, IdbSpecialWire* wire, SegmentType type)
{
  int32_t coordinate_value = -1;
  /// /// horizontal
  if (_layer->is_horizontal()) {
    if (segment->get_point_start()->get_y() == segment->get_point_second()->get_y()) {
      coordinate_value = segment->get_point_start()->get_y();
    }
  } else {
    /// vertical
    if (segment->get_point_start()->get_x() == segment->get_point_second()->get_x()) {
      coordinate_value = segment->get_point_start()->get_x();
    }
  }

  if (coordinate_value == -1) {
    return;
  }

  if (type == SegmentType::kVDD) {
    IdbSpecialNetEdgeSegment* vdd_new = add_vdd();
    vdd_new->set_coordinate(coordinate_value);
    vdd_new->reset_segment_list(segment);
    vdd_new->set_wire(wire);
  } else if (type == SegmentType::kVSS) {
    IdbSpecialNetEdgeSegment* vdd_new = add_vss();
    vdd_new->set_coordinate(coordinate_value);
    vdd_new->reset_segment_list(segment);
    vdd_new->set_wire(wire);
  } else {
    /// error
    std::cout << "Error : segment type error." << std::endl;
  }
}

/// find segment in all segment list
IdbSpecialNetEdgeSegment* IdbSpecialNetEdgeSegmenArray::findSegmentByCoordinate(IdbCoordinate<int32_t>* coordinate)
{
  /// find in vdd list
  for (IdbSpecialNetEdgeSegment* edge_segment : _vdd_list) {
    vector<IdbSpecialWireSegment*>& wire_segment_list = edge_segment->get_segment_list();
    if (wire_segment_list.size() <= 0) {
      continue;
    }

    for (IdbSpecialWireSegment* wire_segment : wire_segment_list) {
      if (wire_segment->get_bounding_box()->containPoint(coordinate)) {
        return edge_segment;
      }
    }
  }
  /// find in vss list
  for (IdbSpecialNetEdgeSegment* edge_segment : _vss_list) {
    vector<IdbSpecialWireSegment*>& wire_segment_list = edge_segment->get_segment_list();
    if (wire_segment_list.size() <= 0) {
      continue;
    }

    for (IdbSpecialWireSegment* wire_segment : wire_segment_list) {
      if (wire_segment->get_bounding_box()->containPoint(coordinate)) {
        return edge_segment;
      }
    }
  }

  return nullptr;
}

////find edge segment
IdbSpecialNetEdgeSegment* IdbSpecialNetEdgeSegmenArray::find_segment_edge_by_coordinate(int32_t coordinate_x_y)
{
  if (_segment_vdd_1->get_coordinate() == coordinate_x_y) {
    return _segment_vdd_1;
  }

  if (_segment_vdd_2->get_coordinate() == coordinate_x_y) {
    return _segment_vdd_2;
  }

  if (_segment_vss_1->get_coordinate() == coordinate_x_y) {
    return _segment_vss_1;
  }

  if (_segment_vss_2->get_coordinate() == coordinate_x_y) {
    return _segment_vss_2;
  }

  return nullptr;
}

bool IdbSpecialNetEdgeSegmenArray::addSegmentByCoordinateList(vector<IdbCoordinate<int32_t>*>& coordinate_list)
{
  int32_t point_size = coordinate_list.size();
  if (point_size < _POINT_MAX_) {
    std::cout << "Error : size of point list should be larger than 2 to connect stripe." << std::endl;
    return false;
  }

  IdbCoordinate<int32_t>* point_start = coordinate_list[0];
  IdbSpecialNetEdgeSegment* edge_segment = findSegmentByCoordinate(point_start);
  if (edge_segment != nullptr) {
    if (hasSameOrient(coordinate_list[0], coordinate_list[1])) {
      /// adjust coordinate by pin
      IdbSpecialWireSegment* wire_segment = edge_segment->get_segment_list()[0];
      wire_segment->adjustStripe(coordinate_list[0], coordinate_list[1]);
    } else {
      /// cross the stripe
      //   for(IdbCoordinate<int32_t>* coordinate :coordinate_list)
      cutStripe(edge_segment, coordinate_list[0], coordinate_list[1]);
    }

    edge_segment->add_segment_list_by_coordinate(coordinate_list);

    return true;
  }

  IdbCoordinate<int32_t>* point_end = coordinate_list[point_size - 1];
  edge_segment = findSegmentByCoordinate(point_end);
  if (edge_segment != nullptr) {
    if (hasSameOrient(coordinate_list[point_size - 2], coordinate_list[point_size - 1])) {
      /// adjust coordinate by pin
      IdbSpecialWireSegment* wire_segment = edge_segment->get_segment_list()[0];
      wire_segment->adjustStripe(coordinate_list[point_size - 2], coordinate_list[point_size - 1]);
    } else {
      /// cross the stripe
      cutStripe(edge_segment, coordinate_list[point_size - 2], coordinate_list[point_size - 1]);
    }

    edge_segment->add_segment_list_by_coordinate(coordinate_list);

    return true;
  }

  std::cout << "Error : can not addSegmentByCoordinateList." << std::endl;
  for (IdbCoordinate<int32_t>* point_print : coordinate_list) {
    std::cout << "coordinate." << point_print->get_x() << "," << point_print->get_y() << std::endl;
  }

  return false;
}

/// judge if the direction of 2 points has the same direction of prefer layer direction
bool IdbSpecialNetEdgeSegmenArray::hasSameOrient(IdbCoordinate<int32_t>* start, IdbCoordinate<int32_t>* end)
{
  /// horizontal or vertical
  return ((start->get_y() == end->get_y() && _layer->is_horizontal()) || (start->get_x() == end->get_x() && _layer->is_vertical())) ? true
                                                                                                                                    : false;
}

void IdbSpecialNetEdgeSegmenArray::cutStripe(IdbSpecialNetEdgeSegment* edge_segment_connected, IdbCoordinate<int32_t>* start,
                                             IdbCoordinate<int32_t>* end)
{
  //   IdbSpecialWireSegment* segment_new = _segment_vdd_1->cutStripe(edge_segment_connected, start, end);
  //   if (nullptr != _segment_vdd_1->cutStripe(edge_segment_connected, start, end) ||
  //       nullptr != _segment_vdd_2->cutStripe(edge_segment_connected, start, end) ||
  //       nullptr != _segment_vss_1->cutStripe(edge_segment_connected, start, end) ||
  //       nullptr != _segment_vss_2->cutStripe(edge_segment_connected, start, end)) {
  //   } else {
  //     // std::cout << "Info : do not cut stripe. ( " << start->get_x() << " , " << start->get_y() << " ) ( " <<
  //     // end->get_x()
  //     //           << " , " << end->get_y() << " )" << std::endl;
  //   }
  //   edge_segment_connected->cutStripe(start, end);
  std::cout << "####################################################################" << std::endl;
  std::cout << "Points  == ";
  std::cout << " ( " << start->get_x() << " , " << start->get_y() << " ) , ( " << end->get_x() << " , " << end->get_y() << " ) "
            << std::endl;
  IdbSpecialWireSegment* segment = nullptr;
  std::cout << " VDD 1 ";
  if ((segment = _segment_vdd_1->cutStripe(edge_segment_connected, start, end)) != nullptr) {
    _segment_vdd_1->add_segment(segment);
    std::cout << "success : VDD 1" << std::endl;
  }
  std::cout << " VDD 2 ";
  if ((segment = _segment_vdd_2->cutStripe(edge_segment_connected, start, end)) != nullptr) {
    _segment_vdd_2->add_segment(segment);
    std::cout << "success : VDD 2" << std::endl;
  }
  std::cout << " VSS 1 ";
  if ((segment = _segment_vss_1->cutStripe(edge_segment_connected, start, end)) != nullptr) {
    _segment_vss_1->add_segment(segment);
    std::cout << "success : VSS 1" << std::endl;
  }
  std::cout << " VSS 2 ";
  if ((segment = _segment_vss_2->cutStripe(edge_segment_connected, start, end)) != nullptr) {
    _segment_vss_2->add_segment(segment);
    std::cout << "success : VSS 2" << std::endl;
  }
  std::cout << "####################################################################" << std::endl;
}

}  // namespace idb
