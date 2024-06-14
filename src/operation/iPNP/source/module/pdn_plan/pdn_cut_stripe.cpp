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
#include "pdn_cut_stripe.h"

#include "IdbEnum.h"
#include "IdbSpecialNet.h"
#include "IdbSpecialWire.h"
#include "idm.h"

using namespace idb;
using namespace std;

namespace ipdn {
FPdbSpecialNetEdgeSegment::FPdbSpecialNetEdgeSegment()
{
  _wire = nullptr;
  _type = idb::SegmentType::kNone;
  _coordinate_x_y = -1;
}

FPdbSpecialNetEdgeSegment::FPdbSpecialNetEdgeSegment(idb::SegmentType type)
{
  _wire = nullptr;
  _type = type;
  _coordinate_x_y = -1;
}

FPdbSpecialNetEdgeSegment::~FPdbSpecialNetEdgeSegment()
{
  _segment_list.clear();
  std::vector<idb::IdbSpecialWireSegment*>().swap(_segment_list);

  _wire = nullptr;
}

bool FPdbSpecialNetEdgeSegment::is_vdd()
{
  return _type == idb::SegmentType::kVDD ? true : false;
}

bool FPdbSpecialNetEdgeSegment::is_vss()
{
  return _type == idb::SegmentType::kVSS ? true : false;
}

void FPdbSpecialNetEdgeSegment::set_type_vdd()
{
  _type = idb::SegmentType::kVDD;
}

void FPdbSpecialNetEdgeSegment::set_type_vss()
{
  _type = idb::SegmentType::kVSS;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CutStripe::clear_edge_list()
{
  int size = _edge_list.size();
  if (size > 0) {
    for (FPdbSpecialNetEdgeSegmenArray* edge_segment_array : _edge_list) {
      if (edge_segment_array != nullptr) {
        delete edge_segment_array;
        edge_segment_array = nullptr;
      }
    }
  }

  _edge_list.clear();
  vector<FPdbSpecialNetEdgeSegmenArray*>().swap(_edge_list);
}

void CutStripe::initEdge()
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = dmInst->get_idb_layout();
  auto idb_layer_list = idb_layout->get_layers();
  auto idb_pdn_list = idb_design->get_special_net_list();

  /// celar edge list
  clear_edge_list();
  /// init all layer edge
  for (idb::IdbLayer* layer : idb_layer_list->get_routing_layers()) {
    idb::IdbLayerRouting* routing_layer = dynamic_cast<idb::IdbLayerRouting*>(layer);
    add_edge_segment_array_for_layer(routing_layer);
  }

  /// construct all edge segment of points
  for (idb::IdbSpecialNet* net : idb_pdn_list->get_net_list()) {
    idb::SegmentType type = idb::SegmentType ::kNone;

    if (net->is_vdd()) {
      type = SegmentType::kVDD;
    } else if (net->is_vss()) {
      type = SegmentType::kVSS;
    } else {
      continue;
    }

    for (idb::IdbSpecialWire* wire : net->get_wire_list()->get_wire_list()) {
      for (idb::IdbSpecialWireSegment* segment : wire->get_segment_list()) {
        /// only use the stripe wire to calculate the coordinate
        if (segment->get_point_list().size() < _POINT_MAX_ || !(segment->is_tripe() || segment->is_follow_pin())) {
          continue;
        }

        idb::IdbLayer* layer = segment->get_layer();

        FPdbSpecialNetEdgeSegmenArray* edge_array = find_edge_segment_array_by_layer(layer);

        if (edge_array != nullptr) {
          edge_array->updateSegment(segment, wire, type);
        }
      }
    }
  }

  std::cout << "Init Special Net Edge." << std::endl;
}

bool CutStripe::connectIOPinToPowerStripe(vector<idb::IdbCoordinate<int32_t>*>& point_list, idb::IdbLayer* layer)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = dmInst->get_idb_layout();
  // auto idb_layer_list = idb_layout->get_layers();
  // auto idb_pdn_list = idb_design->get_special_net_list();
  auto idb_io_pin_list = idb_design->get_io_pin_list();
  auto idb_core = idb_layout->get_core();

  if (point_list.size() < _POINT_MAX_ || layer == nullptr) {
    return false;
  }

  /// find the IO pin that covered by the point list
  idb::IdbPin* pin = idb_io_pin_list->find_pin_by_coordinate_list(point_list, layer);
  if (pin == nullptr) {
    std::cout << "Error : no IO pin covered by point list." << std::endl;
    for (idb::IdbCoordinate<int32_t>* pt : point_list) {
      std::cout << " ( " << pt->get_x() << " , " << pt->get_y() << " )";
    }
    std::cout << std::endl;
    return false;
  }

  /// if point list is not horizontal or vertical, gernerate the correct points and adjust the
  /// order
  if (point_list.size() == _POINT_MAX_ && point_list[0]->get_x() != point_list[1]->get_x()
      && point_list[0]->get_y() != point_list[1]->get_y()) {
    /// original value
    int32_t start_x = point_list[0]->get_x();
    int32_t start_y = point_list[0]->get_y();
    int32_t end_x = point_list[1]->get_x();
    int32_t end_y = point_list[1]->get_y();
    /// get the middle coordinate
    int32_t mid_x = (start_x + end_x) / 2;
    int32_t mid_y = (start_y + end_y) / 2;

    if (idb_core->is_side_left_or_right(point_list[0]) || idb_core->is_side_left_or_right(point_list[1])) {
      /// make horizontal
      idb::IdbCoordinate<int32_t>* new_coordinate = new idb::IdbCoordinate<int32_t>(mid_x, start_y);
      point_list.insert(point_list.begin() + 1, new_coordinate);
      new_coordinate = new idb::IdbCoordinate<int32_t>(mid_x, end_y);
      point_list.insert(point_list.begin() + 2, new_coordinate);

    } else if (idb_core->is_side_top_or_bottom(point_list[0]) || idb_core->is_side_top_or_bottom(point_list[1])) {
      /// vertical
      idb::IdbCoordinate<int32_t>* new_coordinate = new idb::IdbCoordinate<int32_t>(start_x, mid_y);
      point_list.insert(point_list.begin() + 1, new_coordinate);
      new_coordinate = new idb::IdbCoordinate<int32_t>(end_x, mid_y);
      point_list.insert(point_list.begin() + 2, new_coordinate);
    } else {
      std::cout << "Error : illegal point list." << std::endl;
      return false;
    }
  }

  return connectIO(point_list, layer);
}

bool CutStripe::connectIO(vector<idb::IdbCoordinate<int32_t>*>& point_list, idb::IdbLayer* layer)
{
  if (point_list.size() < _POINT_MAX_ || layer == nullptr) {
    return false;
  }
  /// find segment array of layer
  FPdbSpecialNetEdgeSegmenArray* layer_segment_array = find_edge_segment_array_by_layer(layer);
  if (layer_segment_array == nullptr) {
    std::cout << "Error : can not find edge." << std::endl;
    return false;
  }

  /// find the segment connected and adjust the point list by segment
  return layer_segment_array->addSegmentByCoordinateList(point_list);
}

/**
 * @brief  The line segment generated when io pad is connected to the power stripe, which belongs
 * to the power special net
 * @param  point_list
 * @param  net_name
 * @param  layer_name
 * @return true
 * @return false
 */
bool CutStripe::addPowerStripe(vector<idb::IdbCoordinate<int32_t>*>& point_list, string net_name, string layer_name, int32_t width)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_pdn_list = idb_design->get_special_net_list();

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

  idb::IdbSpecialWireSegment* segment = wire->get_layer_segment(layer_name);
  if (segment == nullptr) {
    std::cout << "Error : can't find any power stripe in the net." << std::endl;
    return false;
  }

  return wire->add_segment_list(point_list, segment, width) > 0 ? true : false;
}

/**
 * @brief  Find the fpdbspecialnetedgesegmenarray that belongs to this layer
 * @param  layer
 * @return FPdbSpecialNetEdgeSegmenArray*
 */
FPdbSpecialNetEdgeSegmenArray* CutStripe::find_edge_segment_array_by_layer(idb::IdbLayer* layer)
{
  if (layer == nullptr) {
    return nullptr;
  }

  for (FPdbSpecialNetEdgeSegmenArray* segment_array : _edge_list) {
    if (segment_array->get_layer()->compareLayer(layer->get_name())) {
      return segment_array;
    }
  }

  return nullptr;
}

FPdbSpecialNetEdgeSegmenArray* CutStripe::add_edge_segment_array_for_layer(IdbLayerRouting* layer)
{
  FPdbSpecialNetEdgeSegmenArray* pSegment = find_edge_segment_array_by_layer(layer);
  if (pSegment == nullptr) {
    pSegment = new FPdbSpecialNetEdgeSegmenArray();
    pSegment->set_layer(layer);
  }

  _edge_list.emplace_back(pSegment);

  return pSegment;
}
void FPdbSpecialNetEdgeSegment::add_segment_list_by_coordinate(vector<idb::IdbCoordinate<int32_t>*>& point_list)
{
  /// add segment
  if (_segment_list.size() <= 0 || _wire == nullptr) {
    return;
  }

  idb::IdbSpecialWireSegment* wire_segment = _segment_list[0];
  _wire->add_segment_list(point_list, wire_segment);
}

/**
 * @brief
 * @param  start
 * @param  end
 * @return idb::IdbSpecialWireSegment*
 */
idb::IdbSpecialWireSegment* FPdbSpecialNetEdgeSegment::isStripeCrossLine(idb::IdbCoordinate<int32_t>* start,
                                                                         idb::IdbCoordinate<int32_t>* end)
{
  for (idb::IdbSpecialWireSegment* segment : _segment_list) {
    if (segment->containLine(start, end)) {
      return segment;
    }
  }

  return nullptr;
}

idb::IdbSpecialWireSegment* FPdbSpecialNetEdgeSegment::cutStripe(FPdbSpecialNetEdgeSegment* edge_segment_connected,
                                                                 idb::IdbCoordinate<int32_t>* start, idb::IdbCoordinate<int32_t>* end)
{
  ///
  std::cout << "Segment == " << std::endl;
  for (idb::IdbSpecialWireSegment* st : _segment_list) {
    std::cout << " ( " << st->get_point(0)->get_x() << " , " << st->get_point(0)->get_y() << " ) , ( " << st->get_point(1)->get_x() << " , "
              << st->get_point(1)->get_y() << " ) " << std::endl;
  }

  std::cout << "_coordinate_x_y = " << _coordinate_x_y << " edge_segment_connected = " << edge_segment_connected->get_coordinate()
            << std::endl;

  /// exclude the connected stripe
  if (edge_segment_connected->get_coordinate() != _coordinate_x_y) {
    /// judge if overlap
    idb::IdbSpecialWireSegment* segment_cut = isStripeCrossLine(start, end);
    if (segment_cut != nullptr) {
      IdbLayerRouting* layer_routing = dynamic_cast<IdbLayerRouting*>(segment_cut->get_layer());
      int32_t width = segment_cut->get_route_width();
      int32_t space = layer_routing->get_spacing_list()->get_spacing(width);

      if (start->get_y() == end->get_y()) {
        /// horizontal
        /// get original coordinate
        idb::IdbCoordinate<int32_t>* top_point = segment_cut->get_point_top();
        // idb::IdbCoordinate<int32_t>* bottom_point = segment_cut->get_point_bottom();

        /// calculate new y coordinate
        int32_t base_y = end->get_y();
        int32_t top_y = base_y + width + space;
        int32_t bottom_y = base_y - width - space;

        /// add to wire
        idb::IdbCoordinate<int32_t>* coordinate_1 = new idb::IdbCoordinate<int32_t>(top_point->get_x(), top_y);
        idb::IdbCoordinate<int32_t>* coordinate_2 = new idb::IdbCoordinate<int32_t>(top_point->get_x(), top_point->get_y());
        idb::IdbSpecialWireSegment* new_segment = _wire->add_segment_stripe(coordinate_1, coordinate_2, segment_cut);
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
        for (idb::IdbSpecialWireSegment* st : _segment_list) {
          std::cout << " ( " << st->get_point(0)->get_x() << " , " << st->get_point(0)->get_y() << " ) , ( " << st->get_point(1)->get_x()
                    << " , " << st->get_point(1)->get_y() << " ) " << std::endl;
        }

        return new_segment;

      } else {
        /// vertical cut
        /// get original coordinate
        // idb::IdbCoordinate<int32_t>* left_point  = segment_cut->get_point_left();
        idb::IdbCoordinate<int32_t>* right_point = segment_cut->get_point_right();

        /// calculate new y coordinate
        int32_t base_x = end->get_x();
        int32_t right_x = base_x + width + space;
        int32_t left_x = base_x - width - space;

        /// add to wire
        idb::IdbCoordinate<int32_t>* coordinate_1 = new idb::IdbCoordinate<int32_t>(right_x, right_point->get_y());
        idb::IdbCoordinate<int32_t>* coordinate_2 = new idb::IdbCoordinate<int32_t>(right_point->get_x(), right_point->get_y());
        idb::IdbSpecialWireSegment* new_segment = _wire->add_segment_stripe(coordinate_1, coordinate_2, segment_cut);
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
        for (idb::IdbSpecialWireSegment* st : _segment_list) {
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

  // Cut Stripe Success
  std::cout << "Same Coordinate. " << std::endl;

  return nullptr;
}

FPdbSpecialNetEdgeSegmenArray::FPdbSpecialNetEdgeSegmenArray()
{
  _layer = nullptr;
  _segment_vdd_1 = new FPdbSpecialNetEdgeSegment(idb::SegmentType::kVDD);
  _segment_vdd_2 = new FPdbSpecialNetEdgeSegment(idb::SegmentType::kVDD);
  _segment_vss_1 = new FPdbSpecialNetEdgeSegment(idb::SegmentType::kVSS);
  _segment_vss_2 = new FPdbSpecialNetEdgeSegment(idb::SegmentType::kVSS);
}

bool FPdbSpecialNetEdgeSegmenArray::addSegmentByCoordinateList(vector<idb::IdbCoordinate<int32_t>*>& coordinate_list)
{
  int32_t point_size = coordinate_list.size();
  if (point_size < _POINT_MAX_) {
    std::cout << "Error : size of point list should be larger than 2 to connect stripe." << std::endl;
    return false;
  }

  idb::IdbCoordinate<int32_t>* point_start = coordinate_list[0];
  FPdbSpecialNetEdgeSegment* edge_segment = findSegmentByCoordinate(point_start);
  if (edge_segment != nullptr) {
    if (hasSameOrient(coordinate_list[0], coordinate_list[1])) {
      /// adjust coordinate by pin
      idb::IdbSpecialWireSegment* wire_segment = edge_segment->get_segment_list()[0];
      // wire_segment->adjustStripe(coordinate_list[0], coordinate_list[1]);
      adjustStripe(wire_segment, coordinate_list[0], coordinate_list[1]);
    } else {
      /// cross the stripe
      //   for(idb::IdbCoordinate<int32_t>* coordinate :coordinate_list)
      cutStripe(edge_segment, coordinate_list[0], coordinate_list[1]);
    }

    edge_segment->add_segment_list_by_coordinate(coordinate_list);

    return true;
  }

  idb::IdbCoordinate<int32_t>* point_end = coordinate_list[point_size - 1];
  edge_segment = findSegmentByCoordinate(point_end);
  if (edge_segment != nullptr) {
    if (hasSameOrient(coordinate_list[point_size - 2], coordinate_list[point_size - 1])) {
      /// adjust coordinate by pin
      idb::IdbSpecialWireSegment* wire_segment = edge_segment->get_segment_list()[0];
      // wire_segment->adjustStripe(coordinate_list[point_size - 2],
      //                            coordinate_list[point_size - 1]);
      adjustStripe(wire_segment, coordinate_list[point_size - 2], coordinate_list[point_size - 1]);
    } else {
      /// cross the stripe
      cutStripe(edge_segment, coordinate_list[point_size - 2], coordinate_list[point_size - 1]);
    }

    edge_segment->add_segment_list_by_coordinate(coordinate_list);

    return true;
  }

  std::cout << "Error : can not addSegmentByCoordinateList." << std::endl;
  for (idb::IdbCoordinate<int32_t>* point_print : coordinate_list) {
    std::cout << "coordinate." << point_print->get_x() << "," << point_print->get_y() << std::endl;
  }

  return false;
}

void FPdbSpecialNetEdgeSegmenArray::updateSegment(idb::IdbSpecialWireSegment* segment, IdbSpecialWire* wire, SegmentType type)
{
  if (segment->get_point_list().size() < _POINT_MAX_ || !(segment->is_tripe() || segment->is_follow_pin()))
    return;

  updateSegmentEdgePoints(segment, wire, type);
  updateSegmentArray(segment, wire, type);
}

void FPdbSpecialNetEdgeSegmenArray::updateSegmentEdgePoints(idb::IdbSpecialWireSegment* segment, IdbSpecialWire* wire, SegmentType type)
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

void FPdbSpecialNetEdgeSegmenArray::updateSegmentArray(idb::IdbSpecialWireSegment* segment, IdbSpecialWire* wire, SegmentType type)
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
    FPdbSpecialNetEdgeSegment* vdd_new = add_vdd();
    vdd_new->set_coordinate(coordinate_value);
    vdd_new->reset_segment_list(segment);
    vdd_new->set_wire(wire);
  } else if (type == SegmentType::kVSS) {
    FPdbSpecialNetEdgeSegment* vdd_new = add_vss();
    vdd_new->set_coordinate(coordinate_value);
    vdd_new->reset_segment_list(segment);
    vdd_new->set_wire(wire);
  } else {
    /// error
    std::cout << "Error : segment type error." << std::endl;
  }
}

/// find segment in all segment list
FPdbSpecialNetEdgeSegment* FPdbSpecialNetEdgeSegmenArray::findSegmentByCoordinate(idb::IdbCoordinate<int32_t>* coordinate)
{
  /// find in vdd list
  for (FPdbSpecialNetEdgeSegment* edge_segment : _vdd_list) {
    vector<idb::IdbSpecialWireSegment*>& wire_segment_list = edge_segment->get_segment_list();
    if (wire_segment_list.size() <= 0) {
      continue;
    }

    for (idb::IdbSpecialWireSegment* wire_segment : wire_segment_list) {
      if (wire_segment->get_bounding_box()->containPoint(coordinate)) {
        return edge_segment;
      }
    }
  }
  /// find in vss list
  for (FPdbSpecialNetEdgeSegment* edge_segment : _vss_list) {
    vector<idb::IdbSpecialWireSegment*>& wire_segment_list = edge_segment->get_segment_list();
    if (wire_segment_list.size() <= 0) {
      continue;
    }

    for (idb::IdbSpecialWireSegment* wire_segment : wire_segment_list) {
      if (wire_segment->get_bounding_box()->containPoint(coordinate)) {
        return edge_segment;
      }
    }
  }

  return nullptr;
}

// find edge segment
FPdbSpecialNetEdgeSegment* FPdbSpecialNetEdgeSegmenArray::find_segment_edge_by_coordinate(int32_t coordinate_x_y)
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

/// judge if the direction of 2 points has the same direction of prefer layer direction
bool FPdbSpecialNetEdgeSegmenArray::hasSameOrient(idb::IdbCoordinate<int32_t>* start, idb::IdbCoordinate<int32_t>* end)
{
  /// horizontal or vertical
  return ((start->get_y() == end->get_y() && _layer->is_horizontal()) || (start->get_x() == end->get_x() && _layer->is_vertical())) ? true
                                                                                                                                    : false;
}

void FPdbSpecialNetEdgeSegmenArray::cutStripe(FPdbSpecialNetEdgeSegment* edge_segment_connected, idb::IdbCoordinate<int32_t>* start,
                                              idb::IdbCoordinate<int32_t>* end)
{
  std::cout << "####################################################################" << std::endl;
  std::cout << "Points  == ";
  std::cout << " ( " << start->get_x() << " , " << start->get_y() << " ) , ( " << end->get_x() << " , " << end->get_y() << " ) "
            << std::endl;
  idb::IdbSpecialWireSegment* segment = nullptr;
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

void FPdbSpecialNetEdgeSegmenArray::adjustStripe(idb::IdbSpecialWireSegment* sp_wire, idb::IdbCoordinate<int32_t>* start,
                                                 idb::IdbCoordinate<int32_t>* end)
{
  IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(sp_wire->get_layer());
  IdbRect* bouding_box = sp_wire->get_bounding_box();
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
/**
 * @brief  find intersect coordinate between two special wire segment
 * @param  segment_first
 * @param  segment_second
 * @param  intersect_coordinate
 * @return true
 * @return false
 */
bool CutStripe::get_intersect_coordinate(idb::IdbSpecialWireSegment* segment_first, idb::IdbSpecialWireSegment* segment_second,
                                         IdbRect& intersect_coordinate)
{
  if (!(segment_first->is_line() && segment_second->is_line())) {
    return false;
  }

  /// if intersect
  // idb::IdbCoordinate<int32_t> *start = segment_second->get_point_start();
  // idb::IdbCoordinate<int32_t> *end = segment_second->get_point_second();
  // if (segment_first->containLine(start, end)) {
  //     /// segment compare is horizontal
  //     if (start->get_y() == end->get_y()) {
  //         intersect_coordinate.set_x(segment_first->get_point_start()->get_x());
  //         intersect_coordinate.set_y(start->get_y());
  //     } else {
  //         /// segment compare is vertical
  //         intersect_coordinate.set_x(start->get_x());
  //         intersect_coordinate.set_y(segment_first->get_point_start()->get_y());
  //     }
  //     return true;
  // }
  if (segment_first->get_bounding_box()->isIntersection(*(segment_second->get_bounding_box()))) {
    int32_t rect_llx = segment_first->get_bounding_box()->get_low_x();
    int32_t rect_lly = segment_first->get_bounding_box()->get_low_y();
    int32_t rect_urx = segment_first->get_bounding_box()->get_high_x();
    int32_t rect_ury = segment_first->get_bounding_box()->get_high_y();
    int32_t seg_llx = segment_second->get_bounding_box()->get_low_x();
    int32_t seg_lly = segment_second->get_bounding_box()->get_low_y();
    int32_t seg_urx = segment_second->get_bounding_box()->get_high_x();
    int32_t seg_ury = segment_second->get_bounding_box()->get_high_y();
    std::vector<int32_t> x;
    x.push_back(rect_llx);
    x.push_back(rect_urx);
    x.push_back(seg_llx);
    x.push_back(seg_urx);
    std::vector<int32_t> y;
    y.push_back(rect_lly);
    y.push_back(rect_ury);
    y.push_back(seg_lly);
    y.push_back(seg_ury);
    sort(x.begin(), x.end(), [](int32_t a, int32_t b) { return a < b; });
    sort(y.begin(), y.end(), [](int32_t a, int32_t b) { return a < b; });
    IdbRect overlap = IdbRect(x[1], y[1], x[2], y[2]);
    intersect_coordinate = overlap;
    return true;
  }
  return false;
}
bool get_intersect_coordinate(idb::IdbSpecialWireSegment* segment_first, idb::IdbSpecialWireSegment* segment_second,
                              idb::IdbCoordinate<int32_t>& intersect_coordinate)
{
  if (!(segment_first->is_line() && segment_second->is_line())) {
    return false;
  }

  idb::IdbCoordinate<int32_t>* start = segment_second->get_point_start();
  idb::IdbCoordinate<int32_t>* end = segment_second->get_point_second();
  if (segment_first->containLine(start, end)) {
    /// segment compare is horizontal
    if (start->get_y() == end->get_y()) {
      intersect_coordinate.set_x(segment_first->get_point_start()->get_x());
      intersect_coordinate.set_y(start->get_y());
    } else {
      /// segment compare is vertical
      intersect_coordinate.set_x(start->get_x());
      intersect_coordinate.set_y(segment_first->get_point_start()->get_y());
    }
    return true;
  }

  return false;
}

}  // namespace ipdn