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
/**
 * @project		iDB
 * @file		IdbSpecialWire.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe special wire information.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../../../basic/geometry/IdbGeometry.h"
#include "../IdbEnum.h"
#include "../IdbObject.h"
#include "../db_design/IdbPins.h"
#include "../db_design/IdbVias.h"

namespace idb {

using std::string;
using std::vector;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @description
 *
#####Special Wiring Statement#####
[[+ COVER | + FIXED | + ROUTED | + SHIELD shieldNetName]
  [+ SHAPE shapeType] [+ MASK maskNum]
  + POLYGON layerName pt pt pt ...
  | + RECT layerName pt pt
  | + VIA viaName [orient] pt ...
|{+ COVER | + FIXED | + ROUTED | + SHIELD shieldNetName}
  layerName routeWidth
    [+ SHAPE
      {RING | PADRING | BLOCKRING | STRIPE | FOLLOWPIN
      | IOWIRE | COREWIRE | BLOCKWIRE | BLOCKAGEWIRE | FILLWIRE
      | FILLWIREOPC | DRCFILL}]
    [+ STYLE styleNum]
    routingPoints
  [NEW layerName routeWidth
    [+ SHAPE
      {RING | PADRING | BLOCKRING | STRIPE | FOLLOWPIN
      | IOWIRE | COREWIRE | BLOCKWIRE | BLOCKAGEWIRE | FILLWIRE
      | FILLWIREOPC | DRCFILL}]
    [+ STYLE styleNum]
    routingPoints
  ] ...
] ...
 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define _POINT_START_ 0
#define _POINT_SECOND_ 1
#define _POINT_MAX_ 2

enum class SegmentType : int8_t
{
  kNone,
  kVDD,
  kVSS,
  kMax
};

class IdbSpecialWireSegment : public IdbObject
{
 public:
  IdbSpecialWireSegment();
  ~IdbSpecialWireSegment();

  // getter
  const bool is_new_layer() { return _is_new_layer; }
  //  const string get_layer_name()const {return _layer->get_name();}
  IdbLayer* get_layer() { return _layer; }
  const int32_t get_route_width() const { return _route_width; }
  const IdbWireShapeType get_shape_type() const { return _shape_type; }
  const int32_t get_style() const { return _style; }
  bool is_tripe() { return _shape_type == IdbWireShapeType::kStripe ? true : false; }
  bool is_follow_pin() { return _shape_type == IdbWireShapeType::kFollowPin ? true : false; }
  bool is_line() { return (is_tripe() || is_follow_pin()) && get_point_num() >= _POINT_MAX_ ? true : false; }
  vector<IdbCoordinate<int32_t>*>& get_point_list() { return _point_list; }
  int32_t get_point_num() { return _point_list.size(); }
  IdbCoordinate<int32_t>* get_point(uint32_t index);
  IdbCoordinate<int32_t>* get_point_start();
  IdbCoordinate<int32_t>* get_point_second();
  const bool is_via() { return _is_via; }
  IdbVia* get_via() { return _via; }
  IdbCoordinate<int32_t>* get_point_left();
  IdbCoordinate<int32_t>* get_point_right();
  IdbCoordinate<int32_t>* get_point_top();
  IdbCoordinate<int32_t>* get_point_bottom();
  IdbRect* get_delta_rect() { return _delta_rect; }
  const bool is_rect() { return _is_rect; }
  bool is_horizontal();
  bool is_vertical();

  // setter
  void set_layer_status(bool is_new) { _is_new_layer = is_new; }
  void set_layer_as_new() { _is_new_layer = true; }
  void set_layer(IdbLayer* layer) { _layer = layer; }
  void set_route_width(int32_t width) { _route_width = width; }
  void set_shape_type(IdbWireShapeType shape_type) { _shape_type = shape_type; }
  void set_shape_type(string shape_type);
  void set_style(int32_t style) { _style = style; }
  void set_is_via(bool is_via) { _is_via = is_via; }
  void set_via(IdbVia* via) { _via = via; }
  void set_is_rect(bool is_rect) { _is_rect = is_rect; }
  void set_delta_rect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y);
  IdbVia* copy_via(IdbVia* via);
  IdbCoordinate<int32_t>* add_point(int32_t x, int32_t y);

  bool set_bounding_box();

  // operator
  IdbSpecialWireSegment* copy();
  void adjustStripe(IdbCoordinate<int32_t>* start, IdbCoordinate<int32_t>* end);
  bool containLine(IdbCoordinate<int32_t>* start, IdbCoordinate<int32_t>* end);
  bool get_intersect_coordinate(IdbSpecialWireSegment* segment, IdbCoordinate<int32_t>& intersect_coordinate);

  int32_t length();

 private:
  // string _layer_name;
  IdbLayer* _layer;
  IdbVia* _via;
  int32_t _route_width;
  int32_t _style;
  IdbWireShapeType _shape_type;
  bool _is_new_layer;
  bool _is_via;
  bool _is_rect;
  vector<IdbCoordinate<int32_t>*> _point_list;
  IdbRect* _delta_rect;
};

enum class CoordinatePosition : int8_t
{
  kStart,
  kEnd,
  kBoth
};
void adjustCoordinate(IdbCoordinate<int32_t>* start, IdbCoordinate<int32_t>* end, int32_t width,
                      CoordinatePosition coor_post = CoordinatePosition::kBoth);
class IdbSpecialWire
{
 public:
  IdbSpecialWire();
  ~IdbSpecialWire();

  // getter
  vector<IdbSpecialWireSegment*>& get_segment_list() { return _segment_list; }
  size_t get_num() { return _segment_list.size(); }
  uint get_via_num();
  IdbWiringStatement get_wire_state() { return _wire_state; }
  string& get_shiled_name() { return _shiled_name; }
  IdbSpecialWireSegment* get_segment(size_t index) { return get_num() > 0 && get_num() > index ? _segment_list[index] : nullptr; }

  IdbSpecialWireSegment* get_layer_segment(string layer_name)
  {
    if (layer_name.empty() || get_num() <= 0) {
      return nullptr;
    }
    for (IdbSpecialWireSegment* segment : _segment_list) {
      if (segment->get_layer()->compareLayer(layer_name)) {
        return segment;
      }
    }

    return nullptr;
  }

  // setter
  void set_wire_state(IdbWiringStatement state) { _wire_state = state; }
  void set_wire_state(string state);
  void set_shield_name(string shield_name) { _shiled_name = shield_name; }
  IdbSpecialWireSegment* add_segment(IdbSpecialWireSegment* segment = nullptr);
  IdbSpecialWireSegment* add_segment_stripe(IdbCoordinate<int32_t>* start, IdbCoordinate<int32_t>* end,
                                            IdbSpecialWireSegment* segment_connected, int32_t width = -1);
  //   IdbSpecialWireSegment* addSegmentVia( int32_t coord_x, int32_t coord_y, IdbVia* via);

  int32_t add_segment_list(vector<IdbCoordinate<int32_t>*>& point_list, IdbSpecialWireSegment* segment_connected, int32_t width = -1);

  // operator
  void init(int32_t size) { _segment_list.reserve(size); }
  void reset();

  void removeViaInBoundingBox(IdbRect rect, IdbLayer* layer);

 private:
  int32_t _num;
  IdbWiringStatement _wire_state;
  string _shiled_name;
  vector<IdbSpecialWireSegment*> _segment_list;
};

class IdbSpecialWireList
{
 public:
  IdbSpecialWireList();
  ~IdbSpecialWireList();

  // getter
  vector<IdbSpecialWire*>& get_wire_list() { return _wire_list; }
  int32_t get_num() { return _wire_list.size(); }
  IdbSpecialWire* find_wire(size_t index) { return _wire_list.size() > 0 && _wire_list.size() > index ? _wire_list[index] : nullptr; };

  // setter
  IdbSpecialWire* add_wire(IdbSpecialWire* wire = nullptr, IdbWiringStatement state = IdbWiringStatement::kRouted);
  void reset();

  // operator
  void init(int32_t size) { _wire_list.reserve(size); }

 private:
  int32_t _num;
  vector<IdbSpecialWire*> _wire_list;
};

class IdbSpecialNetEdgeSegment
{
 public:
  IdbSpecialNetEdgeSegment()
  {
    _wire = nullptr;
    _type = SegmentType::kNone;
    _coordinate_x_y = -1;
  }
  IdbSpecialNetEdgeSegment(SegmentType type)
  {
    _wire = nullptr;
    _type = type;
    _coordinate_x_y = -1;
  }
  ~IdbSpecialNetEdgeSegment()
  {
    _segment_list.clear();
    vector<IdbSpecialWireSegment*>().swap(_segment_list);

    _wire = nullptr;
  }

  // getter
  IdbSpecialWire* get_wire() { return _wire; }
  vector<IdbSpecialWireSegment*>& get_segment_list() { return _segment_list; }
  bool is_vdd() { return _type == SegmentType::kVDD ? true : false; }
  bool is_vss() { return _type == SegmentType::kVSS ? true : false; }
  int32_t get_coordinate() { return _coordinate_x_y; }

  // setter
  void set_wire(IdbSpecialWire* wire) { _wire = wire; }
  void add_segment(IdbSpecialWireSegment* segment) { _segment_list.emplace_back(segment); }
  void set_type(SegmentType type) { _type = type; }
  void set_type_vdd() { _type = SegmentType::kVDD; }
  void set_type_vss() { _type = SegmentType::kVSS; }
  void set_coordinate(int32_t value) { _coordinate_x_y = value; }
  void reset_segment_list(IdbSpecialWireSegment* segment)
  {
    _segment_list.clear();
    _segment_list.emplace_back(segment);
  }

  /// operator
  IdbSpecialWireSegment* isStripeCrossLine(IdbCoordinate<int32_t>* start, IdbCoordinate<int32_t>* end);
  IdbSpecialWireSegment* cutStripe(IdbSpecialNetEdgeSegment* edge_segment_connected, IdbCoordinate<int32_t>* start,
                                   IdbCoordinate<int32_t>* end);
  //   bool cutStripe(IdbCoordinate<int32_t>* start, IdbCoordinate<int32_t>* end);

  void add_segment_list_by_coordinate(vector<IdbCoordinate<int32_t>*>& point_list);

 private:
  IdbSpecialWire* _wire;
  vector<IdbSpecialWireSegment*> _segment_list;
  int32_t _coordinate_x_y;  /// x or y of the coordinate, in order to diff from other segment
  SegmentType _type;
};

class IdbSpecialNetEdgeSegmenArray
{
  enum EdgeIndex : int8_t
  {
    kLeftOrBottomFirst = 0,
    kLeftOrBottomSecond = 1,
    kRightOrTopSecond = 2,
    kRightOrTopFirst = 3,
    kMax = 4,
  };

 public:
  IdbSpecialNetEdgeSegmenArray()
  {
    _layer = nullptr;
    _segment_vdd_1 = new IdbSpecialNetEdgeSegment(SegmentType::kVDD);
    _segment_vdd_2 = new IdbSpecialNetEdgeSegment(SegmentType::kVDD);
    _segment_vss_1 = new IdbSpecialNetEdgeSegment(SegmentType::kVSS);
    _segment_vss_2 = new IdbSpecialNetEdgeSegment(SegmentType::kVSS);
  }

  ~IdbSpecialNetEdgeSegmenArray()
  {
    if (_segment_vdd_1 != nullptr) {
      delete _segment_vdd_1;
      _segment_vdd_1 = nullptr;
    }
    if (_segment_vdd_2 != nullptr) {
      delete _segment_vdd_2;
      _segment_vdd_2 = nullptr;
    }
    if (_segment_vss_1 != nullptr) {
      delete _segment_vss_1;
      _segment_vss_1 = nullptr;
    }
    if (_segment_vss_2 != nullptr) {
      delete _segment_vss_2;
      _segment_vss_2 = nullptr;
    }

    for (IdbSpecialNetEdgeSegment* vdd_segment : _vdd_list) {
      if (vdd_segment != nullptr) {
        delete vdd_segment;
        vdd_segment = nullptr;
      }
    }
    _vdd_list.clear();
    vector<IdbSpecialNetEdgeSegment*>().swap(_vdd_list);

    for (IdbSpecialNetEdgeSegment* vss_segment : _vss_list) {
      if (vss_segment != nullptr) {
        delete vss_segment;
        vss_segment = nullptr;
      }
    }
    _vss_list.clear();
    vector<IdbSpecialNetEdgeSegment*>().swap(_vss_list);
  }

  // getter
  IdbLayerRouting* get_layer() { return _layer; }
  IdbSpecialNetEdgeSegment* get_segment_vdd_1() { return _segment_vdd_1; }
  IdbSpecialNetEdgeSegment* get_segment_vdd_2() { return _segment_vdd_2; }
  IdbSpecialNetEdgeSegment* get_segment_vss_1() { return _segment_vss_1; }
  IdbSpecialNetEdgeSegment* get_segment_vss_2() { return _segment_vss_2; }
  vector<IdbSpecialNetEdgeSegment*>& get_vdd_list() { return _vdd_list; }
  vector<IdbSpecialNetEdgeSegment*>& get_vss_list() { return _vss_list; }

  // setter
  void set_layer(IdbLayerRouting* layer) { _layer = layer; }

  IdbSpecialNetEdgeSegment* add_vdd()
  {
    IdbSpecialNetEdgeSegment* new_segment = new IdbSpecialNetEdgeSegment();
    new_segment->set_type_vdd();
    _vdd_list.emplace_back(new_segment);
    return new_segment;
  }
  IdbSpecialNetEdgeSegment* add_vss()
  {
    IdbSpecialNetEdgeSegment* new_segment = new IdbSpecialNetEdgeSegment();
    new_segment->set_type_vss();
    _vss_list.emplace_back(new_segment);
    return new_segment;
  }

  // operator
  void updateSegment(IdbSpecialWireSegment* segment, IdbSpecialWire* wire, SegmentType type);
  void updateSegmentEdgePoints(IdbSpecialWireSegment* segment, IdbSpecialWire* wire, SegmentType type);
  void updateSegmentEdgeVias(IdbSpecialWireSegment* segment, IdbSpecialWire* wire, SegmentType type);
  void updateSegmentArray(IdbSpecialWireSegment* segment, IdbSpecialWire* wire, SegmentType type);

  IdbSpecialNetEdgeSegment* findSegmentByCoordinate(IdbCoordinate<int32_t>* coordinate);
  IdbSpecialNetEdgeSegment* find_segment_edge_by_coordinate(int32_t coordinate_x_y);
  bool addSegmentByCoordinateList(vector<IdbCoordinate<int32_t>*>& coordinate_list);

  bool hasSameOrient(IdbCoordinate<int32_t>* start, IdbCoordinate<int32_t>* end);
  void cutStripe(IdbSpecialNetEdgeSegment* edge_segment, IdbCoordinate<int32_t>* start, IdbCoordinate<int32_t>* end);

 private:
  IdbLayerRouting* _layer;
  IdbSpecialNetEdgeSegment* _segment_vdd_1;  /// bottom or left vdd segment
  IdbSpecialNetEdgeSegment* _segment_vdd_2;  /// top or right vdd segment
  IdbSpecialNetEdgeSegment* _segment_vss_1;  /// bottom or left vss segment
  IdbSpecialNetEdgeSegment* _segment_vss_2;  /// top or right vdd segment

  vector<IdbSpecialNetEdgeSegment*> _vdd_list;
  vector<IdbSpecialNetEdgeSegment*> _vss_list;
};

}  // namespace idb
