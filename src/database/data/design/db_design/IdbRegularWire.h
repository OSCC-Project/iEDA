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
 * @file		IdbRegularWire.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe Regular wire information.
 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////Wire////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../../basic/geometry/IdbGeometry.h"
#include "../IdbEnum.h"
#include "IdbPins.h"
#include "IdbVias.h"

namespace idb {

using std::string;
using std::vector;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @description
 *
############Regular Wiring Statement############
{+ COVER | + FIXED | + ROUTED | + NOSHIELD}
  layerName [TAPER | TAPERRULE ruleName] [STYLE styleNum]
    routingPoints
  [NEW layerName [TAPER | TAPERRULE ruleName] [STYLE styleNum]
    routingPoints
] ...
 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define _POINT_START_ 0
#define _POINT_SECOND_ 1
#define _POINT_MAX_ 2
class IdbRegularWireSegment
{
 public:
  IdbRegularWireSegment();
  ~IdbRegularWireSegment();

  // getter
  const bool is_new_layer() { return _is_new_layer; }
  const string get_layer_name() const { return _layer_name; }
  IdbLayer* get_layer() { return _layer; }
  //  const int32_t get_route_width() const {return _route_width;}
  //  const IdbWireShapeType get_shape_type() const {return _shape_type;}
  //  const int32_t get_style() const {return _style;}
  vector<IdbCoordinate<int32_t>*>& get_point_list() { return _point_list; }
  int32_t get_point_number() { return _point_list.size(); }
  IdbCoordinate<int32_t>* get_point(size_t index);
  IdbCoordinate<int32_t>* get_point_start();
  IdbCoordinate<int32_t>* get_point_second();
  IdbCoordinate<int32_t>* get_point_end();
  const bool is_wire() { return _point_list.size() >= _POINT_MAX_ ? true : false; }
  const bool is_via() { return _is_via; }
  vector<IdbVia*> get_via_list() { return _via_list; }
  const bool is_rect() { return _is_rect; }
  IdbRect* get_delta_rect() { return _delta_rect; }
  IdbRect get_segment_rect();

  // setter
  void set_layer_status(bool is_new) { _is_new_layer = is_new; }
  void set_layer_as_new() { _is_new_layer = true; }
  void set_layer_name(string name) { _layer_name = name; }
  void set_layer(IdbLayer* layer) { _layer = layer; }
  void init_point_list(int32_t size) { _point_list.reserve(size); }
  IdbCoordinate<int32_t>* add_point(int32_t x, int32_t y);
  IdbCoordinate<int32_t>* add_virtual_point(int32_t x, int32_t y);
  void set_is_via(bool is_via) { _is_via = is_via; }
  void set_via_list(vector<IdbVia*> via_list);
  void set_via(IdbVia* via) { _via_list.emplace_back(via); }
  IdbVia* copy_via(IdbVia* via);
  void init_via_list(int32_t size) { _via_list.reserve(size); }
  void set_is_rect(bool is_rect) { _is_rect = is_rect; }
  void set_delta_rect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y);
  bool is_virtual(IdbCoordinate<int32_t>* point) { return _virtual_points.contains(point); }
  // operator
  void clearPoints();
  void clear();
  uint64_t length();
  bool isIntersection(IdbLayerShape* layer_shape);
  bool isIntersection(IdbRegularWireSegment* segment);

 private:
  bool _is_new_layer;
  bool _is_via;
  bool _is_rect;

  string _layer_name;
  IdbLayer* _layer;
  // int32_t _route_width;
  // IdbWireShapeType _shape_type;
  // int32_t _style;

  vector<IdbVia*> _via_list;
  IdbRect* _delta_rect;

  std::unordered_set<IdbCoordinate<int32_t>*> _virtual_points;
  vector<IdbCoordinate<int32_t>*> _point_list;

  /// connection check
  bool isConnectWireToWire(IdbRegularWireSegment* segment);
  bool isConnectWireToVia(IdbRegularWireSegment* segment);
  bool isConnectWireToDeltaRect(IdbRegularWireSegment* segment);
  bool isConnectViaToVia(IdbRegularWireSegment* segment);
  bool isConnectRectToVia(IdbRegularWireSegment* segment);
  bool isConnectRectToRect(IdbRegularWireSegment* segment);
};

class IdbRegularWire
{
 public:
  IdbRegularWire();
  ~IdbRegularWire();

  // getter
  vector<IdbRegularWireSegment*>& get_segment_list() { return _segment_list; }
  int32_t get_num() { return _segment_list.size(); }
  uint get_via_num();
  uint64_t get_wire_num()
  {
    uint64_t number = 0;
    for (auto seg : _segment_list) {
      if (seg->is_wire()) {
        number++;
      }
    }

    return number;
  }

  uint64_t get_patch_num()
  {
    uint64_t number = 0;
    for (auto seg : _segment_list) {
      if (seg->is_rect()) {
        number++;
      }
    }

    return number;
  }

  IdbWiringStatement get_wire_statement() { return _wire_state; }
  string& get_shiled_name() { return _shiled_name; }

  // setter
  void set_wire_state(IdbWiringStatement state) { _wire_state = state; }
  void set_wire_state(string state);
  void set_shield_name(string shield_name) { _shiled_name = shield_name; }
  IdbRegularWireSegment* add_segment(IdbRegularWireSegment* segment = nullptr);
  IdbRegularWireSegment* add_segment(string layer_name);
  void delete_seg(IdbRegularWireSegment* seg_del);
  void reset();

  // operator
  void init(int32_t size) { _segment_list.reserve(size); }
  void clear_segment();
  uint64_t wireLength();

 private:
  IdbWiringStatement _wire_state;
  string _shiled_name;
  vector<IdbRegularWireSegment*> _segment_list;
};

class IdbRegularWireList
{
 public:
  IdbRegularWireList();
  ~IdbRegularWireList();

  // getter
  vector<IdbRegularWire*>& get_wire_list() { return _wire_list; }
  int32_t get_num() { return _wire_list.size(); }
  IdbRegularWire* find_wire(size_t index) { return _wire_list.size() > index ? _wire_list.at(index) : nullptr; }

  // setter
  IdbRegularWire* add_wire(IdbRegularWire* wire = nullptr);
  void reset();

  // operator
  void init(int32_t size) { _wire_list.reserve(size); }
  void clear();
  uint64_t wireLength();

 private:
  vector<IdbRegularWire*> _wire_list;
};

}  // namespace idb
