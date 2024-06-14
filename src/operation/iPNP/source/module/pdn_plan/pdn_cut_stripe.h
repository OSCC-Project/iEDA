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

#include <string>
#include <vector>

namespace idb {

class IdbSpecialWireSegment;
class IdbSpecialWire;
class IdbLayerRouting;
class IdbLayer;
class IdbRect;

template <typename T>
class IdbCoordinate;

enum class SegmentType : int8_t;

}  // namespace idb

namespace ipdn {

class FPdbSpecialNetEdgeSegment
{
 public:
  FPdbSpecialNetEdgeSegment();
  FPdbSpecialNetEdgeSegment(idb::SegmentType type);

  ~FPdbSpecialNetEdgeSegment();

  // getter
  idb::IdbSpecialWire* get_wire() { return _wire; }
  std::vector<idb::IdbSpecialWireSegment*>& get_segment_list() { return _segment_list; }
  bool is_vdd();
  bool is_vss();
  int32_t get_coordinate() { return _coordinate_x_y; }

  // setter
  void set_wire(idb::IdbSpecialWire* wire) { _wire = wire; }
  void add_segment(idb::IdbSpecialWireSegment* segment) { _segment_list.emplace_back(segment); }
  void set_type(idb::SegmentType type) { _type = type; }
  void set_type_vdd();
  void set_type_vss();
  void set_coordinate(int32_t value) { _coordinate_x_y = value; }
  void reset_segment_list(idb::IdbSpecialWireSegment* segment)
  {
    _segment_list.clear();
    _segment_list.emplace_back(segment);
  }

  /// operator
  idb::IdbSpecialWireSegment* isStripeCrossLine(idb::IdbCoordinate<int32_t>* start, idb::IdbCoordinate<int32_t>* end);
  idb::IdbSpecialWireSegment* cutStripe(FPdbSpecialNetEdgeSegment* edge_segment_connected, idb::IdbCoordinate<int32_t>* start,
                                        idb::IdbCoordinate<int32_t>* end);
  void add_segment_list_by_coordinate(std::vector<idb::IdbCoordinate<int32_t>*>& point_list);

 private:
  idb::IdbSpecialWire* _wire;
  std::vector<idb::IdbSpecialWireSegment*> _segment_list;
  int32_t _coordinate_x_y;  /// x or y of the coordinate, in order to diff from other segment
  idb::SegmentType _type;
};

class FPdbSpecialNetEdgeSegmenArray
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
  FPdbSpecialNetEdgeSegmenArray();

  ~FPdbSpecialNetEdgeSegmenArray()
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

    for (FPdbSpecialNetEdgeSegment* vdd_segment : _vdd_list) {
      delete vdd_segment;
      vdd_segment = nullptr;
    }
    std::vector<FPdbSpecialNetEdgeSegment*>().swap(_vdd_list);

    for (FPdbSpecialNetEdgeSegment* vss_segment : _vss_list) {
      delete vss_segment;
      vss_segment = nullptr;
    }
    std::vector<FPdbSpecialNetEdgeSegment*>().swap(_vss_list);
  }

  // getter
  idb::IdbLayerRouting* get_layer() { return _layer; }
  FPdbSpecialNetEdgeSegment* get_segment_vdd_1() { return _segment_vdd_1; }
  FPdbSpecialNetEdgeSegment* get_segment_vdd_2() { return _segment_vdd_2; }
  FPdbSpecialNetEdgeSegment* get_segment_vss_1() { return _segment_vss_1; }
  FPdbSpecialNetEdgeSegment* get_segment_vss_2() { return _segment_vss_2; }
  std::vector<FPdbSpecialNetEdgeSegment*>& get_vdd_list() { return _vdd_list; }
  std::vector<FPdbSpecialNetEdgeSegment*>& get_vss_list() { return _vss_list; }

  // setter
  void set_layer(idb::IdbLayerRouting* layer) { _layer = layer; }

  FPdbSpecialNetEdgeSegment* add_vdd()
  {
    FPdbSpecialNetEdgeSegment* new_segment = new FPdbSpecialNetEdgeSegment();
    new_segment->set_type_vdd();
    _vdd_list.emplace_back(new_segment);
    return new_segment;
  }
  FPdbSpecialNetEdgeSegment* add_vss()
  {
    FPdbSpecialNetEdgeSegment* new_segment = new FPdbSpecialNetEdgeSegment();
    new_segment->set_type_vss();
    _vss_list.emplace_back(new_segment);
    return new_segment;
  }

  // operator
  void updateSegment(idb::IdbSpecialWireSegment* segment, idb::IdbSpecialWire* wire, idb::SegmentType type);
  void updateSegmentEdgePoints(idb::IdbSpecialWireSegment* segment, idb::IdbSpecialWire* wire, idb::SegmentType type);
  void updateSegmentEdgeVias(idb::IdbSpecialWireSegment* segment, idb::IdbSpecialWire* wire, idb::SegmentType type);
  void updateSegmentArray(idb::IdbSpecialWireSegment* segment, idb::IdbSpecialWire* wire, idb::SegmentType type);

  FPdbSpecialNetEdgeSegment* findSegmentByCoordinate(idb::IdbCoordinate<int32_t>* coordinate);
  FPdbSpecialNetEdgeSegment* find_segment_edge_by_coordinate(int32_t coordinate_x_y);
  bool addSegmentByCoordinateList(std::vector<idb::IdbCoordinate<int32_t>*>& coordinate_list);

  bool hasSameOrient(idb::IdbCoordinate<int32_t>* start, idb::IdbCoordinate<int32_t>* end);
  void cutStripe(FPdbSpecialNetEdgeSegment* edge_segment, idb::IdbCoordinate<int32_t>* start, idb::IdbCoordinate<int32_t>* end);

  void adjustStripe(idb::IdbSpecialWireSegment* sp_wire, idb::IdbCoordinate<int32_t>* start, idb::IdbCoordinate<int32_t>* end);

 private:
  idb::IdbLayerRouting* _layer;
  FPdbSpecialNetEdgeSegment* _segment_vdd_1;  /// bottom or left vdd segment
  FPdbSpecialNetEdgeSegment* _segment_vdd_2;  /// top or right vdd segment
  FPdbSpecialNetEdgeSegment* _segment_vss_1;  /// bottom or left vss segment
  FPdbSpecialNetEdgeSegment* _segment_vss_2;  /// top or right vdd segment

  std::vector<FPdbSpecialNetEdgeSegment*> _vdd_list;
  std::vector<FPdbSpecialNetEdgeSegment*> _vss_list;
};

class CutStripe
{
 public:
  CutStripe() = default;
  ~CutStripe() { clear_edge_list(); }

  void initEdge();
  bool connectIOPinToPowerStripe(std::vector<idb::IdbCoordinate<int32_t>*>& point_list, idb::IdbLayer* layer);

  bool connectPowerStripe(std::vector<idb::IdbCoordinate<int32_t>*>& point_list, std::string net_name, std::string layer_name,
                          int32_t width = -1)
  {
    return addPowerStripe(point_list, net_name, layer_name, width);
  }

  bool get_intersect_coordinate(idb::IdbSpecialWireSegment* segment_first, idb::IdbSpecialWireSegment* segment_second,
                                idb::IdbRect& intersect_coordinate);
  bool get_intersect_coordinate(idb::IdbSpecialWireSegment* segment_first, idb::IdbSpecialWireSegment* segment_second,
                                idb::IdbCoordinate<int32_t>& intersect_coordinate);
  bool containLine(idb::IdbSpecialWireSegment* segment_first, idb::IdbCoordinate<int32_t>* start, idb::IdbCoordinate<int32_t>* end);

 private:
  std::vector<FPdbSpecialNetEdgeSegmenArray*> _edge_list;

  bool connectIO(std::vector<idb::IdbCoordinate<int32_t>*>& point_list, idb::IdbLayer* layer);

  void clear_edge_list();

  bool addPowerStripe(std::vector<idb::IdbCoordinate<int32_t>*>& point_list, std::string net_name, std::string layer_name, int width = -1);
  FPdbSpecialNetEdgeSegmenArray* add_edge_segment_array_for_layer(idb::IdbLayerRouting* layer);
  FPdbSpecialNetEdgeSegmenArray* find_edge_segment_array_by_layer(idb::IdbLayer* layer);
};
}  // namespace ipdn