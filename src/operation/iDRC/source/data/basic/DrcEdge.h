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
#ifndef IDRC_SRC_DB_DRCEDGE_H_
#define IDRC_SRC_DB_DRCEDGE_H_

#include <algorithm>
#include <memory>

#include "BoostType.h"
#include "DrcCoordinate.h"
#include "DrcCorner.hpp"
#include "DrcEnum.h"
#include "DrcNet.h"
#include "DrcPoly.hpp"

namespace idrc {
class DrcPolygon;
class DrcNet;
class DrcEdge : public BoostSegment
{
 public:
  DrcEdge() : BoostSegment(), _layer_id(-1), _is_fixed(false), _edge_dir(EdgeDirection::kNone) {}
  explicit DrcEdge(const DrcEdge& segment)
      : BoostSegment(segment),
        _layer_id(-1),
        _is_fixed(false),
        _owner_poly(nullptr),
        _owner_net(nullptr),
        _pre_edge(nullptr),
        _next_edge(nullptr),
        _low_corner(nullptr),
        _high_corner(nullptr),
        _edge_dir(EdgeDirection::kNone),
        _directed_segment(segment)

  {
  }
  // explicit DrcEdge(const DrcEdge& other) = default;
  explicit DrcEdge(DrcEdge&& other) = default;
  ~DrcEdge() {}
  DrcEdge& operator=(const DrcEdge& other) = default;
  DrcEdge& operator=(DrcEdge& other) = default;
  // setter
  void setSegment(const BoostSegment& in) { _directed_segment = in; }
  void setSegment(const BoostPoint& bp, const BoostPoint& ep)
  {
    _directed_segment.low(bp);
    _directed_segment.high(ep);
  }
  void setPrevEdge(DrcEdge* in) { _pre_edge = in; }
  void setNextEdge(DrcEdge* in) { _next_edge = in; }
  void setLowCorner(DrcCorner* in) { _low_corner = in; }
  void setHighCorner(DrcCorner* in) { _high_corner = in; }

  void set_owner_polygon(DrcPoly* polygon) { _owner_poly = polygon; }
  void set_layer_id(int layerId) { _layer_id = layerId; }
  void set_is_fixed(bool isFixed) { _is_fixed = isFixed; }
  void set_edge_dir(EdgeDirection dir) { _edge_dir = dir; }
  void set_segment(const BoostSegment& boost_segment) { _directed_segment = boost_segment; }
  void setDir()
  {
    if (get_begin_x() == get_end_x()) {
      if (get_begin_y() < get_end_y()) {
        _edge_dir = EdgeDirection::kNorth;
      } else {
        _edge_dir = EdgeDirection::kSouth;
      }
      // WEST / EAST
    } else {
      if (get_begin_x() < get_end_x()) {
        _edge_dir = EdgeDirection::kEast;
      } else {
        _edge_dir = EdgeDirection::kWest;
      }
    }
  }
  // getter
  int getLength() { return std::max(get_max_x() - get_min_x(), get_max_y() - get_min_y()); }
  DrcEdge* getPreEdge() { return _pre_edge; }
  DrcEdge* getNextEdge() { return _next_edge; }
  int get_layer_id() const { return _layer_id; }
  bool get_is_fixed() const { return _is_fixed; }
  EdgeDirection get_edge_dir() const { return _edge_dir; }
  BoostPoint get_begin() const { return _directed_segment.low(); }
  BoostPoint get_end() const { return _directed_segment.high(); }
  int get_begin_x() const { return get_begin().x(); }
  int get_begin_y() const { return get_begin().y(); }
  int get_end_x() const { return get_end().x(); }
  int get_end_y() const { return get_end().y(); }

  int get_min_x() const { return std::min(get_begin_x(), get_end_x()); }
  int get_min_y() const { return std::min(get_begin_y(), get_end_y()); }
  int get_max_x() const { return std::max(get_begin_x(), get_end_x()); }
  int get_max_y() const { return std::max(get_begin_y(), get_end_y()); }

  DrcPoly* get_owner_polygon() { return _owner_poly; }
  // function
  bool isHorizontal() const { return get_begin_y() == get_end_y(); }
  bool isVertical() const { return get_begin_x() == get_end_x(); }

  void addToPoly(DrcPoly* in) { _owner_poly = in; }
  void addToNet(DrcNet* in) { _owner_net = in; }

 private:
  int _layer_id = -1;
  // int _width = -1;
  bool _is_fixed;
  DrcPoly* _owner_poly;
  DrcNet* _owner_net;
  DrcEdge* _pre_edge;
  DrcEdge* _next_edge;
  DrcCorner* _low_corner;
  DrcCorner* _high_corner;
  EdgeDirection _edge_dir;
  BoostSegment _directed_segment;
};
}  // namespace idrc

#endif