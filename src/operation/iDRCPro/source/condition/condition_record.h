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

#include <vector>

#include "condition.h"
#include "drc_basic_rect.h"
#include "drc_basic_segment.h"
#include "idrc_engine.h"

namespace idrc {

class ConditionRecord
{
 public:
  ConditionRecord() { clear(); }
  ConditionRecord(Condition* condition, DrcCoordinate width_forward, DrcCoordinate width_backward, bool is_vertical)
      : _condition(condition), _wire_region(width_backward, width_forward), _is_wire_vertical(!is_vertical)
  {
  }

  ~ConditionRecord() = default;

  void clear()
  {
    _condition = nullptr;
    _spacing_region_forward.clear();
    _spacing_region_backward.clear();
  }

  // setter
  void set_condition(Condition* condition) { _condition = condition; }
  void set_polygon(DrcCoordinate polygon_forward, DrcCoordinate polygon_backward)
  {
    _wire_region.set_ur(polygon_forward);
    _wire_region.set_lb(polygon_backward);
  }
  void set_wire_region(DrcBasicRect wire_region) { _wire_region = wire_region; }
  void set_is_wire_vertical(bool is_vertical) { _is_wire_vertical = is_vertical; }

  // getter
  Condition* get_condition() { return _condition; }
  DrcBasicRect get_wire_region() { return _wire_region; }

  int get_polygon_forward() { return _is_wire_vertical ? _wire_region.get_ur().get_x() : _wire_region.get_ur().get_y(); }
  int get_polygon_backward() { return _is_wire_vertical ? _wire_region.get_lb().get_x() : _wire_region.get_lb().get_y(); }

  void recordSpacing(DrcBasicSegment spacing_forward, DrcBasicSegment spacing_backward)
  {
    recordSpacingForward(spacing_forward);
    recordSpacingBackward(spacing_backward);
  }

  void recordSpacingForward(DrcBasicSegment spacing_forward)
  {
    if (_condition) {
      if (!_spacing_region_forward.empty()) {
        auto& spacing_rect = _spacing_region_forward.back();
        spacing_rect.addPoint(spacing_forward.get_begin());
        spacing_rect.addPoint(spacing_forward.get_end());
      }
      _spacing_region_forward.emplace_back(spacing_forward.get_begin(), spacing_forward.get_end());
    }
  }

  void recordSpacingBackward(DrcBasicSegment spacing_backward)
  {
    if (_condition) {
      if (!_spacing_region_backward.empty()) {
        auto& spacing_rect = _spacing_region_backward.back();
        spacing_rect.addPoint(spacing_backward.get_begin());
        spacing_rect.addPoint(spacing_backward.get_end());
      }
      _spacing_region_backward.emplace_back(spacing_backward.get_begin(), spacing_backward.get_end());
    }
  }

  void endRecording(int bucket_coord)
  {
    // if (!_spacing_region_forward.empty()) {
    //   auto& last_rect_forward = _spacing_region_forward.back();
    //   is_vertical ? last_rect_forward.get_ur().set_x(bucket_coord) : last_rect_forward.get_ur().set_y(bucket_coord);
    // }
    // if (!_spacing_region_backward.empty()) {
    //   auto& last_rect_backward = _spacing_region_backward.back();
    //   is_vertical ? last_rect_backward.get_ur().set_x(bucket_coord) : last_rect_backward.get_ur().set_y(bucket_coord);
    // }
    _is_wire_vertical ? _wire_region.set_ur_y(bucket_coord) : _wire_region.set_ur_x(bucket_coord);
    // TODO: check rule
    int a = 0;
  }

  bool contains(int forward, int backward) { return get_polygon_forward() >= forward && get_polygon_backward() <= backward; }
  bool contains(DrcCoordinate* segment_forward, DrcCoordinate* segment_backward)
  {
    return _is_wire_vertical ? contains(segment_forward->get_x(), segment_backward->get_x())
                             : contains(segment_forward->get_y(), segment_backward->get_y());
  }

  bool in(int forward, int backward) { return get_polygon_forward() <= forward && get_polygon_backward() >= backward; }
  bool in(DrcCoordinate* segment_forward, DrcCoordinate* segment_backward)
  {
    return _is_wire_vertical ? in(segment_forward->get_x(), segment_backward->get_x())
                             : in(segment_forward->get_y(), segment_backward->get_y());
  }

  void copySpacing(ConditionRecord* record)
  {
    _spacing_region_forward = record->_spacing_region_forward;
    _spacing_region_backward = record->_spacing_region_backward;
  }

  DrcBasicSegment getSpacingInCurrentBucket(DrcBasicRect& spacing_rect, int bucket_coord, bool is_vertical)
  {
    auto coord_backward = spacing_rect.get_lb();
    is_vertical ? coord_backward.set_x(bucket_coord) : coord_backward.set_y(bucket_coord);
    auto coord_forward = spacing_rect.get_ur();
    is_vertical ? coord_forward.set_x(bucket_coord) : coord_forward.set_y(bucket_coord);
    return DrcBasicSegment(coord_backward, coord_forward);
  }

  DrcBasicSegment getCurrentSpacingForward(int bucket_coord, bool is_vertical)
  {
    return getSpacingInCurrentBucket(_spacing_region_forward.back(), bucket_coord, is_vertical);
  }

  DrcBasicSegment getCurrentSpacingBackward(int bucket_coord, bool is_vertical)
  {
    return getSpacingInCurrentBucket(_spacing_region_backward.back(), bucket_coord, is_vertical);
  }

 private:
  Condition* _condition;

  // int _polygon_forward;
  // int _polygon_backward;
  DrcBasicRect _wire_region;
  bool _is_wire_vertical;
  std::vector<DrcBasicRect> _spacing_region_forward;
  std::vector<DrcBasicRect> _spacing_region_backward;
};
}  // namespace idrc