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

#include "Direction.hpp"
#include "EXTLayerRect.hpp"
#include "RTHeader.hpp"
#include "Utility.hpp"

namespace irt {

class DRPatch
{
 public:
  DRPatch() = default;
  DRPatch(const PlanarRect& planar_rect, const int32_t layer_idx)
  {
    _patch.set_real_rect(planar_rect);
    _patch.set_layer_idx(layer_idx);
  }
  ~DRPatch() = default;
  // getter
  EXTLayerRect& get_patch() { return _patch; }
  double get_fixed_rect_cost() const { return _fixed_rect_cost; }
  double get_routed_rect_cost() const { return _routed_rect_cost; }
  double get_violation_cost() const { return _violation_cost; }
  Direction get_direction() const { return _direction; }
  int32_t get_overlap_area() const { return _overlap_area; }
  // const getter
  const EXTLayerRect& get_patch() const { return _patch; }
  // setter
  void set_patch(const EXTLayerRect& patch) { _patch = patch; }
  void set_fixed_rect_cost(const double fixed_rect_cost) { _fixed_rect_cost = fixed_rect_cost; }
  void set_routed_rect_cost(const double routed_rect_cost) { _routed_rect_cost = routed_rect_cost; }
  void set_violation_cost(const double violation_cost) { _violation_cost = violation_cost; }
  void set_direction(const Direction& direction) { _direction = direction; }
  void set_overlap_area(const int32_t overlap_area) { _overlap_area = overlap_area; }
  // function
  double getTotalCost() { return (_fixed_rect_cost + _routed_rect_cost + _violation_cost); }

 private:
  EXTLayerRect _patch;
  double _fixed_rect_cost = 0.0;
  double _routed_rect_cost = 0.0;
  double _violation_cost = 0.0;
  Direction _direction = Direction::kNone;
  int32_t _overlap_area = 0;
};

struct CmpDRPatch
{
  bool operator()(const DRPatch& a, const DRPatch& b, Direction& layer_direction) const
  {
    SortStatus sort_status = SortStatus::kEqual;
    // fixed_rect_cost 大小升序
    if (sort_status == SortStatus::kEqual) {
      double a_fixed_rect_cost = a.get_fixed_rect_cost();
      double b_fixed_rect_cost = b.get_fixed_rect_cost();
      if (a_fixed_rect_cost < b_fixed_rect_cost) {
        sort_status = SortStatus::kTrue;
      } else if (a_fixed_rect_cost == b_fixed_rect_cost) {
        sort_status = SortStatus::kEqual;
      } else {
        sort_status = SortStatus::kFalse;
      }
    }
    // violation_cost 大小升序
    if (sort_status == SortStatus::kEqual) {
      double a_violation_cost = a.get_violation_cost();
      double b_violation_cost = b.get_violation_cost();
      if (a_violation_cost < b_violation_cost) {
        sort_status = SortStatus::kTrue;
      } else if (a_violation_cost == b_violation_cost) {
        sort_status = SortStatus::kEqual;
      } else {
        sort_status = SortStatus::kFalse;
      }
    }
    // routed_rect_cost 大小升序
    if (sort_status == SortStatus::kEqual) {
      double a_routed_rect_cost = a.get_routed_rect_cost();
      double b_routed_rect_cost = b.get_routed_rect_cost();
      if (a_routed_rect_cost < b_routed_rect_cost) {
        sort_status = SortStatus::kTrue;
      } else if (a_routed_rect_cost == b_routed_rect_cost) {
        sort_status = SortStatus::kEqual;
      } else {
        sort_status = SortStatus::kFalse;
      }
    }
    // 层方向优先
    if (sort_status == SortStatus::kEqual) {
      Direction a_direction = a.get_direction();
      Direction b_direction = b.get_direction();
      if (a_direction == layer_direction && b_direction != layer_direction) {
        sort_status = SortStatus::kTrue;
      } else if (a_direction != layer_direction && b_direction == layer_direction) {
        sort_status = SortStatus::kFalse;
      } else {
        sort_status = SortStatus::kEqual;
      }
    }
    // 重叠面积降序
    if (sort_status == SortStatus::kEqual) {
      int32_t a_overlap_area = a.get_overlap_area();
      int32_t b_overlap_area = b.get_overlap_area();
      if (a_overlap_area > b_overlap_area) {
        sort_status = SortStatus::kTrue;
      } else if (a_overlap_area == b_overlap_area) {
        sort_status = SortStatus::kEqual;
      } else {
        sort_status = SortStatus::kFalse;
      }
    }
    // real_rect比较
    if (sort_status == SortStatus::kEqual) {
      const PlanarRect& a_real_rect = a.get_patch().get_real_rect();
      const PlanarRect& b_real_rect = b.get_patch().get_real_rect();
      if (a_real_rect != b_real_rect) {
        if (CmpPlanarRectByXASC()(a_real_rect, b_real_rect)) {
          sort_status = SortStatus::kTrue;
        } else {
          sort_status = SortStatus::kFalse;
        }
      } else {
        sort_status = SortStatus::kEqual;
      }
    }
    // grid_rect比较
    if (sort_status == SortStatus::kEqual) {
      const PlanarRect& a_grid_rect = a.get_patch().get_grid_rect();
      const PlanarRect& b_grid_rect = b.get_patch().get_grid_rect();
      if (a_grid_rect != b_grid_rect) {
        if (CmpPlanarRectByXASC()(a_grid_rect, b_grid_rect)) {
          sort_status = SortStatus::kTrue;
        } else {
          sort_status = SortStatus::kFalse;
        }
      } else {
        sort_status = SortStatus::kEqual;
      }
    }
    if (sort_status == SortStatus::kTrue) {
      return true;
    } else if (sort_status == SortStatus::kFalse) {
      return false;
    }
    return false;
  }
};

}  // namespace irt
