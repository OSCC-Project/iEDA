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
#include "LayerCoord.hpp"
#include "RTHeader.hpp"
#include "Utility.hpp"

namespace irt {

class DPPatch
{
 public:
  DPPatch() = default;
  DPPatch(const PlanarRect& planar_rect, const int32_t layer_idx)
  {
    _patch.set_real_rect(planar_rect);
    _patch.set_layer_idx(layer_idx);
  }
  ~DPPatch() = default;
  // getter
  EXTLayerRect& get_patch() { return _patch; }
  Direction get_direction() const { return _direction; }
  int32_t get_overlap_area() const { return _overlap_area; }
  double get_env_cost() const { return _env_cost; }
  // setter
  void set_patch(const EXTLayerRect& patch) { _patch = patch; }
  void set_direction(const Direction& direction) { _direction = direction; }
  void set_overlap_area(const int32_t overlap_area) { _overlap_area = overlap_area; }
  void set_env_cost(const double env_cost) { _env_cost = env_cost; }
  // function

 private:
  EXTLayerRect _patch;
  Direction _direction = Direction::kNone;
  int32_t _overlap_area = 0;
  double _env_cost = 0.0;
};

struct CmpDPPatch
{
  bool operator()(const DPPatch& a, const DPPatch& b, Direction& layer_direction) const
  {
    SortStatus sort_status = SortStatus::kEqual;
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
    // env_cost 大小升序
    if (sort_status == SortStatus::kEqual) {
      double a_env_cost = a.get_env_cost();
      double b_env_cost = b.get_env_cost();
      if (a_env_cost < b_env_cost) {
        sort_status = SortStatus::kTrue;
      } else if (a_env_cost == b_env_cost) {
        sort_status = SortStatus::kEqual;
      } else {
        sort_status = SortStatus::kFalse;
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
