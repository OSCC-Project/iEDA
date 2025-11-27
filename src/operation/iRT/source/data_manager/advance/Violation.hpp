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

#include "EXTLayerRect.hpp"
#include "RTHeader.hpp"
#include "SortStatus.hpp"
#include "ViolationType.hpp"

namespace irt {

class Violation
{
 public:
  Violation() = default;
  ~Violation() = default;
  // getter
  ViolationType get_violation_type() const { return _violation_type; }
  EXTLayerRect& get_violation_shape() { return _violation_shape; }
  bool get_is_routing() const { return _is_routing; }
  std::set<int32_t>& get_violation_net_set() { return _violation_net_set; }
  int32_t get_required_size() const { return _required_size; }
  // const getter
  const EXTLayerRect& get_violation_shape() const { return _violation_shape; }
  const std::set<int32_t>& get_violation_net_set() const { return _violation_net_set; }
  // setter
  void set_violation_type(const ViolationType& violation_type) { _violation_type = violation_type; }
  void set_violation_shape(const EXTLayerRect& violation_shape) { _violation_shape = violation_shape; }
  void set_is_routing(const bool is_routing) { _is_routing = is_routing; }
  void set_violation_net_set(const std::set<int32_t>& violation_net_set) { _violation_net_set = violation_net_set; }
  void set_required_size(const int32_t required_size) { _required_size = required_size; }
  // function

 private:
  ViolationType _violation_type = ViolationType::kNone;
  EXTLayerRect _violation_shape;
  bool _is_routing = true;
  std::set<int32_t> _violation_net_set;
  int32_t _required_size = 0;
};

struct CmpViolation
{
  bool operator()(const Violation& a, const Violation& b) const
  {
    SortStatus sort_status = SortStatus::kEqual;
    // 类型升序
    if (sort_status == SortStatus::kEqual) {
      ViolationType a_violation_type = a.get_violation_type();
      ViolationType b_violation_type = b.get_violation_type();
      if (a_violation_type < b_violation_type) {
        sort_status = SortStatus::kTrue;
      } else if (a_violation_type > b_violation_type) {
        sort_status = SortStatus::kFalse;
      } else {
        sort_status = SortStatus::kEqual;
      }
    }
    // real_rect比较
    if (sort_status == SortStatus::kEqual) {
      const PlanarRect& a_real_rect = a.get_violation_shape().get_real_rect();
      const PlanarRect& b_real_rect = b.get_violation_shape().get_real_rect();
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
      const PlanarRect& a_grid_rect = a.get_violation_shape().get_grid_rect();
      const PlanarRect& b_grid_rect = b.get_violation_shape().get_grid_rect();
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
    // layer_idx升序
    if (sort_status == SortStatus::kEqual) {
      int32_t a_layer_idx = a.get_violation_shape().get_layer_idx();
      int32_t b_layer_idx = b.get_violation_shape().get_layer_idx();
      if (a_layer_idx < b_layer_idx) {
        sort_status = SortStatus::kTrue;
      } else if (a_layer_idx > b_layer_idx) {
        sort_status = SortStatus::kFalse;
      } else {
        sort_status = SortStatus::kEqual;
      }
    }
    // is_routing升序
    if (sort_status == SortStatus::kEqual) {
      bool a_is_routing = a.get_is_routing();
      bool b_is_routing = b.get_is_routing();
      if (a_is_routing < b_is_routing) {
        sort_status = SortStatus::kTrue;
      } else if (a_is_routing > b_is_routing) {
        sort_status = SortStatus::kFalse;
      } else {
        sort_status = SortStatus::kEqual;
      }
    }
    // violation_net_set升序
    if (sort_status == SortStatus::kEqual) {
      const std::set<int32_t>& a_violation_net_set = a.get_violation_net_set();
      const std::set<int32_t>& b_violation_net_set = b.get_violation_net_set();
      if (a_violation_net_set < b_violation_net_set) {
        sort_status = SortStatus::kTrue;
      } else if (a_violation_net_set > b_violation_net_set) {
        sort_status = SortStatus::kFalse;
      } else {
        sort_status = SortStatus::kEqual;
      }
    }
    // required_size升序
    if (sort_status == SortStatus::kEqual) {
      int32_t a_required_size = a.get_required_size();
      int32_t b_required_size = b.get_required_size();
      if (a_required_size < b_required_size) {
        sort_status = SortStatus::kTrue;
      } else if (a_required_size > b_required_size) {
        sort_status = SortStatus::kFalse;
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

  bool operator()(Violation* a, Violation* b) const { return operator()(*a, *b); }
};

}  // namespace irt
