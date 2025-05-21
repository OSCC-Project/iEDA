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

#include "ConnectType.hpp"
#include "LayerCoord.hpp"
#include "LayerRect.hpp"
#include "SRGroup.hpp"
#include "SortStatus.hpp"

namespace irt {

class SRTask
{
 public:
  SRTask() = default;
  ~SRTask() = default;
  // getter
  int32_t get_net_idx() { return _net_idx; }
  ConnectType& get_connect_type() { return _connect_type; }
  std::vector<SRGroup>& get_sr_group_list() { return _sr_group_list; }
  PlanarRect& get_bounding_box() { return _bounding_box; }
  int32_t get_routed_times() { return _routed_times; }
  // const getter
  const ConnectType& get_connect_type() const { return _connect_type; }
  const std::vector<SRGroup>& get_sr_group_list() const { return _sr_group_list; }
  const PlanarRect& get_bounding_box() const { return _bounding_box; }
  // setter
  void set_net_idx(const int32_t net_idx) { _net_idx = net_idx; }
  void set_connect_type(const ConnectType& connect_type) { _connect_type = connect_type; }
  void set_sr_group_list(const std::vector<SRGroup>& sr_group_list) { _sr_group_list = sr_group_list; }
  void set_bounding_box(const PlanarRect& bounding_box) { _bounding_box = bounding_box; }
  void set_routed_times(const int32_t routed_times) { _routed_times = routed_times; }
  // function
  void addRoutedTimes() { ++_routed_times; }

 private:
  int32_t _net_idx = -1;
  ConnectType _connect_type = ConnectType::kNone;
  std::vector<SRGroup> _sr_group_list;
  PlanarRect _bounding_box;
  int32_t _routed_times = 0;
};

struct CmpSRTask
{
  bool operator()(const SRTask* a, const SRTask* b) const
  {
    SortStatus sort_status = SortStatus::kEqual;
    // 时钟线网优先
    if (sort_status == SortStatus::kEqual) {
      ConnectType a_connect_type = a->get_connect_type();
      ConnectType b_connect_type = b->get_connect_type();
      if (a_connect_type == ConnectType::kClock && b_connect_type != ConnectType::kClock) {
        sort_status = SortStatus::kTrue;
      } else if (a_connect_type != ConnectType::kClock && b_connect_type == ConnectType::kClock) {
        sort_status = SortStatus::kFalse;
      } else {
        sort_status = SortStatus::kEqual;
      }
    }
    // BoundingBox 大小升序
    if (sort_status == SortStatus::kEqual) {
      double a_routing_area = a->get_bounding_box().getArea();
      double b_routing_area = b->get_bounding_box().getArea();
      if (a_routing_area < b_routing_area) {
        sort_status = SortStatus::kTrue;
      } else if (a_routing_area == b_routing_area) {
        sort_status = SortStatus::kEqual;
      } else {
        sort_status = SortStatus::kFalse;
      }
    }
    // PinNum 降序
    if (sort_status == SortStatus::kEqual) {
      int32_t a_pin_num = static_cast<int32_t>(a->get_sr_group_list().size());
      int32_t b_pin_num = static_cast<int32_t>(b->get_sr_group_list().size());
      if (a_pin_num > b_pin_num) {
        sort_status = SortStatus::kTrue;
      } else if (a_pin_num == b_pin_num) {
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
