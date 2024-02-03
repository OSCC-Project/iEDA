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

#include "GuideSeg.hpp"
#include "LayerCoord.hpp"
#include "LayerRect.hpp"
#include "RoutingState.hpp"
#include "TAGroup.hpp"

namespace irt {

class TATask
{
 public:
  TATask() = default;
  ~TATask() = default;
  // getter
  irt_int get_net_idx() { return _net_idx; }
  irt_int get_task_idx() { return _task_idx; }
  ConnectType& get_connect_type() { return _connect_type; }
  std::vector<TAGroup>& get_ta_group_list() { return _ta_group_list; }
  PlanarRect& get_bounding_box() { return _bounding_box; }
  irt_int get_routed_times() const { return _routed_times; }
  std::vector<Segment<LayerCoord>>& get_routing_segment_list() { return _routing_segment_list; }
  // const getter
  const ConnectType& get_connect_type() const { return _connect_type; }
  const PlanarRect& get_bounding_box() const { return _bounding_box; }
  // setter
  void set_net_idx(const irt_int net_idx) { _net_idx = net_idx; }
  void set_task_idx(const irt_int task_idx) { _task_idx = task_idx; }
  void set_connect_type(const ConnectType& connect_type) { _connect_type = connect_type; }
  void set_ta_group_list(const std::vector<TAGroup>& ta_group_list) { _ta_group_list = ta_group_list; }
  void set_bounding_box(const PlanarRect& bounding_box) { _bounding_box = bounding_box; }
  void set_routed_times(const irt_int routed_times) { _routed_times = routed_times; }
  void set_routing_segment_list(const std::vector<Segment<LayerCoord>>& routing_segment_list)
  {
    _routing_segment_list = routing_segment_list;
  }
  // function
  void addRoutedTimes() { ++_routed_times; }

 private:
  irt_int _net_idx = -1;
  irt_int _task_idx = -1;
  ConnectType _connect_type = ConnectType::kNone;
  std::vector<TAGroup> _ta_group_list;

  PlanarRect _bounding_box;
  irt_int _routed_times = 0;
  std::vector<Segment<LayerCoord>> _routing_segment_list;
};

struct CmpTATask
{
  bool operator()(const TATask* a, const TATask* b) const
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

    // BoundingBox 大小降序
    // if (sort_status == SortStatus::kEqual) {
    //   double a_routing_area = a->get_bounding_box().getArea();
    //   double b_routing_area = b->get_bounding_box().getArea();

    //   if (a_routing_area > b_routing_area) {
    //     sort_status = SortStatus::kTrue;
    //   } else if (a_routing_area == b_routing_area) {
    //     sort_status = SortStatus::kEqual;
    //   } else {
    //     sort_status = SortStatus::kFalse;
    //   }
    // }

    if (sort_status == SortStatus::kTrue) {
      return true;
    }
    return false;
  }
};

}  // namespace irt
