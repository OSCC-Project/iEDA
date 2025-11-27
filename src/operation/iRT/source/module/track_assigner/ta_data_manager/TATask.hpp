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
#include "TAGroup.hpp"

namespace irt {

class TATask
{
 public:
  TATask() = default;
  ~TATask() = default;
  // getter
  int32_t get_net_idx() { return _net_idx; }
  int32_t get_task_idx() { return _task_idx; }
  ConnectType& get_connect_type() { return _connect_type; }
  std::vector<TAGroup>& get_ta_group_list() { return _ta_group_list; }
  PlanarRect& get_bounding_box() { return _bounding_box; }
  int32_t get_routed_times() const { return _routed_times; }
  // const getter
  const int32_t get_net_idx() const { return _net_idx; }
  const ConnectType& get_connect_type() const { return _connect_type; }
  const std::vector<TAGroup>& get_ta_group_list() const { return _ta_group_list; }
  const PlanarRect& get_bounding_box() const { return _bounding_box; }
  // setter
  void set_net_idx(const int32_t net_idx) { _net_idx = net_idx; }
  void set_task_idx(const int32_t task_idx) { _task_idx = task_idx; }
  void set_connect_type(const ConnectType& connect_type) { _connect_type = connect_type; }
  void set_ta_group_list(const std::vector<TAGroup>& ta_group_list) { _ta_group_list = ta_group_list; }
  void set_bounding_box(const PlanarRect& bounding_box) { _bounding_box = bounding_box; }
  void set_routed_times(const int32_t routed_times) { _routed_times = routed_times; }
  // function
  void addRoutedTimes() { ++_routed_times; }

 private:
  int32_t _net_idx = -1;
  int32_t _task_idx = -1;
  ConnectType _connect_type = ConnectType::kNone;
  std::vector<TAGroup> _ta_group_list;
  PlanarRect _bounding_box;
  int32_t _routed_times = 0;
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
    if (sort_status == SortStatus::kEqual) {
      double a_routing_area = a->get_bounding_box().getArea();
      double b_routing_area = b->get_bounding_box().getArea();
      if (a_routing_area > b_routing_area) {
        sort_status = SortStatus::kTrue;
      } else if (a_routing_area == b_routing_area) {
        sort_status = SortStatus::kEqual;
      } else {
        sort_status = SortStatus::kFalse;
      }
    }

    if (sort_status == SortStatus::kEqual) {
      int32_t a_net_idx = a->get_net_idx();
      int32_t b_net_idx = b->get_net_idx();
      if (a_net_idx < b_net_idx) {
        sort_status = SortStatus::kTrue;
      } else if (a_net_idx == b_net_idx) {
        sort_status = SortStatus::kEqual;
      } else {
        sort_status = SortStatus::kFalse;
      }
    }

    if (sort_status == SortStatus::kEqual) {
      std::vector<LayerCoord> a_coord_list;
      std::vector<LayerCoord> b_coord_list;
      for (const TAGroup& ta_group : a->get_ta_group_list()) {
        for (const LayerCoord& coord : ta_group.get_coord_list()) {
          a_coord_list.push_back(coord);
        }
      }
      for (const TAGroup& ta_group : b->get_ta_group_list()) {
        for (const LayerCoord& coord : ta_group.get_coord_list()) {
          b_coord_list.push_back(coord);
        }
      }
      std::sort(a_coord_list.begin(), a_coord_list.end(), CmpLayerCoordByXASC());
      std::sort(b_coord_list.begin(), b_coord_list.end(), CmpLayerCoordByXASC());

      if (std::lexicographical_compare(a_coord_list.begin(), a_coord_list.end(), b_coord_list.begin(), b_coord_list.end(), CmpLayerCoordByXASC())) {
        sort_status = SortStatus::kTrue;
      } else if (a_coord_list == b_coord_list) {
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
