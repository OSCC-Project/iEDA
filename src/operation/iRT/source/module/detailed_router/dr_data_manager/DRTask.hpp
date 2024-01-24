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
#include "DRGroup.hpp"
#include "LayerCoord.hpp"
#include "LayerRect.hpp"
#include "RoutingState.hpp"
#include "SortStatus.hpp"

namespace irt {

class DRTask
{
 public:
  DRTask() = default;
  ~DRTask() = default;
  // getter
  irt_int get_net_idx() { return _net_idx; }
  ConnectType& get_connect_type() { return _connect_type; }
  std::vector<DRGroup>& get_dr_group_list() { return _dr_group_list; }
  PlanarRect& get_bounding_box() { return _bounding_box; }
  irt_int get_routed_times() { return _routed_times; }
  std::vector<Segment<LayerCoord>>& get_routing_segment_list() { return _routing_segment_list; }
  std::vector<EXTLayerRect>& get_patch_list() { return _patch_list; }
  // const getter
  const ConnectType& get_connect_type() const { return _connect_type; }
  const std::vector<DRGroup>& get_dr_group_list() const { return _dr_group_list; }
  const PlanarRect& get_bounding_box() const { return _bounding_box; }
  // setter
  void set_net_idx(const irt_int net_idx) { _net_idx = net_idx; }
  void set_connect_type(const ConnectType& connect_type) { _connect_type = connect_type; }
  void set_dr_group_list(const std::vector<DRGroup>& dr_group_list) { _dr_group_list = dr_group_list; }
  void set_bounding_box(const PlanarRect& bounding_box) { _bounding_box = bounding_box; }
  void set_routed_times(const irt_int routed_times) { _routed_times = routed_times; }
  void set_routing_segment_list(const std::vector<Segment<LayerCoord>>& routing_segment_list)
  {
    _routing_segment_list = routing_segment_list;
  }
  void set_patch_list(const std::vector<EXTLayerRect>& patch_list) { _patch_list = patch_list; }
  // function
  void addRoutedTimes() { ++_routed_times; }

 private:
  irt_int _net_idx = -1;
  ConnectType _connect_type = ConnectType::kNone;
  std::vector<DRGroup> _dr_group_list;
  PlanarRect _bounding_box;
  irt_int _routed_times = 0;
  std::vector<Segment<LayerCoord>> _routing_segment_list;
  std::vector<EXTLayerRect> _patch_list;
};

struct CmpDRTask
{
  bool operator()(const DRTask* a, const DRTask* b) const
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
    // if (sort_status == SortStatus::kEqual) {
    //   double a_routing_area = a->get_bounding_box().getArea();
    //   double b_routing_area = b->get_bounding_box().getArea();

    //   if (a_routing_area < b_routing_area) {
    //     sort_status = SortStatus::kTrue;
    //   } else if (a_routing_area == b_routing_area) {
    //     sort_status = SortStatus::kEqual;
    //   } else {
    //     sort_status = SortStatus::kFalse;
    //   }
    // }

    // // PinNum 降序
    // if (sort_status == SortStatus::kEqual) {
    //   irt_int a_pin_num = static_cast<irt_int>(a->get_dr_group_list().size());
    //   irt_int b_pin_num = static_cast<irt_int>(b->get_dr_group_list().size());

    //   if (a_pin_num > b_pin_num) {
    //     sort_status = SortStatus::kTrue;
    //   } else if (a_pin_num == b_pin_num) {
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
