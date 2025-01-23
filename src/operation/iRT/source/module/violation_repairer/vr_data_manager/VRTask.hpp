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
#include "SortStatus.hpp"
#include "VRGroup.hpp"

namespace irt {

class VRTask
{
 public:
  VRTask() = default;
  ~VRTask() = default;
  // getter
  int32_t get_net_idx() { return _net_idx; }
  ConnectType& get_connect_type() { return _connect_type; }
  int32_t get_routed_times() { return _routed_times; }
  // const getter
  const ConnectType& get_connect_type() const { return _connect_type; }
  // setter
  void set_net_idx(const int32_t net_idx) { _net_idx = net_idx; }
  void set_connect_type(const ConnectType& connect_type) { _connect_type = connect_type; }
  void set_routed_times(const int32_t routed_times) { _routed_times = routed_times; }
  // function
  void addRoutedTimes() { ++_routed_times; }

 private:
  int32_t _net_idx = -1;
  ConnectType _connect_type = ConnectType::kNone;
  int32_t _routed_times = 0;
};

struct CmpVRTask
{
  bool operator()(const VRTask* a, const VRTask* b) const
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
    if (sort_status == SortStatus::kTrue) {
      return true;
    } else if (sort_status == SortStatus::kFalse) {
      return false;
    }
    return false;
  }
};

}  // namespace irt
