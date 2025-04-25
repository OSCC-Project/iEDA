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
#include "PPObsType.hpp"
#include "RTHeader.hpp"
#include "Utility.hpp"

namespace irt {

class PPNode : public LayerCoord
{
 public:
  PPNode() = default;
  ~PPNode() = default;
  // getter
  std::map<PPObsType, std::set<int32_t>>& get_obs_type_fixed_rect_map() { return _obs_type_fixed_rect_map; }
  std::map<PPObsType, std::set<int32_t>>& get_obs_type_routed_rect_map() { return _obs_type_routed_rect_map; }
  std::map<PPObsType, int32_t>& get_obs_type_violation_number_map() { return _obs_type_violation_number_map; }
  // setter
  void set_obs_type_fixed_rect_map(const std::map<PPObsType, std::set<int32_t>>& obs_type_fixed_rect_map)
  {
    _obs_type_fixed_rect_map = obs_type_fixed_rect_map;
  }
  void set_obs_type_routed_rect_map(const std::map<PPObsType, std::set<int32_t>>& obs_type_routed_rect_map)
  {
    _obs_type_routed_rect_map = obs_type_routed_rect_map;
  }
  void set_obs_type_violation_number_map(const std::map<PPObsType, int32_t>& obs_type_violation_number_map)
  {
    _obs_type_violation_number_map = obs_type_violation_number_map;
  }
  // function
  double getFixedRectCost(int32_t net_idx, PPObsType pp_obs_type, double fixed_rect_unit)
  {
    int32_t fixed_rect_num = 0;
    if (RTUTIL.exist(_obs_type_fixed_rect_map, pp_obs_type)) {
      std::set<int32_t>& net_set = _obs_type_fixed_rect_map[pp_obs_type];
      fixed_rect_num = static_cast<int32_t>(net_set.size());
      if (RTUTIL.exist(net_set, net_idx)) {
        fixed_rect_num--;
      }
      if (fixed_rect_num < 0) {
        RTLOG.error(Loc::current(), "The fixed_rect_num < 0!");
      }
    }
    double cost = 0;
    if (fixed_rect_num > 0) {
      cost = fixed_rect_unit;
    }
    return cost;
  }
  double getRoutedRectCost(int32_t net_idx, PPObsType pp_obs_type, double routed_rect_unit)
  {
    int32_t routed_rect_num = 0;
    if (RTUTIL.exist(_obs_type_routed_rect_map, pp_obs_type)) {
      std::set<int32_t>& net_set = _obs_type_routed_rect_map[pp_obs_type];
      routed_rect_num = static_cast<int32_t>(net_set.size());
      if (RTUTIL.exist(net_set, net_idx)) {
        routed_rect_num--;
      }
      if (routed_rect_num < 0) {
        RTLOG.error(Loc::current(), "The routed_rect_num < 0!");
      }
    }
    double cost = 0;
    if (routed_rect_num > 0) {
      cost = routed_rect_unit;
    }
    return cost;
  }
  double getViolationCost(PPObsType pp_obs_type, double violation_unit)
  {
    int32_t violation_num = 0;
    if (RTUTIL.exist(_obs_type_violation_number_map, pp_obs_type)) {
      violation_num = _obs_type_violation_number_map[pp_obs_type];
    }
    double cost = 0;
    if (violation_num > 0) {
      cost = violation_unit;
    }
    return cost;
  }

 private:
  // obstacle & pin_shape
  std::map<PPObsType, std::set<int32_t>> _obs_type_fixed_rect_map;
  // net_result
  std::map<PPObsType, std::set<int32_t>> _obs_type_routed_rect_map;
  // violation
  std::map<PPObsType, int32_t> _obs_type_violation_number_map;
};

}  // namespace irt
