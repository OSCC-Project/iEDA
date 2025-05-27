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

#include "PlanarRect.hpp"

namespace irt {

class PAShadow
{
 public:
  PAShadow() = default;
  ~PAShadow() = default;
  // getter
  std::map<int32_t, std::set<PlanarRect, CmpPlanarRectByXASC>>& get_net_fixed_rect_map() { return _net_fixed_rect_map; }
  std::map<int32_t, std::set<PlanarRect, CmpPlanarRectByXASC>>& get_net_routed_rect_map() { return _net_routed_rect_map; }
  std::set<PlanarRect, CmpPlanarRectByXASC>& get_violation_set() { return _violation_set; }
  // setter
  void set_net_fixed_rect_map(const std::map<int32_t, std::set<PlanarRect, CmpPlanarRectByXASC>>& net_fixed_rect_map)
  {
    _net_fixed_rect_map = net_fixed_rect_map;
  }
  void set_net_routed_rect_map(const std::map<int32_t, std::set<PlanarRect, CmpPlanarRectByXASC>>& net_routed_rect_map)
  {
    _net_routed_rect_map = net_routed_rect_map;
  }
  void set_violation_set(const std::set<PlanarRect, CmpPlanarRectByXASC>& violation_set) { _violation_set = violation_set; }
  // function

 private:
  std::map<int32_t, std::set<PlanarRect, CmpPlanarRectByXASC>> _net_fixed_rect_map;
  std::map<int32_t, std::set<PlanarRect, CmpPlanarRectByXASC>> _net_routed_rect_map;
  std::set<PlanarRect, CmpPlanarRectByXASC> _violation_set;
};

}  // namespace irt
