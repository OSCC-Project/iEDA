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

#include "LayerRect.hpp"
#include "RTU.hpp"
#include "RTUtil.hpp"

namespace irt {

class ViolationInfo
{
 public:
  ViolationInfo() = default;
  ~ViolationInfo() = default;
  // getter
  bool get_is_routing() { return _is_routing; }
  std::string get_rule_name() { return _rule_name; }
  LayerRect& get_violation_region() { return _violation_region; }
  std::map<irt_int, std::vector<LayerRect>>& get_net_shape_map() { return _net_shape_map; }
  // setter
  void set_is_routing(const bool is_routing) { _is_routing = is_routing; }
  void set_rule_name(const std::string& rule_name) { _rule_name = rule_name; }
  void set_violation_region(const LayerRect& violation_region) { _violation_region = violation_region; }
  void set_net_shape_map(const std::map<irt_int, std::vector<LayerRect>>& net_shape_map) { _net_shape_map = net_shape_map; }
  // function

 private:
  bool _is_routing = true;
  std::string _rule_name;
  LayerRect _violation_region;
  std::map<irt_int, std::vector<LayerRect>> _net_shape_map;
};

}  // namespace irt
