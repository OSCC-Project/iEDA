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
  bool get_is_routing() { return _is_routing; }
  std::set<int32_t>& get_violation_net_set() { return _violation_net_set; }
  // setter
  void set_violation_type(const ViolationType& violation_type) { _violation_type = violation_type; }
  void set_violation_shape(const EXTLayerRect& violation_shape) { _violation_shape = violation_shape; }
  void set_is_routing(const bool is_routing) { _is_routing = is_routing; }
  void set_violation_net_set(const std::set<int32_t>& violation_net_set) { _violation_net_set = violation_net_set; }
  // function

 private:
  ViolationType _violation_type;
  EXTLayerRect _violation_shape;
  bool _is_routing = true;
  std::set<int32_t> _violation_net_set;
};

}  // namespace irt
