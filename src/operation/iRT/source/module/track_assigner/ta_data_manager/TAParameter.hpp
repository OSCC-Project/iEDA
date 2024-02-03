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

namespace irt {

class TAParameter
{
 public:
  TAParameter() = default;
  TAParameter(double fixed_rect_cost, double routed_rect_cost, double violation_cost)
  {
    _prefer_wire_unit = 1;
    _nonprefer_wire_unit = 2;
    _corner_unit = 1;
    _fixed_rect_cost = fixed_rect_cost;
    _routed_rect_cost = routed_rect_cost;
    _violation_cost = violation_cost;
  }
  ~TAParameter() = default;
  // getter
  double get_prefer_wire_unit() const { return _prefer_wire_unit; }
  double get_nonprefer_wire_unit() const { return _nonprefer_wire_unit; }
  double get_corner_unit() const { return _corner_unit; }
  double get_fixed_rect_cost() const { return _fixed_rect_cost; }
  double get_routed_rect_cost() const { return _routed_rect_cost; }
  double get_violation_cost() const { return _violation_cost; }
  // setter
  void set_prefer_wire_unit(const double prefer_wire_unit) { _prefer_wire_unit = prefer_wire_unit; }
  void set_nonprefer_wire_unit(const double nonprefer_wire_unit) { _nonprefer_wire_unit = nonprefer_wire_unit; }
  void set_corner_unit(const double corner_unit) { _corner_unit = corner_unit; }
  void set_fixed_rect_cost(const double fixed_rect_cost) { _fixed_rect_cost = fixed_rect_cost; }
  void set_routed_rect_cost(const double routed_rect_cost) { _routed_rect_cost = routed_rect_cost; }
  void set_violation_cost(const double violation_cost) { _violation_cost = violation_cost; }

 private:
  double _prefer_wire_unit = 0;
  double _nonprefer_wire_unit = 0;
  double _corner_unit = 0;
  double _fixed_rect_cost = 0;
  double _routed_rect_cost = 0;
  double _violation_cost = 0;
};

}  // namespace irt
