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

class TAComParam
{
 public:
  TAComParam() = default;
  TAComParam(double prefer_wire_unit, int32_t schedule_interval, double fixed_rect_unit, double routed_rect_unit, double violation_unit,
             int32_t max_routed_times)
  {
    _prefer_wire_unit = prefer_wire_unit;
    _schedule_interval = schedule_interval;
    _fixed_rect_unit = fixed_rect_unit;
    _routed_rect_unit = routed_rect_unit;
    _violation_unit = violation_unit;
    _max_routed_times = max_routed_times;
  }
  ~TAComParam() = default;
  // getter
  double get_prefer_wire_unit() const { return _prefer_wire_unit; }
  int32_t get_schedule_interval() const { return _schedule_interval; }
  double get_fixed_rect_unit() const { return _fixed_rect_unit; }
  double get_routed_rect_unit() const { return _routed_rect_unit; }
  double get_violation_unit() const { return _violation_unit; }
  int32_t get_max_routed_times() const { return _max_routed_times; }
  // setter
  void set_prefer_wire_unit(const double prefer_wire_unit) { _prefer_wire_unit = prefer_wire_unit; }
  void set_schedule_interval(const int32_t schedule_interval) { _schedule_interval = schedule_interval; }
  void set_fixed_rect_unit(const double fixed_rect_unit) { _fixed_rect_unit = fixed_rect_unit; }
  void set_routed_rect_unit(const double routed_rect_unit) { _routed_rect_unit = routed_rect_unit; }
  void set_violation_unit(const double violation_unit) { _violation_unit = violation_unit; }
  void set_max_routed_times(const int32_t max_routed_times) { _max_routed_times = max_routed_times; }

 private:
  double _prefer_wire_unit = 0;
  int32_t _schedule_interval = -1;
  double _fixed_rect_unit = 0;
  double _routed_rect_unit = 0;
  double _violation_unit = 0;
  int32_t _max_routed_times = 0;
};

}  // namespace irt
