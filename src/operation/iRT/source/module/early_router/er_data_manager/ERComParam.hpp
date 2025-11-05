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

#include "RTHeader.hpp"

namespace irt {

class ERComParam
{
 public:
  ERComParam() = default;
  ERComParam(std::string resolve_congestion, int32_t max_candidate_point_num, int32_t supply_reduction, double boundary_wire_unit, double internal_wire_unit,
             double internal_via_unit, int32_t expand_step_num, int32_t expand_step_length, double via_unit, double overflow_unit, int32_t schedule_interval)
  {
    _resolve_congestion = resolve_congestion;
    _max_candidate_point_num = max_candidate_point_num;
    _supply_reduction = supply_reduction;
    _boundary_wire_unit = boundary_wire_unit;
    _internal_wire_unit = internal_wire_unit;
    _internal_via_unit = internal_via_unit;
    _expand_step_num = expand_step_num;
    _expand_step_length = expand_step_length;
    _via_unit = via_unit;
    _overflow_unit = overflow_unit;
    _schedule_interval = schedule_interval;
  }
  ~ERComParam() = default;
  // getter
  std::string& get_resolve_congestion() { return _resolve_congestion; }
  int32_t get_max_candidate_point_num() const { return _max_candidate_point_num; }
  int32_t get_supply_reduction() const { return _supply_reduction; }
  double get_boundary_wire_unit() const { return _boundary_wire_unit; }
  double get_internal_wire_unit() const { return _internal_wire_unit; }
  double get_internal_via_unit() const { return _internal_via_unit; }
  int32_t get_expand_step_num() const { return _expand_step_num; }
  int32_t get_expand_step_length() const { return _expand_step_length; }
  double get_via_unit() const { return _via_unit; }
  double get_overflow_unit() const { return _overflow_unit; }
  double get_schedule_interval() const { return _schedule_interval; }
  // setter
  void set_resolve_congestion(std::string& resolve_congestion) { _resolve_congestion = resolve_congestion; }
  void set_max_candidate_point_num(const int32_t max_candidate_point_num) { _max_candidate_point_num = max_candidate_point_num; }
  void set_supply_reduction(const int32_t supply_reduction) { _supply_reduction = supply_reduction; }
  void set_boundary_wire_unit(const double boundary_wire_unit) { _boundary_wire_unit = boundary_wire_unit; }
  void set_internal_wire_unit(const double internal_wire_unit) { _internal_wire_unit = internal_wire_unit; }
  void set_internal_via_unit(const double internal_via_unit) { _internal_via_unit = internal_via_unit; }
  void set_expand_step_num(const int32_t expand_step_num) { _expand_step_num = expand_step_num; }
  void set_expand_step_length(const int32_t expand_step_length) { _expand_step_length = expand_step_length; }
  void set_via_unit(const double via_unit) { _via_unit = via_unit; }
  void set_overflow_unit(const double overflow_unit) { _overflow_unit = overflow_unit; }
  void set_schedule_interval(const double schedule_interval) { _schedule_interval = schedule_interval; }

 private:
  std::string _resolve_congestion;
  int32_t _max_candidate_point_num = -1;
  int32_t _supply_reduction = -1;
  double _boundary_wire_unit = -1;
  double _internal_wire_unit = -1;
  double _internal_via_unit = -1;
  int32_t _expand_step_num = 0;
  int32_t _expand_step_length = 0;
  double _via_unit = 0;
  double _overflow_unit = 0;
  int32_t _schedule_interval = 0;
};

}  // namespace irt
