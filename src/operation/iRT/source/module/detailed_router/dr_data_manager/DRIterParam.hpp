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

class DRIterParam
{
 public:
  DRIterParam() = default;
  DRIterParam(double prefer_wire_unit, double non_prefer_wire_unit, double bend_unit, double via_unit, int32_t size, int32_t offset, int32_t schedule_interval,
              double fixed_rect_unit, double routed_rect_unit, double violation_unit, int32_t max_routed_times, int32_t max_candidate_patch_num)
  {
    _prefer_wire_unit = prefer_wire_unit;
    _non_prefer_wire_unit = non_prefer_wire_unit;
    _bend_unit = bend_unit;
    _via_unit = via_unit;
    _size = size;
    _offset = offset;
    _schedule_interval = schedule_interval;
    _fixed_rect_unit = fixed_rect_unit;
    _routed_rect_unit = routed_rect_unit;
    _violation_unit = violation_unit;
    _max_routed_times = max_routed_times;
    _max_candidate_patch_num = max_candidate_patch_num;
  }
  ~DRIterParam() = default;
  // getter
  double get_prefer_wire_unit() const { return _prefer_wire_unit; }
  double get_non_prefer_wire_unit() const { return _non_prefer_wire_unit; }
  double get_bend_unit() const { return _bend_unit; }
  double get_via_unit() const { return _via_unit; }
  int32_t get_size() const { return _size; }
  int32_t get_offset() const { return _offset; }
  int32_t get_schedule_interval() const { return _schedule_interval; }
  double get_fixed_rect_unit() const { return _fixed_rect_unit; }
  double get_routed_rect_unit() const { return _routed_rect_unit; }
  double get_violation_unit() const { return _violation_unit; }
  int32_t get_max_routed_times() const { return _max_routed_times; }
  int32_t get_max_candidate_patch_num() const { return _max_candidate_patch_num; }
  // setter
  void set_prefer_wire_unit(const double prefer_wire_unit) { _prefer_wire_unit = prefer_wire_unit; }
  void set_non_prefer_wire_unit(const double non_prefer_wire_unit) { _non_prefer_wire_unit = non_prefer_wire_unit; }
  void set_bend_unit(const double bend_unit) { _bend_unit = bend_unit; }
  void set_via_unit(const double via_unit) { _via_unit = via_unit; }
  void set_size(const int32_t size) { _size = size; }
  void set_offset(const int32_t offset) { _offset = offset; }
  void set_schedule_interval(const int32_t schedule_interval) { _schedule_interval = schedule_interval; }
  void set_fixed_rect_unit(const double fixed_rect_unit) { _fixed_rect_unit = fixed_rect_unit; }
  void set_routed_rect_unit(const double routed_rect_unit) { _routed_rect_unit = routed_rect_unit; }
  void set_violation_unit(const double violation_unit) { _violation_unit = violation_unit; }
  void set_max_routed_times(const int32_t max_routed_times) { _max_routed_times = max_routed_times; }
  void set_max_candidate_patch_num(const int32_t max_candidate_patch_num) { _max_candidate_patch_num = max_candidate_patch_num; }

 private:
  double _prefer_wire_unit = 0;
  double _non_prefer_wire_unit = 0;
  double _bend_unit = 0;
  double _via_unit = 0;
  int32_t _size = -1;
  int32_t _offset = -1;
  int32_t _schedule_interval = -1;
  double _fixed_rect_unit = 0;
  double _routed_rect_unit = 0;
  double _violation_unit = 0;
  int32_t _max_routed_times = 0;
  int32_t _max_candidate_patch_num = 0;
};

}  // namespace irt
