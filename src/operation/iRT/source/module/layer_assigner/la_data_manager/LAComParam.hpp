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

class LAComParam
{
 public:
  LAComParam() = default;
  LAComParam(int32_t topo_spilt_length, double prefer_wire_unit, double via_unit, double overflow_unit)
  {
    _topo_spilt_length = topo_spilt_length;
    _prefer_wire_unit = prefer_wire_unit;
    _via_unit = via_unit;
    _overflow_unit = overflow_unit;
  }
  ~LAComParam() = default;
  // getter
  int32_t get_topo_spilt_length() const { return _topo_spilt_length; }
  double get_prefer_wire_unit() const { return _prefer_wire_unit; }
  double get_via_unit() const { return _via_unit; }
  double get_overflow_unit() const { return _overflow_unit; }
  // setter
  void set_topo_spilt_length(const int32_t topo_spilt_length) { _topo_spilt_length = topo_spilt_length; }
  void set_prefer_wire_unit(const double prefer_wire_unit) { _prefer_wire_unit = prefer_wire_unit; }
  void set_via_unit(const double via_unit) { _via_unit = via_unit; }
  void set_overflow_unit(const double overflow_unit) { _overflow_unit = overflow_unit; }

 private:
  int32_t _topo_spilt_length = 0;
  double _prefer_wire_unit = 0;
  double _via_unit = 0;
  double _overflow_unit = 0;
};

}  // namespace irt
