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

class IRParameter
{
 public:
  IRParameter()
  {
    _topo_spilt_length = 10;
    _congestion_unit = 2;
    _prefer_wire_unit = 1;
    _nonprefer_wire_unit = 2;
    _via_unit = 1;
    _corner_unit = 1;
  }
  ~IRParameter() = default;
  // getter
  int32_t get_topo_spilt_length() const { return _topo_spilt_length; }
  double get_congestion_unit() const { return _congestion_unit; }
  double get_prefer_wire_unit() const { return _prefer_wire_unit; }
  double get_nonprefer_wire_unit() const { return _nonprefer_wire_unit; }
  double get_via_unit() const { return _via_unit; }
  double get_corner_unit() const { return _corner_unit; }
  // setter
  void set_topo_spilt_length(const int32_t topo_spilt_length) { _topo_spilt_length = topo_spilt_length; }
  void set_congestion_unit(const double congestion_unit) { _congestion_unit = congestion_unit; }
  void set_prefer_wire_unit(const double prefer_wire_unit) { _prefer_wire_unit = prefer_wire_unit; }
  void set_nonprefer_wire_unit(const double nonprefer_wire_unit) { _nonprefer_wire_unit = nonprefer_wire_unit; }
  void set_via_unit(const double via_unit) { _via_unit = via_unit; }
  void set_corner_unit(const double corner_unit) { _corner_unit = corner_unit; }

 private:
  int32_t _topo_spilt_length = 0;
  double _congestion_unit = 0;
  double _prefer_wire_unit = 0;
  double _nonprefer_wire_unit = 0;
  double _via_unit = 0;
  double _corner_unit = 0;
};

}  // namespace irt
