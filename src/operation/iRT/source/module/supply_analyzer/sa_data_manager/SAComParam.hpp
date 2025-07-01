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

class SAComParam
{
 public:
  SAComParam() = default;
  SAComParam(double supply_reduction, double boundary_wire_unit, double internal_wire_unit, double internal_via_unit)
  {
    _supply_reduction = supply_reduction;
    _boundary_wire_unit = boundary_wire_unit;
    _internal_wire_unit = internal_wire_unit;
    _internal_via_unit = internal_via_unit;
  }
  ~SAComParam() = default;
  // getter
  double get_supply_reduction() const { return _supply_reduction; }
  double get_boundary_wire_unit() const { return _boundary_wire_unit; }
  double get_internal_wire_unit() const { return _internal_wire_unit; }
  double get_internal_via_unit() const { return _internal_via_unit; }
  // setter
  void set_supply_reduction(const double supply_reduction) { _supply_reduction = supply_reduction; }
  void set_boundary_wire_unit(const double boundary_wire_unit) { _boundary_wire_unit = boundary_wire_unit; }
  void set_internal_wire_unit(const double internal_wire_unit) { _internal_wire_unit = internal_wire_unit; }
  void set_internal_via_unit(const double internal_via_unit) { _internal_via_unit = internal_via_unit; }

 private:
  double _supply_reduction = -1;
  double _boundary_wire_unit = -1;
  double _internal_wire_unit = -1;
  double _internal_via_unit = -1;
};

}  // namespace irt
