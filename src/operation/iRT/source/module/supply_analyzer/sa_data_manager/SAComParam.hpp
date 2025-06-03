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
  SAComParam(double supply_reduction, double wire_unit, double via_unit)
  {
    _supply_reduction = supply_reduction;
    _wire_unit = wire_unit;
    _via_unit = via_unit;
  }
  ~SAComParam() = default;
  // getter
  double get_supply_reduction() const { return _supply_reduction; }
  double get_wire_unit() const { return _wire_unit; }
  double get_via_unit() const { return _via_unit; }
  // setter
  void set_supply_reduction(const double supply_reduction) { _supply_reduction = supply_reduction; }
  void set_wire_unit(const double wire_unit) { _wire_unit = wire_unit; }
  void set_via_unit(const double via_unit) { _via_unit = via_unit; }

 private:
  double _supply_reduction = -1;
  double _wire_unit = -1;
  double _via_unit = -1;
};

}  // namespace irt
