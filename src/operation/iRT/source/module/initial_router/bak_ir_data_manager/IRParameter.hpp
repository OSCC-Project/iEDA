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

#include "ConnectType.hpp"
#include "DRBox.hpp"
#include "DRPin.hpp"
#include "GridMap.hpp"
#include "Guide.hpp"
#include "MTree.hpp"
#include "Net.hpp"
#include "PhysicalNode.hpp"
#include "Pin.hpp"
#include "TNode.hpp"

namespace irt {

class IRParameter
{
 public:
  IRParameter() = default;
  IRParameter(irt_int prefer_wire_unit, irt_int via_unit, irt_int corner_unit)
  {
    _prefer_wire_unit = prefer_wire_unit;
    _via_unit = via_unit;
    _corner_unit = corner_unit;
  }
  ~IRParameter() = default;
  // getter
  irt_int get_prefer_wire_unit() const { return _prefer_wire_unit; }
  irt_int get_via_unit() const { return _via_unit; }
  irt_int get_corner_unit() const { return _corner_unit; }
  // setter
  void set_prefer_wire_unit(const irt_int prefer_wire_unit) { _prefer_wire_unit = prefer_wire_unit; }
  void set_via_unit(const irt_int via_unit) { _via_unit = via_unit; }
  void set_corner_unit(const irt_int corner_unit) { _corner_unit = corner_unit; }

 private:
  double _prefer_wire_unit = 1;
  double _via_unit = 1;
  double _corner_unit = 1;
};

}  // namespace irt
