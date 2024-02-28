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
#include "Pin.hpp"
#include "TNode.hpp"

namespace irt {

class DRParameter
{
 public:
  DRParameter() = default;
  DRParameter(int32_t size, int32_t offset, int32_t fixed_rect_unit, int32_t routed_rect_unit, int32_t violation_unit, bool complete_ripup)
  {
    _prefer_wire_unit = 1;
    _nonprefer_wire_unit = 2;
    _via_unit = 1;
    _corner_unit = 1;
    _size = size;
    _offset = offset;
    _fixed_rect_unit = fixed_rect_unit;
    _routed_rect_unit = routed_rect_unit;
    _violation_unit = violation_unit;
    _complete_ripup = complete_ripup;
  }
  ~DRParameter() = default;
  // getter
  double get_prefer_wire_unit() const { return _prefer_wire_unit; }
  double get_nonprefer_wire_unit() const { return _nonprefer_wire_unit; }
  double get_via_unit() const { return _via_unit; }
  double get_corner_unit() const { return _corner_unit; }
  int32_t get_size() const { return _size; }
  int32_t get_offset() const { return _offset; }
  double get_fixed_rect_unit() const { return _fixed_rect_unit; }
  double get_routed_rect_unit() const { return _routed_rect_unit; }
  double get_violation_unit() const { return _violation_unit; }
  bool get_complete_ripup() const { return _complete_ripup; }
  // setter
  void set_prefer_wire_unit(const double prefer_wire_unit) { _prefer_wire_unit = prefer_wire_unit; }
  void set_nonprefer_wire_unit(const double nonprefer_wire_unit) { _nonprefer_wire_unit = nonprefer_wire_unit; }
  void set_via_unit(const double via_unit) { _via_unit = via_unit; }
  void set_corner_unit(const double corner_unit) { _corner_unit = corner_unit; }
  void set_size(const int32_t size) { _size = size; }
  void set_offset(const int32_t offset) { _offset = offset; }
  void set_fixed_rect_unit(const double fixed_rect_unit) { _fixed_rect_unit = fixed_rect_unit; }
  void set_routed_rect_unit(const double routed_rect_unit) { _routed_rect_unit = routed_rect_unit; }
  void set_violation_unit(const double violation_unit) { _violation_unit = violation_unit; }
  void set_complete_ripup(const bool complete_ripup) { _complete_ripup = complete_ripup; }

 private:
  double _prefer_wire_unit = 0;
  double _nonprefer_wire_unit = 0;
  double _via_unit = 0;
  double _corner_unit = 0;
  int32_t _size = -1;
  int32_t _offset = -1;
  double _fixed_rect_unit = 0;
  double _routed_rect_unit = 0;
  double _violation_unit = 0;
  bool _complete_ripup = true;
};

}  // namespace irt
