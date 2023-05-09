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

#include "RTU.hpp"

namespace irt {

class DRTaskPriority
{
 public:
  DRTaskPriority() = default;
  ~DRTaskPriority() = default;
  // getter
  ConnectType get_connect_type() const { return _connect_type; }
  double get_routing_area() const { return _routing_area; }
  double get_length_width_ratio() const { return _length_width_ratio; }
  irt_int get_pin_num() const { return _pin_num; }
  // setter
  void set_connect_type(const ConnectType connect_type) { _connect_type = connect_type; }
  void set_routing_area(const double routing_area) { _routing_area = routing_area; }
  void set_length_width_ratio(const double length_width_ratio) { _length_width_ratio = length_width_ratio; }
  void set_pin_num(const irt_int pin_num) { _pin_num = pin_num; }
  // function

 private:
  ConnectType _connect_type = ConnectType::kNone;
  double _routing_area = -1;
  double _length_width_ratio = 1;
  irt_int _pin_num = -1;
};

}  // namespace irt
