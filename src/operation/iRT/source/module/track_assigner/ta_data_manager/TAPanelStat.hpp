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

class TAPanelStat
{
 public:
  TAPanelStat() = default;
  ~TAPanelStat() = default;
  // getter
  double get_total_wire_length() const { return _total_wire_length; }
  double get_total_net_and_obs_violation_area() const { return _total_net_and_obs_violation_area; }
  double get_total_net_and_net_violation_area() const { return _total_net_and_net_violation_area; }
  // setter
  void set_total_wire_length(const double total_wire_length) { _total_wire_length = total_wire_length; }
  void set_total_net_and_obs_violation_area(const double total_net_and_obs_violation_area)
  {
    _total_net_and_obs_violation_area = total_net_and_obs_violation_area;
  }
  void set_total_net_and_net_violation_area(const double total_net_and_net_violation_area)
  {
    _total_net_and_net_violation_area = total_net_and_net_violation_area;
  }
  // function

 private:
  double _total_wire_length = 0;
  double _total_net_and_obs_violation_area = 0;
  double _total_net_and_net_violation_area = 0;
};

}  // namespace irt
