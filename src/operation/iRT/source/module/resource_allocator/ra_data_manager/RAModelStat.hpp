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

#include <string>

#include "RTU.hpp"

namespace irt {

class RAModelStat
{
 public:
  RAModelStat() = default;
  ~RAModelStat() = default;
  // getter
  double get_max_global_cost() { return _max_global_cost; }
  double get_max_avg_cost() { return _max_avg_cost; }
  std::vector<double>& get_avg_cost_list() { return _avg_cost_list; }
  // setter
  void set_max_global_cost(const double max_global_cost) { _max_global_cost = max_global_cost; }
  void set_max_avg_cost(const double max_avg_cost) { _max_avg_cost = max_avg_cost; }
  // function

 private:
  double _max_global_cost;
  double _max_avg_cost;
  std::vector<double> _avg_cost_list;
};

}  // namespace irt
