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
#ifndef SRC_PLATFORM_EVALUATOR_DATA_TIMINGNET_HPP_
#define SRC_PLATFORM_EVALUATOR_DATA_TIMINGNET_HPP_

#include <vector>

#include "TimingPin.hpp"

namespace eval {

class TimingNet
{
 public:
  TimingNet() = default;
  ~TimingNet() = default;

  // getter
  std::string& get_name() { return _name; }
  std::vector<std::pair<TimingPin*, TimingPin*>>& get_pin_pair_list() { return _pin_pair_list; }

  // setter
  void set_name(const std::string& net_name) { _name = net_name; }

  // adder
  void add_pin_pair(TimingPin* pin_1, TimingPin* pin_2) { _pin_pair_list.push_back(std::make_pair(pin_1, pin_2)); }
  // void add_pin_pair(std::string name_1, int32_t x_1, int32_t y_1, int layer_id_1, std::string name_2, int32_t x_2, int32_t y_2,
  //                   int layer_id_2);
  // void add_pin_pair(int32_t x_1, int32_t y_1, int32_t x_2, int32_t y_2, int layer_id_1, int layer_id_2 = -1, std::string name_1 = "fake",
  //                   std::string name_2 = "fake");

 private:
  std::string _name;
  std::vector<std::pair<TimingPin*, TimingPin*>> _pin_pair_list;
  int32_t _fake_pin_count = 0;
};
}  // namespace eval

#endif // SRC_PLATFORM_EVALUATOR_DATA_TIMINGNET_HPP_
