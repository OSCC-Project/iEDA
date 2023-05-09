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
#ifndef SRC_PLATFORM_EVALUATOR_DATA_TIMINGPIN_HPP_
#define SRC_PLATFORM_EVALUATOR_DATA_TIMINGPIN_HPP_

#include <string>

#include "EvalPoint.hpp"

namespace eval {

class TimingPin
{
 public:
  TimingPin() = default;
  ~TimingPin() = default;

  // getter
  std::string& get_name() { return _name; }
  Point<int64_t>& get_coord() { return _coord; }
  int get_layer_id() const { return _layer_id; }
  int get_id() const { return _id; }

  // setter
  void set_name(const std::string& pin_name) { _name = pin_name; }
  void set_coord(const Point<int64_t>& coord) { _coord = coord; }
  void set_layer_id(const int layer_id) { _layer_id = layer_id; }
  void set_id(const int pin_id) { _id = pin_id; }

  // booler
  bool isRealPin() { return _is_real_pin; }
  void set_is_real_pin(const bool is_real_pin) { _is_real_pin = is_real_pin; }

 private:
  std::string _name;
  Point<int64_t> _coord;
  int _layer_id = 1;
  bool _is_real_pin = false;
  int _id = -1;
};

}  // namespace eval

#endif // SRC_PLATFORM_EVALUATOR_DATA_TIMINGPIN_HPP_
