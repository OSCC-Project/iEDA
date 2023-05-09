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
#ifndef SRC_PLATFORM_EVALUATOR_DATABASE_WLNET_HPP_
#define SRC_PLATFORM_EVALUATOR_DATABASE_WLNET_HPP_

#include <climits>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "WLPin.hpp"

namespace eval {
class WLNet
{
 public:
  WLNet() = default;
  ~WLNet() = default;

  // getter
  std::string get_name() const { return _name; }
  NET_TYPE get_type() const { return _type; }
  WLPin* get_driver_pin() const { return _driver_pin; }
  std::vector<WLPin*> get_sink_pin_list() const { return _sink_pin_list; }
  std::vector<WLPin*> get_pin_list() const { return _pin_list; }
  int64_t get_real_wirelength() const { return _real_wirelength; }

  // setter
  void set_name(const std::string& name) { _name = name; }
  void set_type(const NET_TYPE& type) { _type = type; }
  void set_driver_pin(WLPin* pin) { _driver_pin = pin; }
  void set_pin_list(const std::vector<WLPin*>& pin_list) { _pin_list = pin_list; }
  void set_real_wirelength(const int64_t& wirelength) { _real_wirelength = wirelength; }

  // adder
  void add_pin(const int64_t& x, const int64_t& y);
  void add_driver_pin(const int64_t& x, const int64_t& y, const std::string& name);
  void add_sink_pin(const int64_t& x, const int64_t& y, const std::string& name);
  void add_sink_pin(WLPin* pin) { _sink_pin_list.push_back(pin); }
  void add_pin(WLPin* pin) { _pin_list.push_back(pin); }

  // compute net_wirelength
  int64_t wireLoadModel();
  int64_t HPWL();
  int64_t LShapedWL(const std::string& sink_pin_name);
  int64_t HTree();
  int64_t VTree();
  int64_t Star();
  int64_t Clique();
  int64_t B2B();
  int64_t FluteWL();
  int64_t planeRouteWL();
  int64_t spaceRouteWL();
  int64_t detailRouteWL();

 private:
  std::string _name;
  NET_TYPE _type;
  WLPin* _driver_pin;
  std::vector<WLPin*> _sink_pin_list;
  std::vector<WLPin*> _pin_list;
  int64_t _real_wirelength;
};
}  // namespace eval

#endif  // SRC_PLATFORM_EVALUATOR_DATABASE_WLNET_HPP_
