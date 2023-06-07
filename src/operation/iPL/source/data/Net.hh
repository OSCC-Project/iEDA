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
/*
 * @Author: S.J Chen
 * @Date: 2022-01-20 21:57:19
 * @LastEditTime: 2022-12-05 11:24:18
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/database/Net.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_NET_H
#define IPL_NET_H

#include <string>
#include <vector>

#include "Pin.hh"
#include "Rectangle.hh"

namespace ipl {

enum class NET_TYPE
{
  kNone,
  kSignal,
  kClock,
  kReset,
  kFakeNet
};

enum class NET_STATE
{
  kNone,
  kNormal,
  kDontCare
};

class Net
{
 public:
  Net() = delete;
  explicit Net(std::string name);
  Net(const Net&) = delete;
  Net(Net&&) = delete;
  ~Net() = default;

  Net& operator=(const Net&) = delete;
  Net& operator=(Net&&) = delete;

  // getter.
  int32_t get_net_id() const { return _net_id; }
  std::string get_name() const { return _name; }
  Pin* get_driver_pin() const { return _driver_pin; }
  std::vector<Pin*> get_sink_pins() const { return _sink_pins; }
  std::vector<Pin*> get_pins() const;
  float get_net_weight() const { return _netweight; }
  NET_TYPE get_net_type() const { return _net_type; }
  NET_STATE get_net_state() const { return _net_state; }

  bool isSignalNet() const { return _net_type == NET_TYPE::kSignal; }
  bool isClockNet() const { return _net_type == NET_TYPE::kClock; }
  bool isResetNet() const { return _net_type == NET_TYPE::kReset; }
  bool isFakeNet() const { return _net_type == NET_TYPE::kFakeNet; }

  bool isNormalStateNet() const { return _net_state == NET_STATE::kNormal; }
  bool isDontCareNet() const { return _net_state == NET_STATE::kDontCare; }

  // setter.
  void set_net_id(int32_t id) { _net_id = id; }
  void set_driver_pin(Pin* pin) { _driver_pin = pin; }
  void add_sink_pin(Pin* pin) { _sink_pins.push_back(pin); }
  void set_netweight(float weight) { _netweight = weight; }
  void set_net_type(NET_TYPE type) { _net_type = type; }
  void set_net_state(NET_STATE state) { _net_state = state; }

  // function.
  int32_t get_hpwl() const;
  void disConnectLoadPins();

 private:
  int32_t _net_id;
  std::string _name;
  Pin* _driver_pin;
  std::vector<Pin*> _sink_pins;
  float _netweight;

  NET_TYPE _net_type;
  NET_STATE _net_state;
};

inline Net::Net(std::string name)
    : _net_id(-1), _name(std::move(name)), _driver_pin(nullptr), _netweight(1.0), _net_type(NET_TYPE::kNone), _net_state(NET_STATE::kNone)
{
}

inline std::vector<Pin*> Net::get_pins() const
{
  std::vector<Pin*> pins;
  if (_driver_pin) {
    pins.push_back(_driver_pin);
  }

  pins.insert(pins.end(), _sink_pins.begin(), _sink_pins.end());

  return pins;
}

inline int32_t Net::get_hpwl() const
{
  int32_t lower_x = INT32_MAX;
  int32_t lower_y = INT32_MAX;
  int32_t upper_x = INT32_MIN;
  int32_t upper_y = INT32_MIN;

  for (auto* pin : this->get_pins()) {
    Point<int32_t> pin_coordi = pin->get_center_coordi();
    pin_coordi.get_x() < lower_x ? lower_x = pin_coordi.get_x() : lower_x;
    pin_coordi.get_y() < lower_y ? lower_y = pin_coordi.get_y() : lower_y;
    pin_coordi.get_x() > upper_x ? upper_x = pin_coordi.get_x() : upper_x;
    pin_coordi.get_y() > upper_y ? upper_y = pin_coordi.get_y() : upper_y;
  }

  return Rectangle<int32_t>(lower_x, lower_y, upper_x, upper_y).get_half_perimeter();
}

inline void Net::disConnectLoadPins()
{
  for (auto* sink_pin : _sink_pins) {
    sink_pin->set_net(nullptr);
  }
  _sink_pins.clear();
}

}  // namespace ipl

#endif