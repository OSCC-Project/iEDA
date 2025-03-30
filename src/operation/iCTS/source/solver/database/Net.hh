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
/**
 * @file Net.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once
class Pin;
#include <algorithm>
#include <cstdint>

#include "CTSAPI.hh"
#include "Pin.hh"

namespace icts {

class Net
{
 public:
  Net(const std::string& name) : _name(name) {}
  Net(const std::string& name, Pin* driver_pin, const std::vector<Pin*>& load_pins)
      : _name(name), _driver_pin(driver_pin), _load_pins(load_pins)
  {
    init();
  }
  ~Net()
  {
    _driver_pin = nullptr;
    _load_pins.clear();
  };
  // get
  const std::string& get_name() const { return _name; }
  std::vector<Pin*> get_pins() const
  {
    std::vector<Pin*> pins;
    pins.push_back(_driver_pin);
    pins.insert(pins.end(), _load_pins.begin(), _load_pins.end());
    return pins;
  }
  Pin* get_driver_pin() const { return _driver_pin; }
  std::vector<Pin*> get_load_pins() const { return _load_pins; }

  uint16_t getFanout() const { return _load_pins.size(); }

  // set
  void set_name(const std::string& name) { _name = name; }
  void set_driver_pin(Pin* driver_pin) { _driver_pin = driver_pin; }
  void set_load_pins(const std::vector<Pin*>& load_pins) { _load_pins = load_pins; }

  // add
  void add_load_pin(Pin* load_pin) { _load_pins.push_back(load_pin); }

  // remove
  void remove_load_pin(Pin* load_pin) { _load_pins.erase(std::remove(_load_pins.begin(), _load_pins.end(), load_pin), _load_pins.end()); }

 private:
  void init();
  std::string _name = "";
  Pin* _driver_pin = nullptr;
  std::vector<Pin*> _load_pins = {};
};
}  // namespace icts