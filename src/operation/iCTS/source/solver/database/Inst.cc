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
 * @file Inst.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "Inst.hh"

namespace icts {
const double& Inst::getCapLoad() const
{
  return _load_pin->get_cap_load();
}

const double& Inst::getCapOut() const
{
  return _driver_pin->get_cap_load();
}

void Inst::init()
{
  if (isBuffer() || isNoneLib()) {
    _driver_pin = new Pin(this, _location, _name + "_driver", PinType::kDriver);
  }
  _load_pin = new Pin(this, _location, _name + "_load", PinType::kLoad);
}

void Inst::release()
{
  if (_driver_pin) {
    delete _driver_pin;
    _driver_pin = nullptr;
  }
  if (_load_pin) {
    delete _load_pin;
    _load_pin = nullptr;
  }
}

void Inst::updatePinLocation(const Point& location)
{
  if (_load_pin) {
    _load_pin->set_location(location);
  }
  if (_driver_pin) {
    _driver_pin->set_location(location);
  }
}

}  // namespace icts