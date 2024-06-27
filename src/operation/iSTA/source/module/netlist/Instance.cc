// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file Instance.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-02-03
 */
#include "Instance.hh"

#include "string/Str.hh"

namespace ista {

Instance::Instance(const char* name, LibCell* inst_cell)
    : DesignObject(name), _inst_cell(inst_cell) {
  if (inst_cell) {
    _pins.reserve(_inst_cell->get_num_port());
  }
}

Instance::Instance(Instance&& other)
    : DesignObject(std::move(other)),
      _inst_cell(other._inst_cell),
      _pins(std::move(other._pins)),
      _str2pin(std::move(other._str2pin)),
      _pin_buses(std::move(other._pin_buses)),
      _coordinate(std::move(other._coordinate)) {
  for (auto& pin : _pins) {
    pin->set_own_instance(this);
  }
}

Instance& Instance::operator=(Instance&& rhs) {
  if (this != &rhs) {
    DesignObject::operator=(std::move(rhs));
    _inst_cell = rhs._inst_cell;
    _pins = std::move(rhs._pins);
    _pin_buses = std::move(rhs._pin_buses);
    for (auto& pin : _pins) {
      pin->set_own_instance(this);
    }
    _str2pin = std::move(rhs._str2pin);
    _coordinate = std::move(rhs._coordinate);
  }

  return *this;
}

/**
 * @brief Get pin accord pin name.
 *
 * @param pin_name
 * @return std::optional<Pin*> The found pin.
 */
std::optional<Pin*> Instance::getPin(const char* pin_name) {
  auto p = _str2pin.find(pin_name);
  if (p != _str2pin.end()) {
    return p->second;
  } else {
    // for (auto [str, pin] : _str2pin) {
    //   LOG_INFO << str << " : " << pin;
    // }
    return std::nullopt;
  }
}

Pin* Instance::addPin(const char* name, LibPort* cell_port) {
  auto pin = std::make_unique<Pin>(name, cell_port);
  auto& cell_pin = _pins.emplace_back(std::move(pin));
  cell_pin->set_own_instance(this);
  _str2pin[cell_pin->get_name()] = cell_pin.get();

  return cell_pin.get();
}

Pin* Instance::findPin(LibPort* cell_port) {
  for (auto& pin : _pins) {
    LibPort* port = pin->get_cell_port();
    if (port == cell_port) {
      return pin.get();
    }
  }
  return nullptr;
}

PinIterator::PinIterator(Instance* inst)
    : _inst(inst), _iter(_inst->_pins.begin()) {}

bool PinIterator::hasNext() { return _iter != _inst->_pins.end(); }
Pin* PinIterator::next() { return _iter++->get(); }

PinBusIterator::PinBusIterator(Instance* inst)
    : _inst(inst), _iter(_inst->_pin_buses.begin()) {}

bool PinBusIterator::hasNext() { return _iter != _inst->_pin_buses.end(); }
PinBus* PinBusIterator::next() { return _iter++->get(); }

}  // namespace ista
