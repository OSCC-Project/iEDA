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
 * @file Instance.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class for instance in the netlist.
 * @version 0.1
 * @date 2021-02-03
 */

#pragma once

#include <algorithm>
#include <optional>
#include <utility>

#include "DesignObject.hh"
#include "Pin.hh"
#include "Vector.hh"
#include "liberty/Lib.hh"
#include "string/Str.hh"

namespace ista {

class PinIterator;
class PinBusIterator;

/**
 * @brief class for instance of design.
 *
 */
class Instance : public DesignObject {
 public:
  Instance(const char* name, LibCell* cell_name);
  Instance(Instance&& other);
  Instance& operator=(Instance&& rhs);
  ~Instance() override = default;

  friend PinIterator;
  friend PinBusIterator;

  unsigned isInstance() override { return 1; }

  Pin* addPin(const char* name, LibPort* cell_port);
  Pin* getLastPin() { return _pins.back().get(); }
  std::optional<Pin*> getPin(const char* pin_name);
  LibCell* get_inst_cell() { return _inst_cell; }
  void set_inst_cell(LibCell* inst_cell) { _inst_cell = inst_cell; }

  Pin* findPin(LibPort* port);
  std::string getFullName() override { return get_name(); }

  void addPinBus(std::unique_ptr<PinBus> pin_bus) {
    _pin_buses.emplace_back(std::move(pin_bus));
  }
  PinBus* findPinBus(const std::string& bus_name) {
    auto it = std::find_if(
        _pin_buses.begin(), _pin_buses.end(),
        [&bus_name](auto& pin_bus) { return bus_name == pin_bus->get_name(); });
    if (it != _pin_buses.end()) {
      return it->get();
    }
    return nullptr;
  }

  void set_coordinate(double x, double y) override { _coordinate = {x, y}; }
  std::optional<Coordinate> get_coordinate() override { return _coordinate; }

 private:
  LibCell* _inst_cell;
  Vector<std::unique_ptr<Pin>> _pins;
  StrMap<Pin*> _str2pin;
  Vector<std::unique_ptr<PinBus>> _pin_buses;

  std::optional<Coordinate> _coordinate;
  FORBIDDEN_COPY(Instance);
};

/**
 * @brief The class interator of pin.
 *
 */
class PinIterator {
 public:
  explicit PinIterator(Instance* inst);
  ~PinIterator() = default;

  bool hasNext();
  Pin* next();

 private:
  Instance* _inst;
  Vector<std::unique_ptr<Pin>>::iterator _iter;

  FORBIDDEN_COPY(PinIterator);
};

/**
 * @brief usage:
 * Instance* inst;
 * Pin* pin;
 * FOREACH_INSTANCE_PIN(inst, pin)
 * {
 *    do_something_for_pin();
 * }
 *
 */
#define FOREACH_INSTANCE_PIN(inst, pin) \
  for (ista::PinIterator iter(inst);    \
       iter.hasNext() ? pin = (iter.next()), true : false;)

/**
 * @brief The class iterator of pin bus.
 *
 */
class PinBusIterator {
 public:
  explicit PinBusIterator(Instance* inst);
  ~PinBusIterator() = default;

  bool hasNext();
  PinBus* next();

 private:
  Instance* _inst;
  Vector<std::unique_ptr<PinBus>>::iterator _iter;

  FORBIDDEN_COPY(PinBusIterator);
};

/**
 * @brief usage:
 * Instance* inst;
 * PinBus* pin_bus;
 * FOREACH_INSTANCE_PIN_BUS(inst, pin_bus)
 * {
 *    do_something_for_pinbus();
 * }
 *
 */
#define FOREACH_INSTANCE_PIN_BUS(inst, pin_bus) \
  for (ista::PinBusIterator iter(inst);         \
       iter.hasNext() ? pin_bus = (iter.next()), true : false;)

}  // namespace ista
