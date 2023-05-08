/**
 * @file Pin.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of the pin class.
 * @version 0.1
 * @date 2021-02-03
 *
 * @copyright Copyright (c) 2021
 *
 */

#include <cstring>

#include "Instance.hh"
#include "Pin.hh"
#include "liberty/Liberty.hh"

namespace ista {

Pin::Pin(const char* name, LibertyPort* cell_port)
    : DesignObject(name),
      _cell_port(cell_port),
      _is_VDD(0),
      _is_GND(0),
      _reserverd(0) {}

Pin::Pin(Pin&& other) noexcept
    : DesignObject(std::move(other)),
      _net(other._net),
      _cell_port(other._cell_port),
      _own_instance(other._own_instance),
      _is_VDD(other._is_VDD),
      _is_GND(other._is_GND) {}

Pin& Pin::operator=(Pin&& rhs) noexcept {
  if (this != &rhs) {
    DesignObject::operator=(std::move(rhs));
    _net = rhs._net;
    _cell_port = rhs._cell_port;
    _own_instance = rhs._own_instance;

    _is_VDD = rhs._is_VDD;
    _is_GND = rhs._is_GND;
  }

  return *this;
}

unsigned Pin::isInput() { return _cell_port->isInput(); }
unsigned Pin::isOutput() { return _cell_port->isOutput(); }
unsigned Pin::isInout() { return _cell_port->isInout(); }

double Pin::cap() { return _cell_port->get_port_cap(); }

/**
 * @brief Get the pin cap.
 *
 * @param mode
 * @param trans_type
 * @return double
 */
double Pin::cap(AnalysisMode mode, TransType trans_type) {
  auto ret_value = _cell_port->get_port_cap(mode, trans_type);
  if (!ret_value) {
    return cap();
  }
  return *ret_value;
}

/**
 * @brief Get pin name which is consist of instance name and pin name.
 *
 * @return std::string
 */
std::string Pin::getFullName() {
  const char* instance_name = _own_instance->get_name();
  std::string full_name = instance_name;
  full_name += ":";
  full_name += get_name();
  return full_name;
}

PinBus::PinBus(const char* name, unsigned left, unsigned right, unsigned size)
    : DesignObject(name),
      _left(left),
      _right(right),
      _pins(new Pin*[size]),
      _size(size) {
  std::memset(_pins.get(), 0, sizeof(Pin*) * size);
}

};  // namespace ista