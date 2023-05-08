/**
 * @file Port.cc
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-02-04
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "Port.hh"

namespace ista {

Port::Port(const char* name, PortDir port_dir)
    : DesignObject(name), _port_dir(port_dir), _net(nullptr) {}

Port::Port(Port&& other) noexcept
    : DesignObject(std::move(other)),
      _port_dir(other._port_dir),
      _net(other._net) {}

Port& Port::operator=(Port&& other) noexcept {
  if (this != &other) {
    DesignObject::operator=(std::move(other));
    _port_dir = other._port_dir;
    _net = other._net;
  }

  return *this;
}

/**
 * @brief Get the cap of the port.
 *
 * @param mode
 * @param trans_type
 * @return double
 */
double Port::cap(AnalysisMode mode, TransType trans_type) {
  return _caps[ModeTransPair(mode, trans_type)];
}

/**
 * @brief Get one cap of the port.
 *
 * @return double Cap value.
 */
double Port::cap() {
  // choose one if all is the same.
  return _caps[ModeTransPair(AnalysisMode::kMax, TransType::kRise)];
}

/**
 * @brief Set the cap of the port.
 *
 * @param cap Cap value.
 */
void Port::set_cap(double cap) {
  FOREACH_MODE_TRANS(mode, trans_type) {
    _caps[ModeTransPair(mode, trans_type)] = cap;
  }
}

/**
 * @brief Set the cap of port.
 *
 * @param mode Max/Min.
 * @param trans_type Rise/Fall.
 * @param cap Cap value.
 */
void Port::set_cap(AnalysisMode mode, TransType trans_type, double cap) {
  _caps[ModeTransPair(mode, trans_type)] = cap;
}

PortBus::PortBus(const char* name, unsigned left, unsigned right, unsigned size,
                 PortDir port_dir)
    : DesignObject(name),
      _left(left),
      _right(right),
      _port_dir(port_dir),
      _ports(new Port*[size]),
      _size(size) {}

};  // namespace ista
