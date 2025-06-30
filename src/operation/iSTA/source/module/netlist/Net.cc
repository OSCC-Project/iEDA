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
 * @file Net.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention for Net in the netlist.
 * @version 0.1
 * @date 2021-02-03
 */

#include "Net.hh"

#include <numeric>

#include "Pin.hh"
#include "Port.hh"

namespace ista {

Net::Net(const char* name) : DesignObject(name) {}

Net::Net(Net&& other) noexcept
    : DesignObject(std::move(other)),
      _pin_ports(std::move(other._pin_ports)),
      _is_clock_net(other._is_clock_net) {
  for (auto* pin_port : _pin_ports) {
    pin_port->set_net(this);
  }
}

Net& Net::operator=(Net&& rhs) noexcept {
  if (this != &rhs) {
    DesignObject::operator=(std::move(rhs));
    _pin_ports = std::move(rhs._pin_ports);
    _is_clock_net = rhs._is_clock_net;
    for (auto* pin_port : _pin_ports) {
      pin_port->set_net(this);
    }
  }

  return *this;
}

/**
 * @brief Get net load.
 *
 * @param mode Max/Min analysis mode.
 * @param trans_type  Rise/Fall transition type.
 * @return double
 */
double Net::getLoad(AnalysisMode mode, TransType trans_type) {
  if (auto net_load = get_net_load(mode, trans_type); net_load) {
    return net_load.value();
  }

  double load = std::accumulate(
      _pin_ports.begin(), _pin_ports.end(), 0.0,
      [mode, trans_type, this](float v, DesignObject* pin) {
        return pin == getDriver() ? v : v + pin->cap(mode, trans_type);
      });

  setNetLoad(mode, trans_type, load);
  return load;
}

/**
 * @brief Get the net driver pin or port.
 *
 * @return DesignObject*
 */
DesignObject* Net::getDriver() {
  std::vector<DesignObject*> drivers;
  for (auto* obj : _pin_ports) {
    if (obj->isPort()) {
      auto* port = dynamic_cast<Port*>(obj);
      if (port->isInput()) {
        drivers.push_back(port);
      }
    } else {  // for pin.
      auto* pin = dynamic_cast<Pin*>(obj);
      if (pin->isOutput()) {
        drivers.push_back(pin);
      }
    }
  }

  if (drivers.size() == 1) {
    return drivers[0];
  } else if (drivers.size() > 1) {
    for (auto* driver : drivers) {
      if (!driver->isInout()) {
        return driver;
      }
    }

    LOG_INFO << "Net " << get_name()
             << " connect multiple inout pin/port, random select driver:"
             << drivers[0]->getFullName();
    return drivers[0];
  }

  return nullptr;
}

/**
 * @brief Get net loads.
 *
 * @return std::vector<DesignObject*>
 */
std::vector<DesignObject*> Net::getLoads() {
  auto driver = getDriver();
  std::vector<DesignObject*> loads;
  for (auto* obj : _pin_ports) {
    if (obj != driver) {
      loads.push_back(obj);
    }
  }

  return loads;
}

/**
 * @brief get pin port.
 *
 * @param pin_port_name
 * @return DesignObject*
 */
DesignObject* Net::get_pin_port(const char* pin_port_name) {
  for (auto* obj : _pin_ports) {
    if (obj->getFullName() == pin_port_name) {
      return obj;
    }
  }
  return nullptr;
}

NetPinIterator::NetPinIterator(Net* net)
    : _net(net), _iter(_net->_pin_ports.begin()) {}

bool NetPinIterator::hasNext() { return _iter != _net->_pin_ports.end(); }
DesignObject* NetPinIterator::next() { return *_iter++; }

}  // namespace ista