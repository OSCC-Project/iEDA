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
 * @file NetList.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The netlist class.
 * @version 0.1
 * @date 2021-02-04
 */
#pragma once

#include <list>
#include <utility>
#include <vector>

#include "Config.hh"
#include "FlatMap.hh"
#include "Instance.hh"
#include "Net.hh"
#include "Pin.hh"
#include "Port.hh"
#include "Vector.hh"
#include "string/StrMap.hh"

namespace ista {

class PortIterator;
class PortBusIterator;
class InstanceIterator;
class NetIterator;

/**
 * @brief The netlist class for design.
 *
 */
class Netlist : public DesignObject {
 public:
  Netlist() : DesignObject("top"){};
  ~Netlist() override = default;

  Netlist(Netlist&& other) = default;
  Netlist& operator=(Netlist&& rhs) = default;

  friend PortIterator;
  friend PortBusIterator;
  friend InstanceIterator;
  friend NetIterator;

  struct DieSize {
    double _width;
    double _height;
  };

  unsigned isNetlist() override { return 1; }

  auto get_die_size() { return _die_size; }
  void set_die_size(double width, double height) {
    DieSize die_size;
    die_size._width = width;
    die_size._height = height;
    _die_size = die_size;
  }

  Port& addPort(Port&& port) {
    _ports.emplace_back(std::move(port));
    Port* the_port = &(_ports.back());
    _str2port[the_port->get_name()] = the_port;
    return *the_port;
  }

  Port* findPort(const char* port_name) const {
    auto found_port = _str2port.find(port_name);

    if (found_port != _str2port.end()) {
      return found_port->second;
    }
    return nullptr;
  }

  std::vector<DesignObject*> findPort(const char* pattern, bool regexp,
                                      bool nocase);

  std::vector<DesignObject*> findPin(const char* pattern, bool regexp,
                                     bool nocase);

  std::vector<DesignObject*> findObj(const char* pattern, bool regexp,
                                     bool nocase);

  PortBus& addPortBus(PortBus&& port_bus) {
    _port_buses.emplace_back(std::move(port_bus));
    auto* the_port_bus = &(_port_buses.back());
    _str2portbus[the_port_bus->get_name()] = the_port_bus;
    return *the_port_bus;
  }
  auto& get_port_buses() { return _port_buses; }

  PortBus* findPortBus(const char* port_bus_name) const {
    auto found_port_bus = _str2portbus.find(port_bus_name);

    if (found_port_bus != _str2portbus.end()) {
      return found_port_bus->second;
    }
    return nullptr;
  }

  Net& addNet(Net&& net) {
    _nets.emplace_back(std::move(net));
    Net* the_net = &(_nets.back());
    const char* net_name = the_net->get_name();
    _str2net[net_name] = the_net;
    return *the_net;
  }

  void removeNet(Net* net) {
    const char* net_name = net->get_name();
    _str2net.erase(net_name);

    auto it = std::find_if(_nets.begin(), _nets.end(),
                           [net](auto& the_net) { return net == &the_net; });
    LOG_FATAL_IF(it == _nets.end());
    _nets.erase(it);
  }

  Net* findNet(const char* net_name) const {
    auto found_net = _str2net.find(net_name);

    if (found_net != _str2net.end()) {
      return found_net->second;
    }
    return nullptr;
  }

  Instance& addInstance(Instance&& instance) {
    _instances.emplace_back(std::move(instance));

    Instance* the_instance = &(_instances.back());
    const char* instance_name = the_instance->get_name();
    _str2instance[instance_name] = the_instance;

    return *the_instance;
  }

  void removeInstance(const char* instance_name) {
    auto found_instance = _str2instance.find(instance_name);
    LOG_FATAL_IF(found_instance == _str2instance.end());
    auto* the_instance = found_instance->second;

    auto it = std::find_if(
        _instances.begin(), _instances.end(),
        [the_instance](auto& instance) { return the_instance == &instance; });
    _str2instance.erase(found_instance);
    _instances.erase(it);
  }

  Instance* findInstance(const char* instance_name) const {
    auto found_instance = _str2instance.find(instance_name);

    if (found_instance != _str2instance.end()) {
      return found_instance->second;
    }
    return nullptr;
  }

  auto& get_instances() { return _instances; }

  std::size_t getInstanceNum() { return _instances.size(); }
  std::size_t getNetNum() { return _nets.size(); }
  std::size_t getPortNum() { return _ports.size(); }

  void reset();

  void writeVerilog(const char* verilog_file_name="nl.v",
                    std::set<std::string> exclude_cell_names={});

 private:
  std::list<Port> _ports;
  StrMap<Port*> _str2port;  //!< The port name to port for search.
  std::list<PortBus> _port_buses;
  StrMap<PortBus*> _str2portbus;

  std::list<Net> _nets;
  StrMap<Net*> _str2net;  //!< The net name to net for search.
  std::list<Instance> _instances;
  StrMap<Instance*> _str2instance;

  std::optional<DieSize>
      _die_size;  //!< The core size(width * weight) for FP.

  FORBIDDEN_COPY(Netlist);
};

/**
 * @brief Port Iterator of netlist, that provide Java style access.
 *
 */
class PortIterator {
 public:
  explicit PortIterator(Netlist* nl);
  ~PortIterator() = default;

  bool hasNext();
  Port& next();

 private:
  Netlist* _nl;
  std::list<Port>::iterator _iter;

  FORBIDDEN_COPY(PortIterator);
};

/**
 * @brief usage:
 * Netlist* nl;
 * Port* port;
 * FOREACH_PORT(nl, port)
 * {
 *    do_something_for_port();
 * }
 *
 */
#define FOREACH_PORT(nl, port) \
  for (PortIterator iter(nl);  \
       iter.hasNext() ? port = &(iter.next()), true : false;)

/**
 * @brief Port bus Iterator of netlist, that provide Java style access.
 *
 */
class PortBusIterator {
 public:
  explicit PortBusIterator(Netlist* nl);
  ~PortBusIterator() = default;

  bool hasNext();
  PortBus& next();

 private:
  Netlist* _nl;
  std::list<PortBus>::iterator _iter;

  FORBIDDEN_COPY(PortBusIterator);
};

/**
 * @brief usage:
 * Netlist* nl;
 * PortBUS* port_bus;
 * FOREACH_PORT_BUS(nl, port_bus)
 * {
 *    do_something_for_port_bus();
 * }
 *
 */
#define FOREACH_PORT_BUS(nl, port_bus) \
  for (PortBusIterator iter(nl);       \
       iter.hasNext() ? port_bus = &(iter.next()), true : false;)

/**
 * @brief Instance Iterator of netlist, that provide Java style access.
 *
 */
class InstanceIterator {
 public:
  explicit InstanceIterator(Netlist* nl);
  ~InstanceIterator() = default;

  bool hasNext();
  Instance& next();

 private:
  Netlist* _nl;
  std::list<Instance>::iterator _iter;

  FORBIDDEN_COPY(InstanceIterator);
};

/**
 * @brief usage:
 * Netlist* nl;
 * Instance* inst;
 * FOREACH_PORT(nl, inst)
 * {
 *    do_something_for_inst();
 * }
 *
 */
#define FOREACH_INSTANCE(nl, inst) \
  for (InstanceIterator iter(nl);  \
       iter.hasNext() ? inst = &(iter.next()), true : false;)

/**
 * @brief
 *
 */
class NetIterator {
 public:
  explicit NetIterator(Netlist* nl);
  ~NetIterator() = default;

  bool hasNext();
  Net& next();

 private:
  Netlist* _nl;
  std::list<Net>::iterator _iter;

  FORBIDDEN_COPY(NetIterator);
};

/**
 * @brief usage:
 * Netlist* nl;
 * Net* net;
 * FOREACH_PORT(nl, net)
 * {
 *    do_something_for_net();
 * }
 *
 */
#define FOREACH_NET(nl, net)       \
  for (ista::NetIterator iter(nl); \
       iter.hasNext() ? net = &(iter.next()), true : false;)

}  // namespace ista
