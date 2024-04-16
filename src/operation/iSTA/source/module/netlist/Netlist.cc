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
 * @file NetList.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of netlist class.
 * @version 0.1
 * @date 2021-02-04
 */

#include "Netlist.hh"

#include <regex>

#include "NetlistWriter.hh"
#include "log/Log.hh"

namespace ista {

/**
 * @brief find port accord port pattern, thas maybe regexp, wildcard, nocase.
 *
 * @param pattern The search pattern.
 * @param regexp True, if the pattern is regexp.
 * @param nocase True,  if the pattern do not care case.
 * @return DesignObject* The found port.
 */
std::vector<DesignObject*> Netlist::findPort(const char* pattern, bool regexp,
                                             bool nocase) {
  std::vector<DesignObject*> match_ports;
  Port* port;
  if (!regexp && !nocase) {
    FOREACH_PORT(this, port) {
      if (Str::equal(port->get_name(), pattern)) {
        match_ports.push_back(port);
      }
    }
  } else if (regexp) {
    auto nocase_option = nocase ? std::regex::icase : std::regex::extended;
    std::regex re(pattern, nocase_option);

    FOREACH_PORT(this, port) {
      if (std::regex_match(port->get_name(), re)) {
        match_ports.push_back(port);
      }
    }
  } else {
    // TODO(taosimin) fix wildcard.
    LOG_FATAL << "not support yet";
  }

  return match_ports;
}

/**
 * @brief find pin obj accord pattern, thas maybe regexp, wildcard, nocase.
 *
 * @param pattern The search pattern.
 * @param regexp True, if the pattern is regexp.
 * @param nocase True,  if the pattern do not care case.
 * @return DesignObject* The found pin obj.
 */
std::vector<DesignObject*> Netlist::findPin(const char* pattern, bool regexp,
                                            bool nocase) {
  std::vector<DesignObject*> match_pins;
  const char* sep = "/:";
  if (!regexp && !nocase) {
    auto [instance_name, pin_name] = Str::splitTwoPart(pattern, sep);
    if (pin_name.empty()) {
      // LOG_INFO << pattern << " pin name is empty.";
      return match_pins;
    }

    auto* the_instance = findInstance(instance_name.c_str());
    LOG_FATAL_IF(!the_instance)
        << "The instance " << instance_name << " is not exist.";
    auto the_pin = the_instance->getPin(pin_name.c_str());
    LOG_FATAL_IF(!the_pin) << "The pin " << pin_name
                           << " is not exist of instance " << instance_name;
    match_pins.push_back(*the_pin);

  } else {
    // TODO
  }

  return match_pins;
}
/**
 * @brief find obj accord pattern, thas maybe regexp, wildcard, nocase.
 *
 * @param pattern The search pattern.
 * @param regexp True, if the pattern is regexp.
 * @param nocase True,  if the pattern do not care case.
 * @return DesignObject* The found obj.
 */
std::vector<DesignObject*> Netlist::findObj(const char* pattern, bool regexp,
                                            bool nocase) {
  std::vector<DesignObject*> match_objs;
  if (!regexp && !nocase) {
    auto match_ports = findPort(pattern, regexp, nocase);
    std::move(match_ports.begin(), match_ports.end(),
              std::back_inserter(match_objs));

    if (match_objs.empty()) {
      auto* match_instance =
          findInstance(pattern);  // fixme, need support regexp for instance
      if (match_instance) {
        match_objs.emplace_back(match_instance);
      }
    }

    if (match_objs.empty()) {
      std::vector<DesignObject*> match_pins;
      if (Str::contain(pattern, "/") || Str::contain(pattern, ":")) {
        match_pins = findPin(pattern, regexp, nocase);
      }
      std::move(match_pins.begin(), match_pins.end(),
                std::back_inserter(match_objs));
    }

  } else {
    // TODO
  }

  return match_objs;
}

/**
 * @brief clear netlist content.
 *
 */
void Netlist::reset() {
  _ports.clear();
  _str2port.clear();
  _port_buses.clear();
  _str2portbus.clear();

  _nets.clear();
  _str2net.clear();
  _instances.clear();
  _str2instance.clear();
}

/**
 * @brief dump verilog file of netlist.
 *
 * @param verilog_file_name
 * @param sort
 * @param include_pwr_gnd_pins
 */
void Netlist::writeVerilog(const char* verilog_file_name,
                           std::set<std::string> exclude_cell_names) {
  NetlistWriter writer(verilog_file_name, exclude_cell_names, *this);
  writer.writeModule();
}

PortIterator::PortIterator(Netlist* nl) : _nl(nl), _iter(nl->_ports.begin()) {}

bool PortIterator::hasNext() { return _iter != _nl->_ports.end(); }
Port& PortIterator::next() { return *_iter++; }

PortBusIterator::PortBusIterator(Netlist* nl)
    : _nl(nl), _iter(nl->_port_buses.begin()) {}

bool PortBusIterator::hasNext() { return _iter != _nl->_port_buses.end(); }
PortBus& PortBusIterator::next() { return *_iter++; }

InstanceIterator::InstanceIterator(Netlist* nl)
    : _nl(nl), _iter(nl->_instances.begin()) {}

bool InstanceIterator::hasNext() { return _iter != _nl->_instances.end(); }
Instance& InstanceIterator::next() { return *_iter++; }

NetIterator::NetIterator(Netlist* nl) : _nl(nl), _iter(nl->_nets.begin()) {}

bool NetIterator::hasNext() { return _iter != _nl->_nets.end(); }
Net& NetIterator::next() { return *_iter++; }

}  // namespace ista