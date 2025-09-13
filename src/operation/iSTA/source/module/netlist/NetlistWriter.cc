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
 * @file NetlistWriter.cc
 * @author longshy (longshy@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-10-26
 */
#include "NetlistWriter.hh"

#include <string>

#include "time/Time.hh"

namespace ista {

NetlistWriter::NetlistWriter(const char *file_name,
                             std::set<std::string> &exclude_cell_names,
                             Netlist &netlist)
    : _file_name(file_name),
      _exclude_cell_names(exclude_cell_names),
      _netlist(netlist) {
  _stream = std::fopen(file_name, "w");
}

NetlistWriter::~NetlistWriter() { std::fclose(_stream); }

/**
 * @brief write the verilog design.
 *
 */
void NetlistWriter::writeModule() {
  if (!_stream) {
    LOG_ERROR << "File " << _file_name << " NotWritable";
  }

  LOG_INFO << "start write verilog file " << _file_name;

  std::fprintf(_stream, "//Generate the verilog at %s by iSTA.\n",
               Time::getNowWallTime());

  fprintf(_stream, "module %s (", _netlist.get_name());
  fprintf(_stream, "\n");
  writePorts();
  fprintf(_stream, "\n");
  writePortDcls();
  fprintf(_stream, "\n");
  writeWire();
  fprintf(_stream, "\n");
  writeAssign();
  fprintf(_stream, "\n");
  writeInstances();
  fprintf(_stream, "\n");
  fprintf(_stream, "endmodule\n");

  LOG_INFO << "finish write verilog file " << _file_name;
}

/**
 * @brief write the port of the verilog design.
 *
 */
void NetlistWriter::writePorts() {
  bool first = true;
  Port *port;
  FOREACH_PORT(&_netlist, port) {
    if (port->get_port_bus()) {
      continue;
    }

    if (port->isInput() || port->isOutput() || port->isInout()) {
      if (!first) {
        fprintf(_stream, ",\n");
      }
      const char *port_name = port->get_name();
      fprintf(_stream, "%s", port_name);
      first = false;
    }
  }

  PortBus *port_bus;
  FOREACH_PORT_BUS(&_netlist, port_bus) {
    if (!first) {
      fprintf(_stream, ",\n");
    }

    const char *port_bus_name = port_bus->get_name();
    for (int i = 0; i < port_bus->get_size(); i++) {
      fprintf(_stream, "%s[%d]\n", port_bus_name, i);
    }
    first = false;
  }

  fprintf(_stream, "\n);\n");
}

/**
 * @brief write the directed port of the verilog design.
 *
 */
void NetlistWriter::writePortDcls() {
  Port *port;
  FOREACH_PORT(&_netlist, port) {
    if (port->get_port_bus()) {
      continue;
    }

    PortDir port_dir = port->get_port_dir();
    std::string port_name = port->getFullName();
    if (port_dir == PortDir::kIn) {
      fprintf(_stream, "input %s ;\n", port_name.c_str());
    } else if (port_dir == PortDir::kOut) {
      fprintf(_stream, "output %s ;\n", port_name.c_str());
    } else if (port_dir == PortDir::kInOut) {
      fprintf(_stream, "inout %s ;\n", port_name.c_str());
    } else {
      continue;
    }
  }

  PortBus *port_bus;
  FOREACH_PORT_BUS(&_netlist, port_bus) {
    PortDir port_dir = port_bus->get_port_dir();
    std::string port_name = port_bus->getFullName();
    const char *bus_range =
        Str::printf("[%d:%d]", port_bus->get_left(), port_bus->get_right());
    if (port_dir == PortDir::kIn) {
      fprintf(_stream, "input %s %s ;\n", bus_range, port_name.c_str());
    } else if (port_dir == PortDir::kOut) {
      fprintf(_stream, "output %s %s ;\n", bus_range, port_name.c_str());
    } else if (port_dir == PortDir::kInOut) {
      fprintf(_stream, "inout %s %s ;\n", bus_range, port_name.c_str());
    } else {
      continue;
    }
  }
}

/**
 * @brief write the net of the verilog design.
 *
 */
void NetlistWriter::writeWire() {
  Net *net;
  FOREACH_NET(&_netlist, net) {
    std::string net_name = net->getFullName();
    std::string new_net_name = Str::replace(net_name, R"(\\)", "");
    std::string escape_net_name = escapeName(new_net_name);
    fprintf(_stream, "wire %s ;", escape_net_name.c_str());
    fprintf(_stream, "\n");
  }
}

/**
 * @brief write the assign of the verilog design.
 *
 */
void NetlistWriter::writeAssign() {
  Net *net;
  FOREACH_NET(&_netlist, net) {
    std::string net_name = net->getFullName();
    std::string new_net_name = Str::replace(net_name, R"(\\)", "");
    std::string escape_net_name = escapeName(new_net_name);
    for (const auto &pin_port : net->get_pin_ports()) {
      // assign net = input_port;
      if (pin_port->isPort() && pin_port->isInput() &&
          !Str::equal(pin_port->get_name(), escape_net_name.c_str())) {
        fprintf(_stream, "assign %s = %s ;\n", escape_net_name.c_str(),
                pin_port->get_name());
      }

      // assign output_port = net;
      // assign output_port = input_port;
      if (pin_port->isPort() && pin_port->isOutput() &&
          !Str::equal(pin_port->get_name(), escape_net_name.c_str())) {
        fprintf(_stream, "assign %s = %s ;\n", pin_port->get_name(),
                escape_net_name.c_str());
      }
    }
  }
}

/**
 * @brief write the instances of the verilog design.
 *
 */
void NetlistWriter::writeInstances() {
  std::vector<Instance *> instances;
  Instance *inst;
  FOREACH_INSTANCE(&_netlist, inst) {
    if (inst->isInstance()) {
      instances.push_back(inst);
    }
  }

  for (const auto &instance : instances) {
    if (std::string inst_cell_name = instance->get_inst_cell()->get_cell_name();
        _exclude_cell_names.contains(inst_cell_name)) {
      continue;
    }
    writeInstance(instance);
  }
}

/**
 * @brief write the instance of the verilog design.
 *
 */
void NetlistWriter::writeInstance(Instance *inst) {
  auto *inst_cell = inst->get_inst_cell();
  const char *inst_cell_name = inst_cell->get_cell_name();

  std::string inst_name = inst->getFullName();
  std::string new_inst_name = Str::replace(inst_name, R"(\\)", "");

  std::string inst_escape_name = escapeName(new_inst_name);
  fprintf(_stream, "%s %s ( ", inst_cell_name, inst_escape_name.c_str());

  auto get_pin_net_name = [this](Pin *pin) -> std::string {
    std::string pin_net_name;
    if (auto *net = pin->get_net(); net) {
      pin_net_name = pin->get_net()->get_name();
    } else {
      if (pin->isInput()) {
        pin_net_name = R"(1'b0)";
      }
    }

    std::string new_pin_net_name = Str::replace(pin_net_name, R"(\\)", "");
    std::string escape_pin_net_name = escapeName(new_pin_net_name);

    return escape_pin_net_name;
  };

  bool first_pin = true;
  Pin *pin;
  FOREACH_INSTANCE_PIN(inst, pin) {
    // pin bus need concatenate.
    if (pin->get_pin_bus()) {
      continue;
    }

    if (!first_pin) {
      fprintf(_stream, ", ");
    }

    std::string pin_net_name = get_pin_net_name(pin);
    fprintf(_stream, ".%s(%s )", pin->get_name(), pin_net_name.c_str());
    first_pin = false;
  }

  PinBus *pin_bus;
  FOREACH_INSTANCE_PIN_BUS(inst, pin_bus) {
    if (!first_pin) {
      fprintf(_stream, ", ");
    }

    const char *pin_bus_name = pin_bus->get_name();
    auto bus_size = pin_bus->get_size();

    std::string concate_str = "{ ";
    for (int index = bus_size - 1; index >= 0; --index) {
      auto *pin = pin_bus->getPin(index);
      if (!pin) {
        continue;
      }
      std::string pin_net_name = get_pin_net_name(pin);

      concate_str += " ";
      concate_str += pin_net_name;

      if (index != 0) {
        concate_str += " , ";
      }
    }

    concate_str += " }";

    fprintf(_stream, ".%s(%s )", pin_bus_name, concate_str.c_str());
    first_pin = false;
  }

  fprintf(_stream, " );\n");
}

/**
 * @brief judge whether a string need escape.
 *
 * @param name
 * @return true
 * @return false
 */
bool NetlistWriter::isNeedEscape(const std::string &name) {
  bool is_need_escape = false;
  for (const auto &ch : name) {
    if (ch == '/' || ch == '[' || ch == ']' || ch == '.') {
      is_need_escape = true;
      break;
    }
  }
  return is_need_escape;
}

/**
 * @brief escape the name.
 *
 * @param name
 * @return std::string escape_name.
 */
std::string NetlistWriter::escapeName(const std::string &name) {
  std::string escape_name = isNeedEscape(name) ? ("\\" + name) : name;
  return escape_name;
}

}  // namespace ista
