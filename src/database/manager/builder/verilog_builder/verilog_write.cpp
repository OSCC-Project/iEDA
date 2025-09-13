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
 * @file verilog_writer.cpp
 * @author longshy (longshy@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-12-03
 */
#include "verilog_write.h"

#include <cassert>
#include <map>
#include <regex>

#include "log/Log.hh"
#include "string/Str.hh"
#include "time/Time.hh"

namespace idb {
VerilogWriter::VerilogWriter(const char* file_name, std::set<std::string>& exclude_cell_names, IdbDesign& idb_design,
                             bool is_add_space_for_escape_name)
    : _file_name(file_name),
      _exclude_cell_names(exclude_cell_names),
      _idb_design(idb_design),
      _is_add_space_for_escape_name(is_add_space_for_escape_name)
{
  _stream = std::fopen(file_name, "w");
}

VerilogWriter::~VerilogWriter()
{
  std::fclose(_stream);
}

/**
 * @brief write the verilog design.
 *
 */
void VerilogWriter::writeModule()
{
  if (!_stream) {
    LOG_INFO << "File" << _file_name << "NotWritable";
  }
  LOG_INFO << "start write verilog file " << _file_name;

  fprintf(_stream, "//Generate the verilog at %s by iSTA.\n", ieda::Time::getNowWallTime());

  fprintf(_stream, "module %s (", _idb_design.get_design_name().c_str());
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
void VerilogWriter::writePorts()
{
  bool first = true;

  vector<IdbPin*> io_pin_list = _idb_design.get_io_pin_list()->get_pin_list();

  for (const auto& io_pin : io_pin_list) {
    std::string pin_name = io_pin->get_pin_name();
    auto [pin_bus_name, is_bus] = ieda::Str::matchBusName(pin_name.c_str());

    if (is_bus) {
      continue;
    }

    auto pin_type = io_pin->get_term()->get_type();

    if (pin_type == IdbConnectType::kPower || pin_type == IdbConnectType::kGround) {
      continue;
    }

    if (io_pin->get_term()->get_direction() == IdbConnectDirection::kInput
        || io_pin->get_term()->get_direction() == IdbConnectDirection::kOutput
        || io_pin->get_term()->get_direction() == IdbConnectDirection::kInOut) {
      if (!first) {
        fprintf(_stream, ",\n");
      }

      fprintf(_stream, "%s", pin_name.c_str());
      first = false;
    }
  }

  std::set<std::string> bus_processed;
  for (const auto& io_pin : io_pin_list) {
    std::string pin_name = io_pin->get_pin_name();
    auto [pin_bus_name, is_bus] = ieda::Str::matchBusName(pin_name.c_str());

    if (!is_bus) {
      continue;
    }

    // if (bus_processed.contains(pin_bus_name)) {
    //   continue;
    // }

    if (!first) {
      fprintf(_stream, ",\n");
    }

    // bus_processed.insert(pin_bus_name);

    fprintf(_stream, "\\%s ", pin_name.c_str());
    first = false;
  }

  fprintf(_stream, ");\n");
}

/**
 * @brief write the directed port of the verilog design.
 *
 */
void VerilogWriter::writePortDcls()
{
  std::vector<std::string> input_port_names;
  std::vector<std::string> output_port_names;
  std::vector<std::string> inout_port_names;

  vector<IdbPin*> io_pin_list = _idb_design.get_io_pin_list()->get_pin_list();

  for (const auto& io_pin : io_pin_list) {
    std::string pin_name = io_pin->get_pin_name();
    auto [pin_bus_name, is_bus] = ieda::Str::matchBusName(pin_name.c_str());

    if (is_bus) {
      continue;
    }

    auto pin_type = io_pin->get_term()->get_type();
    if (pin_type == IdbConnectType::kPower || pin_type == IdbConnectType::kGround) {
      continue;
    }

    IdbConnectDirection port_dir = io_pin->get_term()->get_direction();

    if (port_dir == IdbConnectDirection::kInput) {
      fprintf(_stream, "input %s ;\n", pin_name.c_str());
    } else if (port_dir == IdbConnectDirection::kOutput) {
      fprintf(_stream, "output %s ;\n", pin_name.c_str());
    } else if (port_dir == IdbConnectDirection::kInOut) {
      fprintf(_stream, "inout %s ;\n", pin_name.c_str());
    } else {
      continue;
    }
  }

  std::set<std::string> bus_processed;
  for (const auto& io_pin : io_pin_list) {
    std::string pin_name = io_pin->get_pin_name();
    auto [pin_bus_name, is_bus] = ieda::Str::matchBusName(pin_name.c_str());

    if (!is_bus) {
      continue;
    }

    // if (bus_processed.contains(pin_bus_name)) {
    //   continue;
    // }

    // bus_processed.insert(pin_bus_name);

    // auto pin_bus = _idb_design.get_bus_list()->findBus(pin_bus_name);
    // unsigned int bus_left = pin_bus->get().get_left();
    // unsigned int bus_right = pin_bus->get().get_right();

    IdbConnectDirection port_dir = io_pin->get_term()->get_direction();

    // const char* bus_range = ieda::Str::printf("[%d:%d]", bus_left, bus_right);

    if (port_dir == IdbConnectDirection::kInput) {
      fprintf(_stream, "input \\%s ;\n",  pin_name.c_str());
    } else if (port_dir == IdbConnectDirection::kOutput) {
      fprintf(_stream, "output \\%s ;\n",  pin_name.c_str());
    } else if (port_dir == IdbConnectDirection::kInOut) {
      fprintf(_stream, "inout \\%s ;\n",  pin_name.c_str());
    } else {
      continue;
    }
  }
}

/**
 * @brief write the net of the verilog design.
 *
 */
void VerilogWriter::writeWire()
{
  vector<IdbNet*> net_list = _idb_design.get_net_list()->get_net_list();

  auto replace_str = [](const string& str, const string& replace_str, const string& new_str) {
    std::regex re(replace_str);
    return std::regex_replace(str, re, new_str);
  };

  for (const auto& net : net_list) {
    std::string net_name = net->get_net_name();

    auto [net_bus_name, is_bus] = ieda::Str::matchBusName(net_name.c_str());

    if (net_bus_name.back() == '\\') {
      is_bus = std::nullopt;
    }

    // bus of bus is not printed as bus
    if (std::ranges::count(net_name, '[') > 1) {
      is_bus = std::nullopt;
    }

    if (is_bus) {
      continue;
    }

    std::string new_net_name = replace_str(net_name, R"(\\)", "");
    std::string escape_net_name = escapeName(new_net_name);
    fprintf(_stream, "wire %s ;\n", escape_net_name.c_str());
  }

  // std::set<std::string> bus_processed;
  // for (const auto& net : net_list) {
  //   std::string net_name = net->get_net_name();

  //   auto [net_bus_name, is_bus] = ieda::Str::matchBusName(net_name.c_str());

  //   if (net_bus_name.back() == '\\') {
  //     is_bus = std::nullopt;
  //   }

  //   // bus of bus is not printed as bus
  //   if (std::ranges::count(net_name, '[') > 1) {
  //     is_bus = std::nullopt;
  //   }

  //   if (!is_bus) {
  //     continue;
  //   }

  //   if (bus_processed.contains(net_bus_name)) {
  //     continue;
  //   }

  //   bus_processed.insert(net_bus_name);
  //   // remove all "\" in net_bus_name
  //   net_bus_name.erase(std::remove(net_bus_name.begin(), net_bus_name.end(), '\\'), net_bus_name.end());
  //   auto net_bus = _idb_design.get_bus_list()->findBus(net_bus_name);
  //   assert(net_bus);
  //   int bus_left = net_bus->get().get_left();
  //   int bus_right = net_bus->get().get_right();

  //   std::string escape_bus_net_name = escapeName(net_bus_name);

  //   fprintf(_stream, "wire [%d:%d] %s ;\n", bus_left, bus_right, escape_bus_net_name.c_str());
  // }
}

/**
 * @brief write assign declarations(assign net=port(such as assign g6265 = 983 ;))
 *
 */
void VerilogWriter::writeAssign()
{
  vector<IdbNet*> net_list = _idb_design.get_net_list()->get_net_list();
  for (const auto& net : net_list) {
    std::string net_name = net->get_net_name();
    for (const auto& io_pin : net->get_io_pins()->get_pin_list()) {
      // assign net = input_port;

      std::string new_net_name = ieda::Str::replace(net_name, R"(\\)", "");
      std::string escape_net_name = escapeName(new_net_name);

      std::string new_io_pin_name = ieda::Str::replace(io_pin->get_pin_name(), R"(\\)", "");
      std::string escape_io_pin_name = escapeName(new_io_pin_name);

      if (io_pin->get_term()->get_direction() == IdbConnectDirection::kInput && io_pin->get_pin_name() != net_name) {
        fprintf(_stream, "assign %s = %s ;\n", escape_net_name.c_str(), escape_io_pin_name.c_str());
      }
      // assign output_port = net;
      // assign output_port = input_port;
      if (io_pin->get_term()->get_direction() == IdbConnectDirection::kOutput && io_pin->get_pin_name() != net_name) {
        fprintf(_stream, "assign %s = %s ;\n", escape_io_pin_name.c_str(), escape_net_name.c_str());
      }
    }
  }
}

/**
 * @brief write the instances of the verilog design.
 *
 */
void VerilogWriter::writeInstances()
{
  std::vector<IdbInstance*> instance_list = _idb_design.get_instance_list()->get_instance_list();

  for (const auto& instance : instance_list) {
    if (std::string inst_cell_name = instance->get_cell_master()->get_name(); _exclude_cell_names.contains(inst_cell_name)) {
      continue;
    }
    writeInstance(instance);
  }
}

/**
 * @brief write the instance of the verilog design.
 *
 * @param inst
 */
void VerilogWriter::writeInstance(IdbInstance* inst)
{
  auto replace_str = [](const string& str, const string& old_str, const string& new_str) {
    std::regex re(old_str);
    return std::regex_replace(str, re, new_str);
  };

  std::string inst_cell_name = inst->get_cell_master()->get_name();
  std::string inst_name = inst->get_name();
  std::string new_inst_name = replace_str(inst_name, R"(\\)", "");
  std::string inst_escape_name = escapeName(new_inst_name);

  fprintf(_stream, "%s %s ( ", inst_cell_name.c_str(), inst_escape_name.c_str());

  bool first_pin = true;
  vector<IdbPin*> pin_list = inst->get_pin_list()->get_pin_list();

  for (const auto& pin : pin_list) {
    std::string pin_name = pin->get_pin_name();

    auto [pin_bus_name, is_bus] = ieda::Str::matchBusName(pin_name.c_str());

    if (is_bus) {
      continue;
    }

    auto pin_type = pin->get_term()->get_type();

    if (pin_type == IdbConnectType::kPower || pin_type == IdbConnectType::kGround) {
      continue;
    }

    std::string pin_net_name;
    if (pin_name == "VDD" || pin_name == "VSS") {
      pin->get_special_net() ? pin_net_name = pin->get_special_net()->get_net_name() : pin_net_name = pin_name;
    } else {
      if (pin->get_net()) {
        pin_net_name = pin->get_net()->get_net_name();
      }
    }

    pin_net_name = escapeName(pin_net_name);

    if (!first_pin) {
      fprintf(_stream, ", ");
    }

    fprintf(_stream, ".%s(%s )", pin_name.c_str(), pin_net_name.c_str());
    first_pin = false;
  }

  std::set<std::string> bus_processed;
  for (const auto& pin : pin_list) {
    std::string pin_name = pin->get_pin_name();

    auto [pin_bus_name, is_bus] = ieda::Str::matchBusName(pin_name.c_str());

    if (!is_bus) {
      continue;
    }

    if (bus_processed.contains(pin_bus_name)) {
      continue;
    }

    bus_processed.insert(pin_bus_name);

    auto bus_name = pin->get_instance()->get_name();
    bus_name += "/";
    bus_name += pin_bus_name;

    auto pin_bus = _idb_design.get_bus_list()->findBus(bus_name);
    assert(pin_bus);
    int bus_left = pin_bus->get().get_left();
    int bus_right = pin_bus->get().get_right();

    std::string concate_str = "{ ";
    for (int index = bus_left; index >= bus_right; --index) {
      auto* one_pin = pin_bus->get().getPin(index);
      assert(one_pin);

      std::string pin_net_name;

      if (one_pin->get_net()) {
        pin_net_name = one_pin->get_net()->get_net_name();
      } else {
        if (one_pin->get_term()->get_direction() == IdbConnectDirection::kInput) {
          pin_net_name = R"(1'b0)";
        }
      }

      pin_net_name = escapeName(pin_net_name);

      concate_str += " ";
      concate_str += pin_net_name;

      if (index != bus_right) {
        concate_str += " , ";
      }
    }

    concate_str += " }";

    if (!first_pin) {
      fprintf(_stream, ", ");
    }

    fprintf(_stream, ".%s(%s )", pin_bus_name.c_str(), concate_str.c_str());

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
bool VerilogWriter::isNeedEscape(const std::string& name)
{
  bool is_need_escape = false;
  for (const auto& ch : name) {
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
 * @return std::string
 */
std::string VerilogWriter::escapeName(const std::string& name)
{
  std::string trim_name = ieda::Str::trimBackslash(name);

  std::string escape_name;
  if (_is_add_space_for_escape_name) {
    escape_name = isNeedEscape(trim_name) ? "\\" + addSpaceForEscapeName(trim_name) : trim_name;
  } else {
    escape_name = isNeedEscape(trim_name) ? "\\" + trim_name : trim_name;
  }

  return escape_name;
}

/**
 * @brief add space for escape name between id and bracket
 * such as \waddrReg_r[2] should be changed to \waddrReg_r [2], which is required by verilator. And such as \waddrReg_r[2]wa should not be
 * changed.
 * @param name
 * @return std::string
 */
std::string VerilogWriter::addSpaceForEscapeName(const std::string& name)
{
  if (std::count(name.begin(), name.end(), '[') > 1) {
    return name;
  }

  size_t start_pos = name.find("[");
  size_t end_pos = name.find("]");
  if (start_pos != string::npos && end_pos == name.size() - 1) {
    std::string replace_str = name;
    replace_str.replace(start_pos, 1, " [");
    return replace_str;
  }

  return name;
}

/**
 * @brief judge whether a string have "[0]" object in middle.
 *
 * @param str
 * @return true
 * @return false
 */
bool VerilogWriter::isMiddleSquareBracket(const std::string& str)
{
  size_t start_pos = str.find('[');
  size_t end_pos = str.find(']', start_pos);

  if (start_pos != std::string::npos && end_pos != std::string::npos && start_pos > 0 && end_pos < str.size() - 1) {
    for (size_t i = start_pos + 1; i < end_pos; ++i) {
      if (!std::isdigit(str[i])) {
        return false;
      }
    }
    return true;
  }

  return false;
}

}  // namespace idb
