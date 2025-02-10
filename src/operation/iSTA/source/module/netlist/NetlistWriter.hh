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
 * @file NetlistWriter.hh
 * @author longshy (longshy@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-10-25
 */

#pragma once

#include <stdio.h>

#include <set>
#include <string>
#include <vector>

#include "netlist/Netlist.hh"

namespace ista {

class Netlist;
class LibCell;
class Instance;
class Port;

class NetlistWriter {
 public:
  NetlistWriter(const char *file_name,
                std::set<std::string> &exclude_cell_names, Netlist &netlist);
  ~NetlistWriter();

  void writeModule();
  bool isNeedEscape(const std::string &name);
  std::string escapeName(const std::string &name);

 protected:
  void writePorts();
  void writePortDcls();
  void writeWire();
  void writeAssign();
  void writeInstances();
  void writeInstance(Instance *inst);
  // void writeInstPin(Instance *inst, Port *port, bool &first_port);
  // void writeInstBusPin(Instance *inst, Port *port, bool &first_port);
  // void writeInstBusPinBit(Instance *inst, Port *port, bool &first_member);

 private:
  const char *_file_name;
  std::set<std::string> _exclude_cell_names;

  FILE *_stream;
  Netlist &_netlist;
};
}  // namespace ista