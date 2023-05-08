/**
 * @file NetlistWriter.hh
 * @author shy long (longshy@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-10-25
 *
 * @copyright Copyright (c) 2021
 *
 */

#pragma once

#include <stdio.h>

#include <set>
#include <string>
#include <vector>

#include "netlist/Netlist.hh"

namespace ista {

class Netlist;
class LibertyCell;
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