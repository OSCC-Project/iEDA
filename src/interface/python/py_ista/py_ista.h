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
#pragma once

#include <set>
#include <string>
#include <vector>


namespace python_interface {

struct WireTimingData
{
  std::string _from_node_name;
  std::string _to_node_name;
  double _wire_resistance;
  double _wire_capacitance;
  double _wire_from_slew;
  double _wire_to_slew;
  double _wire_delay;
};

using PathWireTimingData = std::vector<WireTimingData>;

bool staRun(const std::string& output);

bool staInit(const std::string& output);

bool staReport(const std::string& output);
bool setDesignWorkSpace(const std::string& design_workspace);

bool initLog(std::string log_path);

bool read_lef_def(std::vector<std::string>& lef_files, const std::string& def_file);
bool readVerilog(const std::string& file_name);

bool readLiberty(std::vector<std::string>& lib_files);

bool linkDesign(const std::string& cell_name);

bool readSpef(const std::string& file_name);

bool readSdc(const std::string& file_name);

std::string getNetName(const std::string& pin_port_name);

double getSegmentResistance(int layer_id, double segment_length);
double getSegmentCapacitance(int layer_id, double segment_length);

std::string makeRCTreeInnerNode(const std::string& net_name, int id, float cap);
std::string makeRCTreeObjNode(const std::string& pin_port_name, float cap);
bool makeRCTreeEdge(const std::string& net_name, std::string& node1, std::string& node2, float res);
bool updateRCTreeInfo(const std::string& net_name);
bool updateTiming();
bool reportSta();

std::vector<PathWireTimingData> getWireTimingData();

bool reportTiming(int digits, const std::string& delay_type, std::set<std::string> exclude_cell_names, bool derate);

std::vector<std::string> get_used_libs();

}  // namespace python_interface