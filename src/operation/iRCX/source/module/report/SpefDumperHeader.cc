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
#include "SpefDumper.hh"

#include <ctime>
#include <iomanip>
#include <sstream>

#include "LayoutData.hh"
#include "RCXConfig.hh"
#include "SpefContext.hh"

namespace ircx {

void SpefDumper::writeHeader(std::ofstream& ofs, Size corner_idx) const
{
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  std::ostringstream date_ss;
  date_ss << std::put_time(&tm, "%a %b %d %H:%M:%S %Y");

  ofs << "*SPEF \"IEEE 1481-1998\"\n";
  ofs << "*DESIGN \"" << layout_data_->design_name << "\"\n";
  ofs << "*DATE \"" << date_ss.str() << "\"\n";
  ofs << "*VENDOR \"ECOS\"\n";
  ofs << "*PROGRAM \"iRCX\"\n";
  ofs << "*VERSION \"1.0\"\n";
  ofs << "*DESIGN_FLOW \"PIN_CAP NONE\"\n";
  ofs << "*DIVIDER /\n";
  ofs << "*DELIMITER :\n";
  ofs << "*BUS_DELIMITER []\n";
  ofs << "*T_UNIT 1.0 NS\n";
  ofs << "*C_UNIT 1.0 FF\n";
  ofs << "*R_UNIT 1.0 OHM\n";
  ofs << "*L_UNIT 1.0 HENRY\n";

  if (RCX_CONFIG_INST.get_report_geometry()) {
    ofs << "\n// COMMENTS\n\n";
    ofs << "//   HALF_NODE_SCALING_FACTOR " << std::setprecision(6) << (*corner_data_)[corner_idx].halfNodeScaleFactor() << "\n";
  }
}

void SpefDumper::writeNameMap(std::ofstream& ofs) const
{
  ofs << "\n*NAME_MAP\n";

  for (const auto& [id, name] : name_maps_.port_id_to_name) {
    ofs << "*" << id << " " << name << "\n";
  }
  for (const auto& [id, name] : name_maps_.inst_id_to_name) {
    ofs << "*" << id << " " << name << "\n";
  }
  for (const auto& [id, name] : name_maps_.net_id_to_name) {
    ofs << "*" << id << " " << name << "\n";
  }
}

void SpefDumper::writePorts(std::ofstream& ofs) const
{
  ofs << "\n*PORTS\n\n";
  for (Size port_idx = 0; port_idx < spef_context_->port_names.size(); ++port_idx) {
    const Str& name = spef_context_->port_names[port_idx];
    ofs << "*" << name_maps_.port_name_to_id[name] << " " << spef_context_->port_io[port_idx] << "\n";
  }
}

void SpefDumper::writeLayerMap(std::ofstream& ofs) const
{
  if (!RCX_CONFIG_INST.get_report_geometry()) {
    return;
  }

  ofs << "\n// *LAYER_MAP\n\n";
  for (const ReportLayer& layer : report_layers_) {
    ofs << "// *" << layer.report_id << " " << layer.design_name;
    if (!layer.process_name.empty()) {
      ofs << "    ITF=" << layer.process_name;
    }
    ofs << "\n";
  }
}

}  // namespace ircx
