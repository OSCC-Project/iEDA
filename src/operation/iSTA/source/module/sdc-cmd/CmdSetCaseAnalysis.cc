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
 * @file CmdSetCaseAnalysis.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief set_case_analysis sdc cmd.
 * @version 0.1
 * @date 2025-03-01
 * 
 */
#include "Cmd.hh"

namespace ista {

CmdSetCaseAnalysis::CmdSetCaseAnalysis(const char* cmd_name) : TclCmd(cmd_name) {
  auto* analysis_case_option = new TclIntOption("case", 1);
  addOption(analysis_case_option);

  auto* source_obj_option = new TclStringOption("source_obj", 1, nullptr);
  addOption(source_obj_option);
}

unsigned CmdSetCaseAnalysis::check() { return 1; }

unsigned CmdSetCaseAnalysis::exec() {
  if (!check()) {
    return 0;
  }

  TclOption* case_option = getOptionOrArg("case");
  int case_val = case_option->getIntVal();

  TclOption* source_obj_option = getOptionOrArg("source_obj");

  Sta* ista = Sta::getOrCreateSta();
  Netlist* design_nl = ista->get_netlist();

  auto* source_obj_str = source_obj_option->getStringVal();

  auto object_list = FindObjOfSdc(source_obj_str, design_nl);
  LOG_FATAL_IF(object_list.empty()) << "object list is empty.";

  std::set<DesignObject*> pins;
  for (auto& object : object_list) {
    std::visit(overloaded{
                   [](SdcCommandObj* sdc_obj) {
                     LOG_FATAL << "not support sdc obj.";
                   },
                   [&pins](DesignObject* design_obj) {
                     pins.insert(design_obj);
                   },
               },
               object);
  }

  for (auto* pin : pins) {
    auto* lib_cell = pin->get_own_instance()->get_inst_cell();
    auto& lib_ports = lib_cell->get_cell_ports();
    auto& lib_port = lib_ports[case_val];

    for (auto& cell_arc_set : lib_cell->get_cell_arcs()) {
        auto* cell_arc = cell_arc_set->front();
        const char* src_port_name = cell_arc->get_src_port();
        if (!Str::equal(src_port_name, lib_port->get_port_name())) {
            // hard code all not same port name to disable arc for mux cell.
            // TODO(to taosimin), change to accord port function.
            cell_arc->set_is_disable_arc();
        }
    }

  }
  return 1;
}
}