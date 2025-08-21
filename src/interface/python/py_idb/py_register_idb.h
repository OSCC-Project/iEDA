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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>

#include "py_db.h"
#include "py_db_op.h"

namespace python_interface {
namespace py = pybind11;

void register_idb(py::module& m)
{
  m.def("idb_init", initIdb);
  m.def("tech_lef_init", initTechLef);
  m.def("lef_init", initLef, py::arg("lef_paths"));
  m.def("def_init", initDef, py::arg("def_path"));
  m.def("verilog_init", initVerilog, py::arg("verilog_path"), py::arg("top_module"));
  m.def("def_save", saveDef, py::arg("def_name"));
  m.def("netlist_save", saveNetList, py::arg("netlist_path"), py::arg("exclude_cell_names") = std::set<std::string>{},
        py::arg("is_add_space_for_escape_name") = false);
  m.def("gds_save", saveGDSII, py::arg("gds_name"));
}

void register_idb_op(pybind11::module& m)
{
  m.def("set_net", setNet, py::arg("net_name"), py::arg("net_type"));
  m.def("remove_except_pg_net", removeExceptPgNet);
  m.def("clear_blockage", clearBlockage, py::arg("type"));
  m.def("idb_get", idbGet, py::arg("inst_name") = "", py::arg("net_name") = "", py::arg("file_name") = "");
  m.def("delete_inst", idbDeleteInstance, py::arg("inst_name"));
  m.def("delete_net", idbDeleteNet, py::arg("net_name"));
  m.def("create_inst", idbCreateInstance, py::arg("inst_name"), py::arg("cell_master"), py::arg("coord_x") = 0, py::arg("coord_y") = 0,
        py::arg("orient") = "", py::arg("type") = "", py::arg("status") = "");
  m.def("create_net", idbCreateNet, py::arg("net_name"), py::arg("conn_type") = "");
}
}  // namespace python_interface