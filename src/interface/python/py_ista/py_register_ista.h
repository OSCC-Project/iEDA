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

#include "py_ista.h"

namespace python_interface {
namespace py = pybind11;
void register_ista(py::module& m)
{
  m.def("run_sta", staRun, py::arg("output"));
  m.def("init_sta", staInit, py::arg("output"));
  m.def("report_sta", staReport, ("output"));

  m.def("set_design_workspace", setDesignWorkSpace, ("design_workspace"));
  m.def("read_lef_def", read_lef_def, ("lef_files"), ("def_file"));
  m.def("read_netlist", readVerilog, ("file_name"));
  m.def("read_liberty", readLiberty, ("file_name"));
  m.def("link_design", linkDesign, ("cell_name"));
  m.def("read_spef", readSpef, ("file_name"));
  m.def("read_sdc", readSdc, py::arg("file_name"));
  m.def("report_timing", reportTiming, py::arg("digits"), py::arg("delay_type"), py::arg("exclude_cell_names"), py::arg("derate"));
  // m.def("report_constraint") incomplete
}
}  // namespace python_interface