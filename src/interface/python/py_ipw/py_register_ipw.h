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
#include <set>
#include <string>

namespace py = pybind11;

#include "py_ipw.h"

namespace python_interface {

void register_ipw(py::module& m)
{
  m.def("read_vcd_cpp", &readRustVCD, py::arg("file_name"), py::arg("top_name"));
  m.def("read_pg_spef", &read_pg_spef, py::arg("pg_spef_file"));
  m.def("report_power_cpp", &report_power);
  m.def("report_ir_drop", &report_ir_drop, py::arg("power_nets"));

  // for dataflow.
  m.def("create_data_flow", &create_data_flow);

  py::class_<ipower::ClusterConnection>(m, "ClusterConnection")
    .def_readwrite("dst_cluster_id", &ipower::ClusterConnection::_dst_cluster_id)
    .def_readwrite("stages_each_hop", &ipower::ClusterConnection::_stages_each_hop)
    .def_readwrite("hop", &ipower::ClusterConnection::_hop);
  m.def("build_connection_map", &build_connection_map);


py::class_<ipower::MacroConnection>(m, "MacroConnection")
    .def_readwrite("src_macro_name", &ipower::MacroConnection::_src_macro_name)
    .def_readwrite("dst_macro_name", &ipower::MacroConnection::_dst_macro_name)
    .def_readwrite("stages_each_hop", &ipower::MacroConnection::_stages_each_hop)
    .def_readwrite("hop", &ipower::MacroConnection::_hop);
  m.def("build_macro_connection_map", &build_macro_connection_map);

}

}