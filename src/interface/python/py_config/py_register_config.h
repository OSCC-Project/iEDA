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
#include "py_config.h"
namespace python_interface {
namespace py = pybind11;

void register_config(pybind11::module& m){
    m.def("flow_init", flow_init, py::arg("flow_config"));
    
    m.def("db_init", db_init, 
        py::arg("config_path") = "",
        py::arg("tech_lef_path") = "",
        py::arg("lef_paths") = std::vector<std::string> {},
        py::arg("def_path") = "",
        py::arg("verilog_path") = "",
        py::arg("output_path") = "",
        py::arg("feature_path") = "",
        py::arg("lib_paths") = std::vector<std::string>{},
        py::arg("sdc_path") = ""
    );
}
}  // namespace python_interface