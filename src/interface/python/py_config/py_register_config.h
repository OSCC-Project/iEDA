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
        py::arg("lib_paths") = std::vector<std::string>{},
        py::arg("sdc_path") = ""
    );
}
}  // namespace python_interface