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
  m.def("read_netlist", readVerilog, ("file_name"));
  m.def("read_liberty", readLiberty, ("file_name"));
  m.def("link_design", linkDesign, ("cell_name"));
  m.def("read_spef", readSpef, ("file_name"));
  m.def("read_sdc", readSdc, py::arg("file_name"));
  m.def("report_timing", reportTiming, py::arg("digits"), py::arg("delay_type"), py::arg("exclude_cell_names"), py::arg("derate"));
  // m.def("report_constraint") incomplete
}
}  // namespace python_interface