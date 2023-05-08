#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "py_icts.h"

namespace python_interface {
namespace py = pybind11;
void register_icts(pybind11::module& m) {
  m.def("run_cts", ctsAutoRun, py::arg("cts_config"));
  m.def("cts_report", ctsReport, py::arg("path"));
}

}  // namespace python_interface