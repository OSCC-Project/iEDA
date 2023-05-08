#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "py_idrc.h"

namespace python_interface {
namespace py = pybind11;

void register_idrc(py::module& m)
{
  m.def("run_drc", drcAutoRun, py::arg("config") = "");
}



}  // namespace python_interface