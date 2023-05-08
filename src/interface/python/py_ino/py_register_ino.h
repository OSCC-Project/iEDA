#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "py_ino.h"
namespace python_interface {
namespace py = pybind11;
void register_ino(py::module& m)
{
  m.def("run_no_fixfanout", noRunFixFanout, py::arg("config"));
}
}  // namespace python_interface