#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "py_inst.h"
namespace python_interface {
namespace py = pybind11;
void register_inst(py::module& m)
{
  m.def("place_instance", fpPlaceInst, py::arg("inst_name"), py::arg("llx"), py::arg("lly"), py::arg("orient"), py::arg("cellmaster"),
        py::arg("source") = "");
}
}  // namespace python_interface