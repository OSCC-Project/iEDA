#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "py_ito.h"

namespace python_interface {
namespace py = pybind11;
void register_ito(py::module& m)
{
    m.def("run_to", toAutoRun,py::arg("config"));
    m.def("run_to_drv", toRunDrv,py::arg("config"));
    m.def("run_to_hold", toRunHold, py::arg("config"));
    m.def("run_to_setup", toRunSetup, py::arg("config"));
}

}  // namespace python_interface