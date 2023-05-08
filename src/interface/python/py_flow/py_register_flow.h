#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "py_flow.h"

namespace python_interface {
namespace py = pybind11;
void register_flow(pybind11::module& m)
{
  m.def("flow_run", flowAutoRun);
  m.def("flow_exit", flowExit);
}

}  // namespace python_interface