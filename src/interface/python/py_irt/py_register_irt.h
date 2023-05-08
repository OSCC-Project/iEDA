#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ScriptEngine.hh"
#include "py_irt.h"
namespace python_interface {
namespace py = pybind11;
void register_irt(py::module& m)
{
  m.def("destroy_rt", destroyRT);
  m.def("init_rt", initRT, py::arg("config") = "", py::arg("config_dict") = std::map<std::string, std::string>{});
  m.def("run_dr", runDR);
  m.def("run_gr", runGR);
  m.def("run_rt", runRT);
}
}  // namespace python_interface