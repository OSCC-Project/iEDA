#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ScriptEngine.hh"
#include "py_ipl.h"
namespace python_interface {
namespace py = pybind11;
void register_ipl(py::module& m)
{
  m.def("run_placer", placerAutoRun, py::arg("config"));
  m.def("run_filler", placerRunFiller, py::arg("config"));
  m.def("run_incremental_flow", placerIncrementalFlow, py::arg("config"));
  m.def("run_incremental_lg", placerIncrementalLG);

  m.def("init_pl", placerInit, py::arg("config"));
  m.def("destroy_pl", placerDestroy);
  m.def("placer_run_mp", placerRunMP);
  m.def("placer_run_gp", placerRunGP);
  m.def("placer_run_lg", placerRunLG);
  m.def("placer_run_dp", placerRunDP);
}

}  // namespace python_interface