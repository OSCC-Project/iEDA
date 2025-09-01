// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
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
  m.def("run_ai_placement", placerAiRun, py::arg("config"), py::arg("onnx_path"), py::arg("normalization_path"));

  m.def("init_pl", placerInit, py::arg("config"));
  m.def("destroy_pl", placerDestroy);
  m.def("placer_run_mp", placerRunMP);
  m.def("placer_run_gp", placerRunGP);
  m.def("placer_run_lg", placerRunLG);
  m.def("placer_run_dp", placerRunDP);
}

}  // namespace python_interface