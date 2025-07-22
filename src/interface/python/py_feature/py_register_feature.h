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

#include "py_feature.h"

namespace python_interface {
namespace py = pybind11;
void register_feature(py::module& m)
{
  m.def("feature_summary", feature_summary, py::arg("path"));
  m.def("feature_tool", feature_tool, py::arg("path"), py::arg("step"));
  m.def("feature_pl_eval", feature_pl_eval, py::arg("json_path"), py::arg("grid_size"));
  m.def("feature_cts_eval", feature_cts_eval, py::arg("json_path"), py::arg("grid_size"));

  m.def("feature_eval_map", feature_eval_map, py::arg("path"), py::arg("bin_cnt_x"), py::arg("bin_cnt_y"));
  m.def("feature_route", feature_route, py::arg("path"));
  m.def("feature_route_read", feature_route_read, py::arg("path"));
  m.def("feature_eval_summary", feature_eval_summary, py::arg("path"), py::arg("grid_size"));
  m.def("feature_timing_eval_summary", feature_timing_eval_summary, py::arg("path"));
  m.def("feature_net_eval", feature_net_eval, py::arg("path"));
  m.def("feature_eval_union", feature_eval_union, py::arg("jsonl_path"), py::arg("csv_path"), py::arg("grid_size"));
}

}  // namespace python_interface