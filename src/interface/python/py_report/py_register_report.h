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

#include "py_report.h"

namespace python_interface {
namespace py = pybind11;
void register_report(py::module& m)
{
  m.def("report_wirelength", reportWireLength, py::arg("path") = "");
  m.def("report_db", reportDbSummary, py::arg("path") = "");
  m.def("report_congestion", reportCong, py::arg("path") = "");
  m.def("report_dangling_net", reportDanglingNet, py::arg("path") = "");
  m.def("report_route", reportRoute, py::arg("path") = "", py::arg("net") = "", py::arg("summary") = true);
  m.def("report_place_distribution", reportPlaceDistribution, py::arg("prefixes") = std::vector<std::string>{});
  m.def("report_prefixed_instance", reportPrefixedInst, py::arg("prefix"), py::arg("level") = 1, py::arg("num_threshold") = 1);
  m.def("report_drc", reportDRC, py::arg("path"));
}
}  // namespace python_interface