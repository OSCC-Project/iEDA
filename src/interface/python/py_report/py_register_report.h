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