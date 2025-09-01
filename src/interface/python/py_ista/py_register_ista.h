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

#include "py_ista.h"

namespace python_interface {
namespace py = pybind11;
void register_ista(py::module& m)
{
  m.def("run_sta", staRun, py::arg("output"));
  m.def("init_sta", staInit, py::arg("output"));
  m.def("report_sta", staReport, ("output"));

  m.def("init_log", initLog, ("log_dir"));
  m.def("set_design_workspace", setDesignWorkSpace, ("design_workspace"));
  m.def("read_lef_def", read_lef_def, ("lef_files"), ("def_file"));
  m.def("read_netlist", readVerilog, ("file_name"));
  m.def("read_liberty", readLiberty, ("file_name"));
  m.def("link_design", linkDesign, ("cell_name"));
  m.def("read_spef", readSpef, ("file_name"));
  m.def("read_sdc", readSdc, py::arg("file_name"));

  m.def("get_net_name", getNetName, py::arg("pin_port_name"));
  m.def("get_segment_capacitance", getSegmentCapacitance, py::arg("layer_id"), py::arg("segment_length"), py::arg("route_layer_id"));
  m.def("get_segment_resistance", getSegmentResistance, py::arg("layer_id"), py::arg("segment_length"), py::arg("route_layer_id"));
  
  m.def("make_rc_tree_inner_node", makeRCTreeInnerNode, py::arg("net_name"), py::arg("id"), py::arg("cap"));
  m.def("make_rc_tree_obj_node", makeRCTreeObjNode, py::arg("pin_port_name"), py::arg("cap"));
  m.def("make_rc_tree_edge", makeRCTreeEdge, py::arg("net_name"), py::arg("node1"), py::arg("node2"), py::arg("res"));
  m.def("update_rc_tree_info", updateRCTreeInfo, py::arg("net_name"));
  m.def("update_timing", updateTiming);
  m.def("report_sta", reportSta);

  m.def("report_timing", reportTiming, py::arg("digits"), py::arg("delay_type"), py::arg("exclude_cell_names"), py::arg("derate"));

  m.def("get_used_libs", get_used_libs);

  // get wire timing data
  py::class_<WireTimingData>(m, "WireTimingData")
  .def_readwrite("from_node_name", &WireTimingData::_from_node_name)
  .def_readwrite("to_node_name", &WireTimingData::_to_node_name)
  .def_readwrite("wire_resistance", &WireTimingData::_wire_resistance)
  .def_readwrite("wire_capacitance", &WireTimingData::_wire_capacitance)
  .def_readwrite("wire_from_slew", &WireTimingData::_wire_from_slew)
  .def_readwrite("wire_to_slew", &WireTimingData::_wire_to_slew)
  .def_readwrite("wire_delay", &WireTimingData::_wire_delay);

  m.def("get_wire_timing_data", getWireTimingData);
}
}  // namespace python_interface