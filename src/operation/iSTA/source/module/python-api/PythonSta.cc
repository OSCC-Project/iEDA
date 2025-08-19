// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file PythonSta.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of python interface.
 * @version 0.1
 * @date 2023-09-10
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "PythonSta.hh"

namespace ista {

PYBIND11_MODULE(ista_cpp, m) {
  m.def("set_design_workspace", set_design_workspace, ("design_workspace"));
  m.def("read_lef_def", read_lef_def, ("lef_files"), ("def_file"));
  m.def("read_netlist", read_netlist, ("file_name"));
  m.def("read_liberty", read_liberty, ("file_name"));
  m.def("link_design", link_design, ("cell_name"));
  m.def("read_spef", read_spef, ("file_name"));
  m.def("read_sdc", read_sdc, py::arg("file_name"));
  m.def("report_timing", report_timing);

  m.def("get_core_size", get_core_size);
  m.def("display_timing_map", display_timing_map);
  m.def("display_timing_tns_map", display_timing_tns_map);
  m.def("display_slew_map", display_slew_map);
  m.def("get_used_libs", get_used_libs);
  
  m.def("build_timing_graph", build_timing_graph);
  m.def("update_clock_timing", update_clock_timing);
  m.def("dump_graph_data", dump_graph_data, ("graph_file"));

  // get wire timing data
  py::class_<StaWireTimingData>(m, "WireTimingData")
  .def_readwrite("from_node_name", &StaWireTimingData::_from_node_name)
  .def_readwrite("to_node_name", &StaWireTimingData::_to_node_name)
  .def_readwrite("wire_resistance", &StaWireTimingData::_wire_resistance)
  .def_readwrite("wire_capacitance", &StaWireTimingData::_wire_capacitance)
  .def_readwrite("wire_from_slew", &StaWireTimingData::_wire_from_slew)
  .def_readwrite("wire_to_slew", &StaWireTimingData::_wire_to_slew)
  .def_readwrite("wire_delay", &StaWireTimingData::_wire_delay);

  m.def("get_wire_timing_data", get_wire_timing_data, py::arg("n_worst_path_per_clock"));
}

}  // namespace ista
