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
 * @file PythonPower.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief The implemention of python interface.
 * @version 0.1
 * @date 2023-09-05
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "PythonPower.hh"

#include "python-api/PythonSta.hh"

using namespace ista;

namespace ipower {

PYBIND11_MODULE(ipower_cpp, m) {
  m.def("set_design_workspace", set_design_workspace, ("design_workspace"));
  m.def("read_lef_def", read_lef_def, ("lef_files"), ("def_file"));
  m.def("read_netlist", read_netlist, ("file_name"));
  m.def("read_liberty", read_liberty, ("file_name"));
  m.def("link_design", link_design, ("cell_name"));
  m.def("read_spef", read_spef, ("file_name"));
  m.def("read_sdc", read_sdc, py::arg("file_name"));
  m.def("report_timing", report_timing);
  m.def("display_timing_map", display_timing_map);
  m.def("display_timing_tns_map", display_timing_tns_map);
  m.def("display_slew_map", display_slew_map);

  m.def("read_vcd", &read_vcd, py::arg("vcd_file"), py::arg("top_instance_name"));
  m.def("read_pg_spef", &read_pg_spef, py::arg("pg_spef_file"));
  m.def("report_power", &report_power);
  m.def("report_ir_drop", &report_ir_drop, py::arg("power_nets"));

  m.def("display_power_map", &display_power_map);
  m.def("display_ir_drop_map", &display_ir_drop_map);

  // for dataflow.
  m.def("create_data_flow", &create_data_flow);

  py::class_<ipower::ClusterConnection>(m, "ClusterConnection")
    .def_readwrite("dst_cluster_id", &ipower::ClusterConnection::_dst_cluster_id)
    .def_readwrite("stages_each_hop", &ipower::ClusterConnection::_stages_each_hop)
    .def_readwrite("hop", &ipower::ClusterConnection::_hop);
  m.def("build_connection_map", &build_connection_map);

  py::class_<ipower::MacroConnection>(m, "MacroConnection")
    .def_readwrite("src_macro_name", &ipower::MacroConnection::_src_macro_name)
    .def_readwrite("dst_macro_name", &ipower::MacroConnection::_dst_macro_name)
    .def_readwrite("stages_each_hop", &ipower::MacroConnection::_stages_each_hop)
    .def_readwrite("hop", &ipower::MacroConnection::_hop);
  m.def("build_macro_connection_map", &build_macro_connection_map);


}
}  // namespace ipower