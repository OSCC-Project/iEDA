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
  m.def("read_netlist", read_netlist, ("file_name"));
  m.def("read_liberty", read_liberty, ("file_name"));
  m.def("link_design", link_design, ("cell_name"));
  m.def("read_spef", read_spef, ("file_name"));
  m.def("read_sdc", read_sdc, py::arg("file_name"));
  m.def("report_timing", report_timing);

  m.def("read_vcd", &read_vcd, py::arg("file_name"), py::arg("top_name"));
  m.def("report_power", &report_power);
}
}  // namespace ipower