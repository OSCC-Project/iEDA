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

#include "py_ipdn.h"
namespace python_interface {
namespace py = pybind11;
void register_ipdn(py::module& m)
{
  m.def("add_pdn_io", pdnAddIO, py::arg("pin_name") = "", py::arg("net_name"), py::arg("direction"), py::arg("is_power"));
  m.def("global_net_connect", pdnGlobalConnect, py::arg("net_name"), py::arg("instance_pin_name"), py::arg("is_power"));
  m.def("place_pdn_port", pdnPlacePort, py::arg("pin_name"), py::arg("io_cell_name"), py::arg("offset_x"), py::arg("offset_y"),
        py::arg("width"), py::arg("height"), py::arg("layer"));
  m.def("create_grid", pdnCreateGrid, py::arg("layer_name"), py::arg("net_name_power"), py::arg("net_name_ground"), py::arg("width"),
        py::arg("offset"));
  m.def("create_stripe", pdnCreateStripe, py::arg("layer_name"), py::arg("net_name_power"), py::arg("net_name_ground"), py::arg("width"),
        py::arg("pitch"), py::arg("offset"));
  m.def("connect_two_layer", pdnConnectLayer, py::arg("layers"));
  m.def("connectMacroPdn", pdnConnectMacro, py::arg("pin_layer"), py::arg("pdn_layer"), py::arg("power_pins"), py::arg("ground_pins"),
        py::arg("orient"));
  m.def("connectIoPinToPower", pdnConnectIOPin, py::arg("point_list"), py::arg("layer"));
  m.def("connectPowerStripe", pdnConnectStripe, py::arg("point_list"), py::arg("net_name"), py::arg("layer"), py::arg("width") = -1);
  m.def("add_segment_stripe", pdnAddSegmentStripe, py::arg("net_name") = "", py::arg("point_list") = std::vector<double>{},
        py::arg("layer") = "", py::arg("width") = 0, py::arg("point_begin") = std::vector<double>{}, py::arg("layer_start") = "",
        py::arg("point_end") = std::vector<double>{}, py::arg("layer_end") = "", py::arg("via_width") = 0, py::arg("via_height") = 0);
  m.def("add_segment_via", pdnAddSegmentVia, py::arg("net_name"), py::arg("layer") = "", py::arg("top_layer") = "",
        py::arg("bottom_layer") = "", py::arg("offset_x"), py::arg("offset_y"), py::arg("width"), py::arg("height"));
}
}  // namespace python_interface