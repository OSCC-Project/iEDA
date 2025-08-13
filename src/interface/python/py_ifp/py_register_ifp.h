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

#include "py_ifp.h"
namespace python_interface {
namespace py = pybind11;
void register_ifp(py::module& m)
{
  m.def("init_floorplan", fpInit, py::arg("die_area"), py::arg("core_area"), py::arg("core_site"), py::arg("io_site"),
        py::arg("corner_site"), py::arg("core_util"), py::arg("x_margin"), py::arg("y_margin"), py::arg("xy_ratio"), py::arg("cell_area"));
  m.def("gern_track", fpMakeTracks, py::arg("layer"), py::arg("x_start"), py::arg("x_step"), py::arg("y_start"), py::arg("y_step"));
  m.def("auto_place_pins", fpPlacePins, py::arg("layer"), py::arg("width"), py::arg("height"), py::arg("sides"));
  m.def("place_port", fpPlacePort, py::arg("pin_name"), py::arg("offset_x"), py::arg("offset_y"), py::arg("width"), py::arg("height"),
        py::arg("layer"));
  m.def("place_io_filler", fpPlaceIOFiller, py::arg("filler_types"), py::arg("prefix") = "IOFill");
  m.def("add_placement_blockage", fpAddPlacementBlockage, py::arg("box"));
  m.def("add_placement_halo", fpAddPlacementHalo, py::arg("inst_name"), py::arg("distance"));
  m.def("add_routing_blockage", fpAddRoutingBlockage, py::arg("layer"), py::arg("box"), py::arg("exceptpgnet"));
  m.def("add_routing_halo", fpAddRoutingHalo, py::arg("layer"), py::arg("distance"), py::arg("exceptpgnet") = false, py::arg("inst_name"));
  m.def("tapcell", fpTapCell, py::arg("tapcell"), py::arg("distance"), py::arg("endcap"));
}

}  // namespace python_interface