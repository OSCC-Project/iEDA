#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "py_ifp.h"
namespace python_interface {
namespace py = pybind11;
void register_ifp(py::module& m)
{
  m.def("init_floorplan", fpInit, py::arg("die_area"), py::arg("core_area"), py::arg("core_site"), py::arg("io_site"));
  m.def("gern_track", fpMakeTracks, py::arg("layer"), py::arg("x_start"), py::arg("x_step"), py::arg("y_start"), py::arg("y_step"));
  m.def("auto_place_pins", fpPlacePins, py::arg("layer"), py::arg("width"), py::arg("height"));
  m.def("place_port", fpPlacePort, py::arg("pin_name"), py::arg("offset_x"), py::arg("offset_y"), py::arg("width"), py::arg("height"),
        py::arg("layer"));
  m.def("place_io_filler", fpPlaceIOFiller, py::arg("filler_types"), py::arg("prefix") = "IOFill", py::arg("orient"), py::arg("begin"),
        py::arg("end"), py::arg("source"));
  m.def("add_placement_blockage", fpAddPlacementBlockage, py::arg("box"));
  m.def("add_placement_halo", fpAddPlacementHalo, py::arg("inst_name"), py::arg("distance"));
  m.def("add_routing_blockage", fpAddRoutingBlockage, py::arg("layer"), py::arg("box"), py::arg("exceptpgnet"));
  m.def("add_routing_halo", fpAddRoutingHalo, py::arg("layer"), py::arg("distance"), py::arg("exceptpgnet") = false, py::arg("inst_name"));
  m.def("tapcell", fpTapCell, py::arg("tapcell"), py::arg("distance"), py::arg("endcap"));
}

}  // namespace python_interface