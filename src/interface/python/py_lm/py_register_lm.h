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

#include "py_lm.h"

namespace python_interface {
namespace py = pybind11;
void register_large_model(py::module& m)
{
  m.def("layout_patchs", layout_patchs, py::arg("path"));
  m.def("layout_graph", layout_graph, py::arg("path"));
  m.def("generate_vectors", generate_vectors, py::arg("dir"));

  py::class_<ieval::TimingWireNode>(m, "TimingWireNode")
      .def_readwrite("name", &ieval::TimingWireNode::_name)
      .def_readwrite("is_pin", &ieval::TimingWireNode::_is_pin)
      .def_readwrite("is_port", &ieval::TimingWireNode::_is_port);

  py::class_<ieval::TimingWireEdge>(m, "TimingWireEdge")
      .def_readwrite("from_node", &ieval::TimingWireEdge::_from_node)
      .def_readwrite("to_node", &ieval::TimingWireEdge::_to_node)
      .def_readwrite("is_net_edge", &ieval::TimingWireEdge::_is_net_edge);

  py::class_<ieval::TimingWireGraph>(m, "TimingWireGraph")
      .def_readwrite("nodes", &ieval::TimingWireGraph::_nodes)
      .def_readwrite("edges", &ieval::TimingWireGraph::_edges);

  m.def("get_timing_wire_graph", &get_timing_wire_graph);
}

}  // namespace python_interface