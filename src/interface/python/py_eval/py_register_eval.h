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

#include "py_eval.h"

namespace python_interface {
namespace py = pybind11;
void register_eval(py::module& m)
{
  // wirelength evaluation
  m.def("init_wirelength_eval", init_wirelength_eval);
  m.def("eval_total_wirelength", eval_total_wirelength, py::arg("wirelength_type"));

  // congestion evalation
  m.def("init_cong_eval", init_cong_eval, py::arg("bin_cnt_x"), py::arg("bin_cnt_y"));
  m.def("eval_macro_density", eval_macro_density);
  m.def("eval_macro_pin_density", eval_macro_pin_density);
  m.def("eval_cell_pin_density", eval_cell_pin_density);
  m.def("eval_macro_margin", eval_macro_margin);
  m.def("eval_continuous_white_space", eval_continuous_white_space);
  m.def("eval_macro_channel", eval_macro_channel, py::arg("die_size_ratio"));
  m.def("eval_cell_hierarchy", eval_cell_hierarchy, py::arg("plot_path"), py::arg("level"), py::arg("forward"));
  m.def("eval_macro_hierarchy", eval_macro_hierarchy, py::arg("plot_path"), py::arg("level"), py::arg("forward"));
  m.def("eval_macro_connection", eval_macro_connection, py::arg("plot_path"), py::arg("level"), py::arg("forward"));
  m.def("eval_macro_pin_connection", eval_macro_pin_connection, py::arg("plot_path"), py::arg("level"), py::arg("forward"));
  m.def("eval_macro_io_pin_connection", eval_macro_io_pin_connection, py::arg("plot_path"), py::arg("level"), py::arg("forward"));

  m.def("eval_inst_density", eval_inst_density, py::arg("inst_status"), py::arg("eval_flip_flop"));
  m.def("eval_pin_density", eval_pin_density, py::arg("inst_status"), py::arg("level"));
  m.def("eval_rudy_cong", eval_rudy_cong, py::arg("rudy_type"), py::arg("direction"));
  m.def("eval_overflow", eval_overflow);

  // timing evaluation
  m.def("init_timing_eval", init_timing_eval);

  // plot api
  m.def("plot_bin_value", plot_bin_value, py::arg("plot_path"), py::arg("file_name"), py::arg("value_type"));
  m.def("plot_tile_value", plot_tile_value, py::arg("plot_path"), py::arg("file_name"));
  m.def("plot_flow_value", plot_flow_value, py::arg("plot_path"), py::arg("file_name"), py::arg("step"), py::arg("value"));
  // m.def("eval_net_density", eval_net_density, py::arg("inst_status"));
  // m.def("eval_local_net_density", eval_local_net_density);
  // m.def("eval_global_net_density", eval_global_net_density);
  // m.def("eval_inst_num", eval_inst_num, py::arg("inst_status"));
  // m.def("eval_net_num", eval_net_num, py::arg("net_type"));
  // m.def("eval_pin_num", eval_pin_num, py::arg("inst_status"));
  // m.def("eval_routing_layer_num", eval_routing_layer_num);
  // m.def("eval_track_num", eval_track_num, py::arg("direction"));
  // m.def("eval_track_remain_num", eval_track_remain_num);
  // m.def("eval_track_overflow_num", eval_track_overflow_num);
  // m.def("eval_chip_size", eval_chip_size, py::arg("region_type"));
  // m.def("eval_inst_size", eval_inst_size, py::arg("inst_status");
  // m.def("eval_net_size", eval_net_size);
  // m.def("eval_area", eval_area, py::arg("inst_status"));
  // m.def("eval_macro_peri_area", eval_macro_peri_area);
  // m.def("eval_area_util", eval_area_util, py::arg("inst_status"));
  // m.def("eval_macro_channel_util", eval_macro_channel_util, py::arg("dist_ratio"));
  // m.def("eval_macro_channel_pin_util", eval_macro_channel_pin_util, py::arg("dist_ratio"));
}

}  // namespace python_interface