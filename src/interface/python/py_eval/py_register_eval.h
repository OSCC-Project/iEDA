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
  py::class_<ieval::TotalWLSummary>(m, "TotalWLSummary")
      .def_readwrite("HPWL", &ieval::TotalWLSummary::HPWL)
      .def_readwrite("FLUTE", &ieval::TotalWLSummary::FLUTE)
      .def_readwrite("HTree", &ieval::TotalWLSummary::HTree)
      .def_readwrite("VTree", &ieval::TotalWLSummary::VTree)
      .def_readwrite("GRWL", &ieval::TotalWLSummary::GRWL);

  m.def("total_wirelength_dict", []() -> py::dict {
      ieval::TotalWLSummary summary = total_wirelength();
      py::dict result;
      result["1"] = summary.HPWL;   
      result["2"] = summary.FLUTE;  
      result["3"] = summary.HTree;  
      result["4"] = summary.VTree;  
      result["5"] = summary.GRWL;   
      return result;
  });

  // density evaluation functions
  m.def("cell_density", [](int bin_cnt_x = 256, int bin_cnt_y = 256, const std::string& save_path = "") -> py::tuple {
      auto [max_density, avg_density] = cell_density(bin_cnt_x, bin_cnt_y, save_path);
      return py::make_tuple(max_density, avg_density);
  }, py::arg("bin_cnt_x") = 256, py::arg("bin_cnt_y") = 256, py::arg("save_path") = "");

  m.def("pin_density", [](int bin_cnt_x = 256, int bin_cnt_y = 256, const std::string& save_path = "") -> py::tuple {
      auto [max_density, avg_density] = pin_density(bin_cnt_x, bin_cnt_y, save_path);
      return py::make_tuple(max_density, avg_density);
  }, py::arg("bin_cnt_x") = 256, py::arg("bin_cnt_y") = 256, py::arg("save_path") = "");

  m.def("net_density", [](int bin_cnt_x = 256, int bin_cnt_y = 256, const std::string& save_path = "") -> py::tuple {
      auto [max_density, avg_density] = net_density(bin_cnt_x, bin_cnt_y, save_path);
      return py::make_tuple(max_density, avg_density);
  }, py::arg("bin_cnt_x") = 256, py::arg("bin_cnt_y") = 256, py::arg("save_path") = "");
    
  // congestion evalation
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