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
  m.def("rudy_congestion", [](int bin_cnt_x = 256, int bin_cnt_y = 256, const std::string& save_path = "") -> py::tuple {
      auto [max_congestion, total_congestion] = rudy_congestion(bin_cnt_x, bin_cnt_y, save_path);
      return py::make_tuple(max_congestion, total_congestion);
  }, py::arg("bin_cnt_x") = 256, py::arg("bin_cnt_y") = 256, py::arg("save_path") = "");

  m.def("lut_rudy_congestion", [](int bin_cnt_x = 256, int bin_cnt_y = 256, const std::string& save_path = "") -> py::tuple {
      auto [max_congestion, total_congestion] = lut_rudy_congestion(bin_cnt_x, bin_cnt_y, save_path);
      return py::make_tuple(max_congestion, total_congestion);
  }, py::arg("bin_cnt_x") = 256, py::arg("bin_cnt_y") = 256, py::arg("save_path") = "");

  m.def("egr_congestion", [](const std::string& save_path = "") -> py::tuple {
      auto [max_congestion, total_congestion] = egr_congestion(save_path);
      return py::make_tuple(max_congestion, total_congestion);
  }, py::arg("save_path") = "");  


  // timing and power evaluation
  py::class_<ieval::ClockTiming>(m, "ClockTiming")
      .def_readwrite("clock_name", &ieval::ClockTiming::clock_name)
      .def_readwrite("setup_wns", &ieval::ClockTiming::setup_wns)
      .def_readwrite("setup_tns", &ieval::ClockTiming::setup_tns)
      .def_readwrite("hold_wns", &ieval::ClockTiming::hold_wns)
      .def_readwrite("hold_tns", &ieval::ClockTiming::hold_tns)
      .def_readwrite("suggest_freq", &ieval::ClockTiming::suggest_freq);

  py::class_<ieval::TimingSummary>(m, "TimingSummary")
      .def_readwrite("clock_timings", &ieval::TimingSummary::clock_timings)
      .def_readwrite("static_power", &ieval::TimingSummary::static_power)
      .def_readwrite("dynamic_power", &ieval::TimingSummary::dynamic_power);

  m.def("timing_power_hpwl", []() -> py::dict {
      ieval::TimingSummary summary = timing_power_hpwl();
      py::dict result;
      
      py::list clock_list;
      for (const auto& clock : summary.clock_timings) {
          py::dict clock_dict;
          clock_dict["clock_name"] = clock.clock_name;
          clock_dict["setup_wns"] = clock.setup_wns;
          clock_dict["setup_tns"] = clock.setup_tns;
          clock_dict["hold_wns"] = clock.hold_wns;
          clock_dict["hold_tns"] = clock.hold_tns;
          clock_dict["suggest_freq"] = clock.suggest_freq;
          clock_list.append(clock_dict);
      }
      result["clock_timings"] = clock_list;
      
      result["static_power"] = summary.static_power;
      result["dynamic_power"] = summary.dynamic_power;
      
      return result;
  });

  m.def("timing_power_stwl", []() -> py::dict {
      ieval::TimingSummary summary = timing_power_stwl();
      py::dict result;
      
      py::list clock_list;
      for (const auto& clock : summary.clock_timings) {
          py::dict clock_dict;
          clock_dict["clock_name"] = clock.clock_name;
          clock_dict["setup_wns"] = clock.setup_wns;
          clock_dict["setup_tns"] = clock.setup_tns;
          clock_dict["hold_wns"] = clock.hold_wns;
          clock_dict["hold_tns"] = clock.hold_tns;
          clock_dict["suggest_freq"] = clock.suggest_freq;
          clock_list.append(clock_dict);
      }
      result["clock_timings"] = clock_list;
      
      result["static_power"] = summary.static_power;
      result["dynamic_power"] = summary.dynamic_power;
      
      return result;
  });

  m.def("timing_power_egr", []() -> py::dict {
      ieval::TimingSummary summary = timing_power_egr();
      py::dict result;
      
      py::list clock_list;
      for (const auto& clock : summary.clock_timings) {
          py::dict clock_dict;
          clock_dict["clock_name"] = clock.clock_name;
          clock_dict["setup_wns"] = clock.setup_wns;
          clock_dict["setup_tns"] = clock.setup_tns;
          clock_dict["hold_wns"] = clock.hold_wns;
          clock_dict["hold_tns"] = clock.hold_tns;
          clock_dict["suggest_freq"] = clock.suggest_freq;
          clock_list.append(clock_dict);
      }
      result["clock_timings"] = clock_list;
      
      result["static_power"] = summary.static_power;
      result["dynamic_power"] = summary.dynamic_power;
      
      return result;
  });


  // other evaluation (TO BE DONE)
  m.def("eval_macro_margin", eval_macro_margin);
  m.def("eval_continuous_white_space", eval_continuous_white_space);
  m.def("eval_macro_channel", eval_macro_channel, py::arg("die_size_ratio"));
  m.def("eval_cell_hierarchy", eval_cell_hierarchy, py::arg("plot_path"), py::arg("level"), py::arg("forward"));
  m.def("eval_macro_hierarchy", eval_macro_hierarchy, py::arg("plot_path"), py::arg("level"), py::arg("forward"));
  m.def("eval_macro_connection", eval_macro_connection, py::arg("plot_path"), py::arg("level"), py::arg("forward"));
  m.def("eval_macro_pin_connection", eval_macro_pin_connection, py::arg("plot_path"), py::arg("level"), py::arg("forward"));
  m.def("eval_macro_io_pin_connection", eval_macro_io_pin_connection, py::arg("plot_path"), py::arg("level"), py::arg("forward"));

  m.def("eval_overflow", eval_overflow);


}

}  // namespace python_interface