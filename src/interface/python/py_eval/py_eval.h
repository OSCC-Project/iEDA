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

#include <string>
#include <vector>
namespace python_interface {

// wirelength evaluation
void init_wirelength_eval();
int64_t eval_total_wirelength(int wirelength_type);

// congestion evaluation
void init_cong_eval(int bin_cnt_x, int bin_cnt_y);
void eval_macro_density();
void eval_macro_pin_density();
void eval_cell_pin_density();
void eval_macro_margin();
void eval_macro_channel(float die_size_ratio = 0.5);
void eval_continuous_white_space();
void eval_cell_hierarchy(const std::string& plot_path, int level = 1, int forward = 1);
void eval_macro_hierarchy(const std::string& plot_path, int level = 1, int forward = 1);
void eval_macro_connection(const std::string& plot_path, int level = 1, int forward = 1);
void eval_macro_pin_connection(const std::string& plot_path, int level = 1, int forward = 1);
void eval_macro_io_pin_connection(const std::string& plot_path, int level = 1, int forward = 1);

void eval_inst_density(int inst_status, int eval_flip_flop = 0);
void eval_pin_density(int inst_status, int level = 0);
void eval_rudy_cong(int rudy_type, int direction);
std::vector<float> eval_overflow();

// timing evaluation
void init_timing_eval();

// plot API
void plot_bin_value(const std::string& plot_path, const std::string& file_name, int value_type);
void plot_tile_value(const std::string& plot_path, const std::string& file_name);
void plot_flow_value(const std::string& plot_path, const std::string& file_name, const std::string& step, const std::string& value);

}  // namespace python_interface