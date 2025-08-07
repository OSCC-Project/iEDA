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

#include "wirelength_db.h"
#include "density_db.h"
#include "congestion_db.h"
#include "timing_db.hh"

namespace python_interface {

// wirelength evaluation
ieval::TotalWLSummary total_wirelength();

// density evaluation
ieval::DensityValue cell_density(int bin_cnt_x = 256, int bin_cnt_y = 256, const std::string& save_path = "");
ieval::DensityValue pin_density(int bin_cnt_x = 256, int bin_cnt_y = 256, const std::string& save_path = "");
ieval::DensityValue net_density(int bin_cnt_x = 256, int bin_cnt_y = 256, const std::string& save_path = "");

// congestion evaluation
ieval::CongestionValue rudy_congestion(int bin_cnt_x = 256, int bin_cnt_y = 256, const std::string& save_path = "");
ieval::CongestionValue lut_rudy_congestion(int bin_cnt_x = 256, int bin_cnt_y = 256, const std::string& save_path = "");
ieval::CongestionValue egr_congestion(const std::string& save_path = "");

// timing and power evaluation
ieval::TimingSummary timing_power_hpwl();
ieval::TimingSummary timing_power_stwl();
ieval::TimingSummary timing_power_egr();




void eval_macro_margin();
void eval_macro_channel(float die_size_ratio = 0.5);
void eval_continuous_white_space();
void eval_cell_hierarchy(const std::string& plot_path, int level = 1, int forward = 1);
void eval_macro_hierarchy(const std::string& plot_path, int level = 1, int forward = 1);
void eval_macro_connection(const std::string& plot_path, int level = 1, int forward = 1);
void eval_macro_pin_connection(const std::string& plot_path, int level = 1, int forward = 1);
void eval_macro_io_pin_connection(const std::string& plot_path, int level = 1, int forward = 1);

std::vector<float> eval_overflow();


}  // namespace python_interface