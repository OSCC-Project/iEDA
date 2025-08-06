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
#include "py_eval.h"

#include "congestion_api.h"
#include "density_api.h"
#include "timing_api.hh"
#include "wirelength_api.h"

using namespace ieval;

namespace python_interface {

// wirelength evaluation
ieval::TotalWLSummary total_wirelength()
{
  ieval::TotalWLSummary total_wirelength_summary = WIRELENGTH_API_INST->totalWL();
  
  ieval::TotalWLSummary result;
  result.HPWL = total_wirelength_summary.HPWL;
  result.FLUTE = total_wirelength_summary.FLUTE;
  result.HTree = total_wirelength_summary.HTree;
  result.VTree = total_wirelength_summary.VTree;
  result.GRWL = total_wirelength_summary.GRWL;
  
  return result;
}

// density evaluation
ieval::DensityValue cell_density(int bin_cnt_x, int bin_cnt_y, const std::string& save_path)
{
  ieval::DensityValue density_value = DENSITY_API_INST->cellDensity(bin_cnt_x, bin_cnt_y, save_path);
  return density_value;
}

ieval::DensityValue pin_density(int bin_cnt_x, int bin_cnt_y, const std::string& save_path)
{
  ieval::DensityValue density_value = DENSITY_API_INST->pinDensity(bin_cnt_x, bin_cnt_y, save_path);
  return density_value;
}

ieval::DensityValue net_density(int bin_cnt_x, int bin_cnt_y, const std::string& save_path)
{
  ieval::DensityValue density_value = DENSITY_API_INST->netDensity(bin_cnt_x, bin_cnt_y, save_path);
  return density_value;
}

void eval_macro_density()
{
}

void eval_macro_pin_density()
{
}

void eval_cell_pin_density()
{
}

void eval_macro_margin()
{
}

void eval_continuous_white_space()
{
}

void eval_macro_channel(float die_size_ratio)
{
}

void eval_cell_hierarchy(const std::string& plot_path, int level, int forward)
{
}

void eval_macro_hierarchy(const std::string& plot_path, int level, int forward)
{
}

void eval_macro_connection(const std::string& plot_path, int level, int forward)
{
}

void eval_macro_pin_connection(const std::string& plot_path, int level, int forward)
{
}

void eval_macro_io_pin_connection(const std::string& plot_path, int level, int forward)
{
}

void eval_inst_density(int inst_status, int eval_flip_flop)
{
}

void eval_pin_density(int inst_status, int level)
{
}

void eval_rudy_cong(int rudy_type, int direction)
{
}

std::vector<float> eval_overflow()
{
  return {};
}

// timing evaluation
void init_timing_eval()
{

}



void plot_flow_value(const std::string& plot_path, const std::string& file_name, const std::string& step, const std::string& value)
{

}

}  // namespace python_interface