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


// congestion evaluation
ieval::CongestionValue rudy_congestion(int bin_cnt_x, int bin_cnt_y, const std::string& save_path)
{
  ieval::CongestionValue congestion_value = CONGESTION_API_INST->rudyCongestion(bin_cnt_x, bin_cnt_y, save_path);
  return congestion_value;
}

ieval::CongestionValue lut_rudy_congestion(int bin_cnt_x, int bin_cnt_y, const std::string& save_path)
{
  ieval::CongestionValue congestion_value = CONGESTION_API_INST->lutRudyCongestion(bin_cnt_x, bin_cnt_y, save_path);
  return congestion_value;
}

ieval::CongestionValue egr_congestion(const std::string& save_path)
{
  ieval::CongestionValue congestion_value = CONGESTION_API_INST->egrCongestion(save_path);
  return congestion_value;
}


// timing and power evaluation
ieval::TimingSummary timing_power_hpwl()
{
    TimingPower_API_INST->evalTiming("HPWL", false);

    std::map<std::string, ieval::TimingSummary> timing_summary = TimingPower_API_INST->evalDesign();
    ieval::TimingSummary hpwl_timing_eval_summary;

    auto hpwl_timing_summary = timing_summary.at("HPWL");

    std::for_each(hpwl_timing_summary.clock_timings.begin(), hpwl_timing_summary.clock_timings.end(),
                  [&hpwl_timing_eval_summary](const auto& clock_timing) {
                      hpwl_timing_eval_summary.clock_timings.push_back({
                          clock_timing.clock_name,
                          clock_timing.setup_tns,
                          clock_timing.setup_wns,
                          clock_timing.hold_tns,
                          clock_timing.hold_wns,
                          clock_timing.suggest_freq
                      });
                  });

    hpwl_timing_eval_summary.static_power = hpwl_timing_summary.static_power;
    hpwl_timing_eval_summary.dynamic_power = hpwl_timing_summary.dynamic_power;

    return hpwl_timing_eval_summary;
}

ieval::TimingSummary timing_power_stwl()
{
    TimingPower_API_INST->evalTiming("FLUTE", false);

    std::map<std::string, ieval::TimingSummary> timing_summary = TimingPower_API_INST->evalDesign();
    ieval::TimingSummary stwl_timing_eval_summary;

    auto stwl_timing_summary = timing_summary.at("FLUTE");

    std::for_each(stwl_timing_summary.clock_timings.begin(), stwl_timing_summary.clock_timings.end(),
                  [&stwl_timing_eval_summary](const auto& clock_timing) {
                      stwl_timing_eval_summary.clock_timings.push_back({
                          clock_timing.clock_name,
                          clock_timing.setup_tns,
                          clock_timing.setup_wns,
                          clock_timing.hold_tns,
                          clock_timing.hold_wns,
                          clock_timing.suggest_freq
                      });
                  });

    stwl_timing_eval_summary.static_power = stwl_timing_summary.static_power;
    stwl_timing_eval_summary.dynamic_power = stwl_timing_summary.dynamic_power;

    return stwl_timing_eval_summary;
}

ieval::TimingSummary timing_power_egr()
{
    TimingPower_API_INST->evalTiming("EGR", false);

    std::map<std::string, ieval::TimingSummary> timing_summary = TimingPower_API_INST->evalDesign();
    ieval::TimingSummary egr_timing_eval_summary;

    auto egr_timing_summary = timing_summary.at("EGR");

    std::for_each(egr_timing_summary.clock_timings.begin(), egr_timing_summary.clock_timings.end(),
                  [&egr_timing_eval_summary](const auto& clock_timing) {
                      egr_timing_eval_summary.clock_timings.push_back({
                          clock_timing.clock_name,
                          clock_timing.setup_tns,
                          clock_timing.setup_wns,
                          clock_timing.hold_tns,
                          clock_timing.hold_wns,
                          clock_timing.suggest_freq
                      });
                  });

    egr_timing_eval_summary.static_power = egr_timing_summary.static_power;
    egr_timing_eval_summary.dynamic_power = egr_timing_summary.dynamic_power;

    return egr_timing_eval_summary;
}

// other evaluation (TO BE DONE)
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


std::vector<float> eval_overflow()
{
  return {};
}


}  // namespace python_interface