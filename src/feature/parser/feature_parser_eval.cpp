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
/**
 * @project		iEDA
 * @file		feature_parser.cpp
 * @author		Yell
 * @date		10/08/2023
 * @version		0.1
 * @description


        feature parser
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "EvalAPI.hpp"
#include "Evaluator.hh"
#include "feature_parser.h"
#include "feature_summary.h"
#include "flow_config.h"
#include "idm.h"
#include "iomanip"
#include "json_parser.h"
// #include "report_evaluator.h"

namespace ieda_feature {

bool FeatureParser::buildSummaryMap(std::string csv_path, int bin_cnt_x, int bin_cnt_y)
{
  eval::EvalAPI& eval_api = eval::EvalAPI::initInst();
  eval_api.initCongDataFromIDB(bin_cnt_x, bin_cnt_y);

  auto inst_status = eval::INSTANCE_STATUS::kFixed;
  eval_api.evalInstDens(inst_status);
  eval_api.plotBinValue(csv_path, "macro_density", eval::CONGESTION_TYPE::kInstDens);
  eval_api.evalPinDens(inst_status);
  eval_api.plotBinValue(csv_path, "macro_pin_density", eval::CONGESTION_TYPE::kPinDens);
  eval_api.evalNetDens(inst_status);
  eval_api.plotBinValue(csv_path, "macro_net_density", eval::CONGESTION_TYPE::kNetCong);

  eval_api.plotMacroChannel(0.5, csv_path + "macro_channel.csv");
  eval_api.evalMacroMargin();
  eval_api.plotBinValue(csv_path, "macro_margin_h", eval::CONGESTION_TYPE::kMacroMarginH);
  eval_api.plotBinValue(csv_path, "macro_margin_v", eval::CONGESTION_TYPE::kMacroMarginV);
  double space_ratio = eval_api.evalMaxContinuousSpace();
  eval_api.plotBinValue(csv_path, "macro_continuous_white_space", eval::CONGESTION_TYPE::kContinuousWS);
  eval_api.evalIOPinAccess(csv_path + "io_pin_access.csv");

  std::cout << std::endl << "Save feature map success, path = " << csv_path << std::endl;
  return true;
}

json FeatureParser::buildSummaryWirelength()
{
  json wirelength_info;

  auto& total_wl_summary = _summary->get_summary_wirelength_eval();

  wirelength_info["HPWL"] = total_wl_summary.HPWL;
  wirelength_info["FLUTE"] = total_wl_summary.FLUTE;
  wirelength_info["HTree"] = total_wl_summary.HTree;
  wirelength_info["VTree"] = total_wl_summary.VTree;
  wirelength_info["GRWL"] = total_wl_summary.GRWL;

  return wirelength_info;
}

json FeatureParser::buildSummaryDensity()
{
  json density_info;

  auto& density_map_summary = _summary->get_summary_density_eval();

  density_info["cell"]["macro_density"] = density_map_summary.cell_map_summary.macro_density;
  density_info["cell"]["stdcell_density"] = density_map_summary.cell_map_summary.stdcell_density;
  density_info["cell"]["allcell_density"] = density_map_summary.cell_map_summary.allcell_density;

  density_info["pin"]["macro_pin_density"] = density_map_summary.pin_map_summary.macro_pin_density;
  density_info["pin"]["stdcell_pin_density"] = density_map_summary.pin_map_summary.stdcell_pin_density;
  density_info["pin"]["allcell_pin_density"] = density_map_summary.pin_map_summary.allcell_pin_density;

  density_info["net"]["local_net_density"] = density_map_summary.net_map_summary.local_net_density;
  density_info["net"]["global_net_density"] = density_map_summary.net_map_summary.global_net_density;
  density_info["net"]["allnet_density"] = density_map_summary.net_map_summary.allnet_density;

  return density_info;
}

json FeatureParser::buildSummaryCongestion()
{
  json congestion_info;

  auto& congestion_summary = _summary->get_summary_congestion_eval();

  congestion_info["map"]["egr"]["horizontal"] = congestion_summary.egr_map_summary.horizontal_sum;
  congestion_info["map"]["egr"]["vertical"] = congestion_summary.egr_map_summary.vertical_sum;
  congestion_info["map"]["egr"]["union"] = congestion_summary.egr_map_summary.union_sum;

  congestion_info["map"]["rudy"]["horizontal"] = congestion_summary.rudy_map_summary.rudy_horizontal;
  congestion_info["map"]["rudy"]["vertical"] = congestion_summary.rudy_map_summary.rudy_vertical;
  congestion_info["map"]["rudy"]["union"] = congestion_summary.rudy_map_summary.rudy_union;

  congestion_info["map"]["lutrudy"]["horizontal"] = congestion_summary.rudy_map_summary.lutrudy_horizontal;
  congestion_info["map"]["lutrudy"]["vertical"] = congestion_summary.rudy_map_summary.lutrudy_vertical;
  congestion_info["map"]["lutrudy"]["union"] = congestion_summary.rudy_map_summary.lutrudy_union;

  congestion_info["overflow"]["total"]["horizontal"] = congestion_summary.overflow_summary.total_overflow_horizontal;
  congestion_info["overflow"]["total"]["vertical"] = congestion_summary.overflow_summary.total_overflow_vertical;
  congestion_info["overflow"]["total"]["union"] = congestion_summary.overflow_summary.total_overflow_union;

  congestion_info["overflow"]["max"]["horizontal"] = congestion_summary.overflow_summary.max_overflow_horizontal;
  congestion_info["overflow"]["max"]["vertical"] = congestion_summary.overflow_summary.max_overflow_vertical;
  congestion_info["overflow"]["max"]["union"] = congestion_summary.overflow_summary.max_overflow_union;

  congestion_info["overflow"]["top average"]["horizontal"] = congestion_summary.overflow_summary.weighted_average_overflow_horizontal;
  congestion_info["overflow"]["top average"]["vertical"] = congestion_summary.overflow_summary.weighted_average_overflow_vertical;
  congestion_info["overflow"]["top average"]["union"] = congestion_summary.overflow_summary.weighted_average_overflow_union;

  congestion_info["utilization"]["rudy"]["max"]["horizontal"] = congestion_summary.rudy_utilization_summary.max_utilization_horizontal;
  congestion_info["utilization"]["rudy"]["max"]["vertical"] = congestion_summary.rudy_utilization_summary.max_utilization_vertical;
  congestion_info["utilization"]["rudy"]["max"]["union"] = congestion_summary.rudy_utilization_summary.max_utilization_union;
  congestion_info["utilization"]["rudy"]["top average"]["horizontal"]
      = congestion_summary.rudy_utilization_summary.weighted_average_utilization_horizontal;
  congestion_info["utilization"]["rudy"]["top average"]["vertical"]
      = congestion_summary.rudy_utilization_summary.weighted_average_utilization_vertical;
  congestion_info["utilization"]["rudy"]["top average"]["union"]
      = congestion_summary.rudy_utilization_summary.weighted_average_utilization_union;

  congestion_info["utilization"]["lutrudy"]["max"]["horizontal"]
      = congestion_summary.lutrudy_utilization_summary.max_utilization_horizontal;
  congestion_info["utilization"]["lutrudy"]["max"]["vertical"] = congestion_summary.lutrudy_utilization_summary.max_utilization_vertical;
  congestion_info["utilization"]["lutrudy"]["max"]["union"] = congestion_summary.lutrudy_utilization_summary.max_utilization_union;
  congestion_info["utilization"]["lutrudy"]["top average"]["horizontal"]
      = congestion_summary.lutrudy_utilization_summary.weighted_average_utilization_horizontal;
  congestion_info["utilization"]["lutrudy"]["top average"]["vertical"]
      = congestion_summary.lutrudy_utilization_summary.weighted_average_utilization_vertical;
  congestion_info["utilization"]["lutrudy"]["top average"]["union"]
      = congestion_summary.lutrudy_utilization_summary.weighted_average_utilization_union;

  return congestion_info;
}

json FeatureParser::buildSummaryTiming()
{
  json timing;
  auto add_routing_timing = [&](const std::string& routing_type, const std::vector<ClockTiming>& clock_timings) {
    for (size_t i = 0; i < clock_timings.size(); i++) {
      timing[routing_type][i]["clock_name"] = clock_timings[i].clock_name;
      timing[routing_type][i]["setup_tns"] = clock_timings[i].setup_tns;
      timing[routing_type][i]["setup_wns"] = clock_timings[i].setup_wns;
      timing[routing_type][i]["hold_tns"] = clock_timings[i].hold_tns;
      timing[routing_type][i]["hold_wns"] = clock_timings[i].hold_wns;
      timing[routing_type][i]["suggest_freq"] = clock_timings[i].suggest_freq;
    }
  };

  auto routing_timing_summary = _summary->get_summary_timing_eval();
  // TBD
  // add_routing_timing("WLM", routing_timing_summary.wlm_timing_eval_summary.clock_timings);
  add_routing_timing("HPWL", routing_timing_summary.hpwl_timing_eval_summary.clock_timings);
  add_routing_timing("FLUTE", routing_timing_summary.flute_timing_eval_summary.clock_timings);
  add_routing_timing("EGR", routing_timing_summary.egr_timing_eval_summary.clock_timings);
  add_routing_timing("DR", routing_timing_summary.dr_timing_eval_summary.clock_timings);

  return timing;
}

json FeatureParser::buildSummaryPower()
{
  json power;
  auto add_routing_power = [&](const std::string& routing_type, const PowerInfo& power_info) {
    power[routing_type]["static_power"] = power_info.static_power;
    power[routing_type]["dynamic_power"] = power_info.dynamic_power;
  };

  auto timing_summary = _summary->get_summary_timing_eval();

  // TBD
  // add_routing_power("WLM", timing_summary.wlm_timing_eval_summary.power_info);
  add_routing_power("HPWL", timing_summary.hpwl_timing_eval_summary.power_info);
  add_routing_power("FLUTE", timing_summary.flute_timing_eval_summary.power_info);
  add_routing_power("EGR", timing_summary.egr_timing_eval_summary.power_info);
  add_routing_power("DR", timing_summary.dr_timing_eval_summary.power_info);

  return power;
}

}  // namespace ieda_feature
