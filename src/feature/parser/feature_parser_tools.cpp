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

#include "CTSAPI.hh"
#include "EvalAPI.hpp"
#include "Evaluator.hh"
#include "PLAPI.hh"
#include "PlacerDB.hh"
#include "RTInterface.hpp"
#include "TimingEngine.hh"
#include "ToApi.hpp"
#include "feature_ipl.h"
#include "feature_irt.h"
#include "feature_parser.h"
#include "feature_summary.h"
#include "flow_config.h"
#include "idm.h"
#include "iomanip"
#include "json_parser.h"
#include "report_evaluator.h"

namespace ieda_feature {


json FeatureParser::buildSummaryPL(std::string step)
{
  json summary_pl;
  // 1:全局布局、详细布局、合法化都需要存储的数据参数，需要根据step存储不同的值
  auto place_density = PlacerDBInst.place_density;
  auto pin_density = PlacerDBInst.pin_density;
  auto HPWL = PlacerDBInst.PL_HPWL;
  auto STWL = PlacerDBInst.PL_STWL;
  auto GRWL = PlacerDBInst.PL_GRWL;
  auto congestion = PlacerDBInst.congestion;
  auto tns = PlacerDBInst.tns;
  auto wns = PlacerDBInst.wns;
  auto suggest_freq = PlacerDBInst.suggest_freq;

  // 2:全局布局、详细布局需要存储的数据参数
  if (step == "place") {
    summary_pl["gplace"]["place_density"] = place_density[0];
    summary_pl["gplace"]["pin_density"] = pin_density[0];
    summary_pl["gplace"]["HPWL"] = HPWL[0];
    summary_pl["gplace"]["STWL"] = STWL[0];
    summary_pl["gplace"]["global_routing_WL"] = GRWL[0];
    summary_pl["gplace"]["congestion"] = congestion[0];
    summary_pl["gplace"]["tns"] = tns[0];
    summary_pl["gplace"]["wns"] = wns[0];
    summary_pl["gplace"]["suggest_freq"] = suggest_freq[0];

    summary_pl["dplace"]["place_density"] = place_density[1];
    summary_pl["dplace"]["pin_density"] = pin_density[1];
    summary_pl["dplace"]["HPWL"] = HPWL[1];
    summary_pl["dplace"]["STWL"] = STWL[1];
    summary_pl["dplace"]["global_routing_WL"] = GRWL[1];
    summary_pl["dplace"]["congestion"] = congestion[1];
    summary_pl["dplace"]["tns"] = tns[1];
    summary_pl["dplace"]["wns"] = wns[1];
    summary_pl["dplace"]["suggest_freq"] = suggest_freq[1];

    auto* pl_design = PlacerDBInst.get_design();
    summary_pl["instance"] = pl_design->get_instances_range();
    int fix_inst_cnt = 0;
    for (auto* inst : pl_design->get_instance_list()) {
      if (inst->isFixed()) {
        fix_inst_cnt++;
      }
    }

    summary_pl["fix_instances"] = fix_inst_cnt;
    summary_pl["nets"] = pl_design->get_nets_range();
    summary_pl["total_pins"] = pl_design->get_pins_range();
    summary_pl["core_area"] = std::to_string(PlacerDBInst.get_layout()->get_core_shape().get_width()) + " * "
                              + std::to_string(PlacerDBInst.get_layout()->get_core_shape().get_height());

    summary_pl["bin_number"] = PlacerDBInst.get_placer_config()->get_nes_config().get_bin_cnt_x()
                               * PlacerDBInst.get_placer_config()->get_nes_config().get_bin_cnt_y();
    summary_pl["bin_size"] = std::to_string(PlacerDBInst.bin_size_x) + " * " + std::to_string(PlacerDBInst.bin_size_y);
    summary_pl["overflow_number"] = PlacerDBInst.gp_overflow_number;
    summary_pl["overflow"] = PlacerDBInst.gp_overflow;
  }
  // 3:合法化需要存储的数据参数
  else if (step == "legalization") {
    summary_pl["legalization"]["place_density"] = place_density[2];
    summary_pl["legalization"]["pin_density"] = pin_density[2];
    summary_pl["legalization"]["HPWL"] = HPWL[2];
    summary_pl["legalization"]["STWL"] = STWL[2];
    summary_pl["legalization"]["global_routing_WL"] = GRWL[2];
    summary_pl["legalization"]["congestion"] = congestion[2];
    summary_pl["legalization"]["tns"] = tns[2];
    summary_pl["legalization"]["wns"] = wns[2];
    summary_pl["legalization"]["suggest_freq"] = suggest_freq[2];

    summary_pl["total_movement"] = PlacerDBInst.lg_total_movement;
    summary_pl["max_movement"] = PlacerDBInst.lg_max_movement;
  }
  // std::ofstream& file_stream = ieda::getOutputFileStream(json_path);
  // file_stream << std::setw(4) << summary_pl;

  // ieda::closeFileStream(file_stream);

  // std::cout << std::endl << "Save feature json success, path = " << json_path << std::endl;

  return summary_pl;
}

json FeatureParser::buildSummaryCTS()
{
  // get CTS data
  json summary_cts;

  CTSAPIInst.initEvalInfo();
  summary_cts["design_area"] = dmInst->dieAreaUm();
  summary_cts["design_utilization"] = dmInst->dieUtilization();

  summary_cts["clock_buffer"] = CTSAPIInst.getInsertCellNum();
  summary_cts["clock_buffer_area"] = CTSAPIInst.getInsertCellArea();
  summary_cts["clock_nets"] = _design->get_net_list()->get_num_clock();
  auto path_info = CTSAPIInst.getPathInfos();
  int max_path = path_info[0].max_depth;
  int min_path = path_info[0].min_depth;

  for (auto path : path_info) {
    max_path = std::max(max_path, path.max_depth);
    min_path = std::min(min_path, path.min_depth);
  }
  auto max_level_of_clock_tree = max_path;

  summary_cts["clock_path_min_buffer"] = min_path;
  summary_cts["clock_path_max_buffer"] = max_path;
  summary_cts["max_level_of_clock_tree"] = max_level_of_clock_tree;
  summary_cts["max_clock_wirelength"] = CTSAPIInst.getMaxClockNetWL();
  summary_cts["total_clock_wirelength"] = CTSAPIInst.getTotalClockNetWL();
  // CTSAPIInst.startDbSta();
  auto _timing_engine = ista::TimingEngine::getOrCreateTimingEngine();
  // 可能有多个clk_name，每一个时钟都需要报告tns、wns、freq
  auto clk_list = _timing_engine->getClockList();
  std::ranges::for_each(clk_list, [&](ista::StaClock* clk) {
    auto clk_name = clk->get_clock_name();
    auto setup_tns = _timing_engine->reportTNS(clk_name, AnalysisMode::kMax);
    auto setup_wns = _timing_engine->reportWNS(clk_name, AnalysisMode::kMax);
    auto hold_tns = _timing_engine->reportTNS(clk_name, AnalysisMode::kMin);
    auto hold_wns = _timing_engine->reportWNS(clk_name, AnalysisMode::kMin);
    auto suggest_freq = 1000.0 / (clk->getPeriodNs() - setup_wns);
    summary_cts[clk_name]["setup_tns"] = setup_tns;
    summary_cts[clk_name]["setup_wns"] = setup_wns;
    summary_cts[clk_name]["hold_tns"] = hold_tns;
    summary_cts[clk_name]["hold_wns"] = hold_wns;
    summary_cts[clk_name]["suggest_freq"] = suggest_freq;
  });

  return summary_cts;
}

json FeatureParser::buildSummaryTO(std::string step)
{
  json summary_to;

#if 1
  // instances, nets, total_pins, core_area, utilization
  // 这些指标在summary里都有
  summary_to["instances"] = _design->get_instance_list()->get_num();
  summary_to["nets"] = _design->get_net_list()->get_num();
  // summary_to["total_pins"] =
  summary_to["core_area"] = dmInst->coreAreaUm();
  summary_to["utilization"] = dmInst->coreUtilization();
#endif

  // HPWL, STWL, Global_routing_WL, congestion
  auto& nets = dmInst->get_idb_design()->get_net_list()->get_net_list();
  auto wl_nets = iplf::EvalWrapper::parallelWrap<eval::WLNet>(nets, iplf::EvalWrapper::wrapWLNet);
  summary_to["HPWL"] = EvalInst.evalTotalWL("kHPWL", wl_nets);
  summary_to["STWL"] = EvalInst.evalTotalWL("kFlute", wl_nets);
  // auto Global_routing_WL =
  // auto congestion =

  // max_fanout, min_slew_slack, min_cap_slack

  // before: 初始值，tns，wns，freq
  json summary_subto;
  auto to_eval_data = ToApiInst.getEvalData();
  for (auto eval_data : to_eval_data) {
    auto clk_name = eval_data.name;
    summary_subto[clk_name]["initial_tns"] = eval_data.initial_tns;
    summary_subto[clk_name]["initial_wns"] = eval_data.initial_wns;
    summary_subto[clk_name]["initial_suggest_freq"] = eval_data.initial_freq;
  }

  // after: 优化后的值
  auto _timing_engine = ista::TimingEngine::getOrCreateTimingEngine();
  auto clk_list = _timing_engine->getClockList();

  std::ranges::for_each(clk_list, [&](ista::StaClock* clk) {
    auto clk_name = clk->get_clock_name();
    auto drv_tns = _timing_engine->reportTNS(clk_name, AnalysisMode::kMax);
    auto drv_wns = _timing_engine->reportWNS(clk_name, AnalysisMode::kMax);
    auto suggest_freq = 1000.0 / (clk->getPeriodNs() - drv_wns);
    summary_subto[clk_name]["optimized_tns"] = drv_tns;
    summary_subto[clk_name]["optimized_wns"] = drv_wns;
    summary_subto[clk_name]["optimized_suggest_freq"] = suggest_freq;
  });

  // delta: 迭代的值，优化后的值减去初始值
  for (auto eval_data : to_eval_data) {
    auto clk_name = eval_data.name;
    summary_subto[clk_name]["delta_tns"]
        = static_cast<double>(summary_subto[clk_name]["optimized_tns"]) - static_cast<double>(summary_subto[clk_name]["initial_tns"]);
    summary_subto[clk_name]["delta_wns"]
        = static_cast<double>(summary_subto[clk_name]["optimized_wns"]) - static_cast<double>(summary_subto[clk_name]["initial_wns"]);
    summary_subto[clk_name]["delta_suggest_freq"] = static_cast<double>(summary_subto[clk_name]["optimized_suggest_freq"])
                                                    - static_cast<double>(summary_subto[clk_name]["initial_suggest_freq"]);
  }

  summary_to["sta"] = summary_subto;

  return summary_to;
}

json FeatureParser::buildSummarySTA()
{
  json summary_sta;
  auto timing_engine = ista::TimingEngine::getOrCreateTimingEngine();
  auto* ista = timing_engine->get_ista();
  auto& all_clocks = ista->get_clocks();

  // iterate the clock group of all the clocks.
  for (unsigned id = 1; auto& clock_group : all_clocks) {
    json::value_type path_group;
    path_group["timing_path_group"]["id"] = id++;
    std::string group_name = clock_group->get_clock_name();
    path_group["timing_path_group"]["name"] = group_name;
    double wns = ista->getWNS(group_name.c_str(), ista::AnalysisMode::kMax);
    double tns = ista->getTNS(group_name.c_str(), ista::AnalysisMode::kMax);
    path_group["timing_path_group"]["WNS"] = wns;
    path_group["timing_path_group"]["TNS"] = tns;
    path_group["timing_path_group"]["NVP"] = 0;  // TBD for negative violated points.
    double freq = 1000.0 / (clock_group->getPeriodNs() - wns);
    path_group["timing_path_group"]["FREQ"] = freq;
    double hold_wns = ista->getWNS(group_name.c_str(), ista::AnalysisMode::kMin);
    double hold_tns = ista->getTNS(group_name.c_str(), ista::AnalysisMode::kMin);
    path_group["timing_path_group"]["hold_WNS"] = wns;
    path_group["timing_path_group"]["hold_TNS"] = tns;
    path_group["timing_path_group"]["hold_NVP"] = 0;  // TBD for hold negative violated points.

    FOREACH_MODE(mode)
    {
      json::value_type analysis_mode;
      analysis_mode["analysis_mode"] = (mode == AnalysisMode::kMax) ? "max_delay/setup" : "min_delay/hold";
      analysis_mode["levels_of_logic"] = 0;       // TBD
      analysis_mode["critical_path_length"] = 0;  // TBD
      analysis_mode["critical_path_slack"] = (mode == AnalysisMode::kMax) ? wns : hold_wns;
      analysis_mode["total_negative_slack"] = (mode == AnalysisMode::kMax) ? tns : hold_tns;
      path_group["timing_path_group"]["analysis_mode_infos"].push_back(analysis_mode);
    }
    summary_sta.push_back(path_group);
  }

  return summary_sta;
}

json FeatureParser::buildSummaryDRC()
{
  json summary_drc;

  //   auto drc_map = idrc::DrcAPIInst.getCheckResult();
  //   // summary_drc["short_nums"] = drc_map
  //   for (auto& [key, value] : drc_map) {
  //     summary_drc[key] = value;
  //   }

  return summary_drc;
}

json FeatureParser::buildSummaryRT()
{
  json summary_rt;

  RTSummary& rt_sum = _summary->get_summary_irt();

  json rt_pa;
  for (auto routing_access_point_num : rt_sum.pa_summary.routing_access_point_num_map) {
    rt_pa["routing_access_point_num_map"][std::to_string(routing_access_point_num.first)] = routing_access_point_num.second;
  }
  for (auto type_access_point_num : rt_sum.pa_summary.type_access_point_num_map) {
    rt_pa["routing_access_point_num_map"][type_access_point_num.first] = type_access_point_num.second;
  }
  rt_pa["routing_access_point_num_map"]["total_access_point_num"] = rt_sum.pa_summary.total_access_point_num;
  summary_rt["PA"] = rt_pa;

  auto& sa_sum = rt_sum.sa_summary;
  json rt_sa;
  for (auto routing_supply_num : rt_sum.sa_summary.routing_supply_map) {
    rt_sa["routing_supply_num_map"][std::to_string(routing_supply_num.first)] = routing_supply_num.second;
  }
  rt_sa["routing_supply_num_map"]["total_supply_num"] = rt_sum.sa_summary.total_supply;

  json rt_ir;
  for (auto demand : rt_sum.ir_summary.routing_demand_map) {
    rt_ir["routing_demand_map"][std::to_string(demand.first)] = demand.second;
  }
  rt_ir["routing_demand_map"]["total_demand"] = rt_sum.ir_summary.total_demand;
  for (auto routing_overflow : rt_sum.ir_summary.routing_overflow_map) {
    rt_ir["routing_overflow_map"][std::to_string(routing_overflow.first)] = routing_overflow.second;
  }
  rt_ir["routing_overflow_map"]["total_overflow"] = rt_sum.ir_summary.total_overflow;
  for (auto routing_wire_length : rt_sum.ir_summary.routing_wire_length_map) {
    rt_ir["routing_wire_length_map"][std::to_string(routing_wire_length.first)] = routing_wire_length.second;
  }
  rt_ir["routing_wire_length_map"]["total_wire_length"] = rt_sum.ir_summary.total_wire_length;
  for (auto cut_via_num : rt_sum.ir_summary.cut_via_num_map) {
    rt_ir["routing_cut_via_num_map"][std::to_string(cut_via_num.first)] = cut_via_num.second;
  }
  rt_ir["routing_cut_via_num_map"]["total_cut_via_num"] = rt_sum.ir_summary.total_via_num;
  for (auto timing : rt_sum.ir_summary.timing) {
    rt_ir["routing_timing_map"][timing.first] = timing.second;
  }
  summary_rt["IR"] = rt_ir;

  // // GR
  for (auto [id, gr_sum] : rt_sum.iter_gr_summary_map) {
    json rt_gr;
    // 和ir一样
    for (auto demand : gr_sum.routing_demand_map) {
      rt_gr["routing_demand_map"][std::to_string(demand.first)] = demand.second;
    }
    rt_gr["routing_demand_map"]["total_demand"] = gr_sum.total_demand;
    for (auto routing_overflow : gr_sum.routing_overflow_map) {
      rt_gr["routing_overflow_map"][std::to_string(routing_overflow.first)] = routing_overflow.second;
    }
    rt_gr["routing_overflow_map"]["total_overflow"] = gr_sum.total_overflow;
    for (auto routing_wire_length : gr_sum.routing_wire_length_map) {
      rt_gr["routing_wire_length_map"][std::to_string(routing_wire_length.first)] = routing_wire_length.second;
    }
    rt_gr["routing_wire_length_map"]["total_wire_length"] = gr_sum.total_wire_length;
    for (auto cut_via_num : gr_sum.cut_via_num_map) {
      rt_gr["routing_cut_via_num_map"][std::to_string(cut_via_num.first)] = cut_via_num.second;
    }
    rt_gr["routing_cut_via_num_map"]["total_cut_via_num"] = gr_sum.total_via_num;
    for (auto timing : gr_sum.timing) {
      rt_gr["routing_timing_map"][timing.first] = timing.second;
    }
    summary_rt["GR"][std::to_string(id)] = rt_gr;
  }
  // TA
  json rt_ta;
  // wirelength, violation
  for (auto routing_wire_length : rt_sum.ta_summary.routing_wire_length_map) {
    rt_ta["routing_wire_length_map"][std::to_string(routing_wire_length.first)] = routing_wire_length.second;
  }
  rt_ta["routing_wire_length_map"]["total_wire_length"] = rt_sum.ta_summary.total_wire_length;
  for (auto routing_violation : rt_sum.ta_summary.routing_violation_num_map) {
    rt_ta["routing_violation_map"][std::to_string(routing_violation.first)] = routing_violation.second;
  }
  rt_ta["routing_violation_map"]["total_violation"] = rt_sum.ta_summary.total_violation_num;
  summary_rt["TA"] = rt_ta;

  // DR
  for (auto [id, dr_sum] : rt_sum.iter_dr_summary_map) {
    json rt_dr;
    for (auto routing_wire_length : dr_sum.routing_wire_length_map) {
      rt_dr["routing_wire_length_map"][std::to_string(routing_wire_length.first)] = routing_wire_length.second;
    }
    rt_dr["routing_wire_length_map"]["total_wire_length"] = dr_sum.total_wire_length;
    for (auto cut_via_num : dr_sum.cut_via_num_map) {
      rt_dr["routing_cut_via_num_map"][std::to_string(cut_via_num.first)] = cut_via_num.second;
    }
    rt_dr["routing_cut_via_num_map"]["total_cut_via_num"] = dr_sum.total_via_num;
    // violation
    for (auto routing_violation : dr_sum.routing_violation_num_map) {
      rt_dr["routing_violation_map"][std::to_string(routing_violation.first)] = routing_violation.second;
    }
    rt_dr["routing_violation_map"]["total_violation"] = dr_sum.total_violation_num;
    for (auto routing_patch_num : dr_sum.routing_patch_num_map) {
      rt_dr["routing_patch_num_map"][std::to_string(routing_patch_num.first)] = routing_patch_num.second;
    }
    rt_dr["routing_patch_num_map"]["total_patch_num"] = dr_sum.total_patch_num;
    for (auto timing : dr_sum.timing) {
      rt_dr["routing_timing_map"][timing.first] = timing.second;
    }
    summary_rt["DR"][std::to_string(id)] = rt_dr;
  }
  return summary_rt;
}

}  // namespace ieda_feature
