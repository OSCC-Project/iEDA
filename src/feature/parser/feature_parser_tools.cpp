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
#include "feature_irt.h"
#include "feature_parser.h"
#include "feature_summary.h"

namespace ieda_feature {

json FeatureParser::buildSummaryRT()
{
  json json_rt;

  RTSummary& summary_irt = _summary->get_summary_irt();

  {
    // PASummary
    std::vector<json> pa_json_list;
    for (auto [iter, pa_summary] : summary_irt.iter_pa_summary_map) {
      json pa_json;
      pa_json["iter"] = iter;
      for (auto& [routing_layer_idx, wire_length] : pa_summary.routing_wire_length_map) {
        pa_json["routing_wire_length_map"][std::to_string(routing_layer_idx)] = wire_length;
      }
      pa_json["total_wire_length"] = pa_summary.total_wire_length;
      for (auto& [cut_layer_idx, via_num] : pa_summary.cut_via_num_map) {
        pa_json["cut_via_num_map"][std::to_string(cut_layer_idx)] = via_num;
      }
      pa_json["total_via_num"] = pa_summary.total_via_num;
      for (auto& [routing_layer_idx, patch_num] : pa_summary.routing_patch_num_map) {
        pa_json["routing_patch_num_map"][std::to_string(routing_layer_idx)] = patch_num;
      }
      pa_json["total_patch_num"] = pa_summary.total_patch_num;
      for (auto& [routing_layer_idx, violation_num] : pa_summary.routing_violation_num_map) {
        pa_json["routing_violation_num_map"][std::to_string(routing_layer_idx)] = violation_num;
      }
      pa_json["total_violation_num"] = pa_summary.total_violation_num;
      pa_json_list.push_back(pa_json);
    }
    json_rt["PA"] = pa_json_list;
  }

  {
    // SASummary
    json sa_json;
    for (auto& [routing_layer_idx, supply] : summary_irt.sa_summary.routing_supply_map) {
      sa_json["routing_supply_map"][std::to_string(routing_layer_idx)] = supply;
    }
    sa_json["total_supply"] = summary_irt.sa_summary.total_supply;
    json_rt["SA"] = sa_json;
  }

  {
    // TGSummary
    json tg_json;
    tg_json["total_demand"] = summary_irt.tg_summary.total_demand;
    tg_json["total_overflow"] = summary_irt.tg_summary.total_overflow;
    tg_json["total_wire_length"] = summary_irt.tg_summary.total_wire_length;
    for (auto& [clock_name, timing] : summary_irt.tg_summary.clock_timing_map) {
      tg_json["clock_timing_map"]["clock_name"] = clock_name;
      tg_json["clock_timing_map"]["timing"] = timing;
    }
    for (auto& [type, power] : summary_irt.tg_summary.type_power_map) {
      tg_json["type_power_map"]["type"] = type;
      tg_json["type_power_map"]["power"] = power;
    }
    json_rt["TG"] = tg_json;
  }

  {
    // LASummary
    json la_json;
    for (auto& [routing_layer_idx, demand] : summary_irt.la_summary.routing_demand_map) {
      la_json["routing_demand_map"][std::to_string(routing_layer_idx)] = demand;
    }
    la_json["total_demand"] = summary_irt.la_summary.total_demand;
    for (auto& [routing_layer_idx, overflow] : summary_irt.la_summary.routing_overflow_map) {
      la_json["routing_overflow_map"][std::to_string(routing_layer_idx)] = overflow;
    }
    la_json["total_overflow"] = summary_irt.la_summary.total_overflow;
    for (auto& [routing_layer_idx, wire_length] : summary_irt.la_summary.routing_wire_length_map) {
      la_json["routing_wire_length_map"][std::to_string(routing_layer_idx)] = wire_length;
    }
    la_json["total_wire_length"] = summary_irt.la_summary.total_wire_length;
    for (auto& [cut_layer_idx, via_num] : summary_irt.la_summary.cut_via_num_map) {
      la_json["cut_via_num_map"][std::to_string(cut_layer_idx)] = via_num;
    }
    la_json["total_via_num"] = summary_irt.la_summary.total_via_num;
    for (auto& [clock_name, timing] : summary_irt.la_summary.clock_timing_map) {
      la_json["clock_timing_map"]["clock_name"] = clock_name;
      la_json["clock_timing_map"]["timing"] = timing;
    }
    for (auto& [type, power] : summary_irt.la_summary.type_power_map) {
      la_json["type_power_map"]["type"] = type;
      la_json["type_power_map"]["power"] = power;
    }
    json_rt["LA"] = la_json;
  }

  {
    // SRSummary
    std::vector<json> sr_json_list;
    for (auto [iter, sr_summary] : summary_irt.iter_sr_summary_map) {
      json sr_json;
      sr_json["iter"] = iter;
      for (auto& [routing_layer_idx, demand] : sr_summary.routing_demand_map) {
        sr_json["routing_demand_map"][std::to_string(routing_layer_idx)] = demand;
      }
      sr_json["total_demand"] = sr_summary.total_demand;
      for (auto& [routing_layer_idx, overflow] : sr_summary.routing_overflow_map) {
        sr_json["routing_overflow_map"][std::to_string(routing_layer_idx)] = overflow;
      }
      sr_json["total_overflow"] = sr_summary.total_overflow;
      for (auto& [routing_layer_idx, wire_length] : sr_summary.routing_wire_length_map) {
        sr_json["routing_wire_length_map"][std::to_string(routing_layer_idx)] = wire_length;
      }
      sr_json["total_wire_length"] = sr_summary.total_wire_length;
      for (auto& [cut_layer_idx, via_num] : sr_summary.cut_via_num_map) {
        sr_json["cut_via_num_map"][std::to_string(cut_layer_idx)] = via_num;
      }
      sr_json["total_via_num"] = sr_summary.total_via_num;
      for (auto& [clock_name, timing] : sr_summary.clock_timing_map) {
        sr_json["clock_timing_map"]["clock_name"] = clock_name;
        sr_json["clock_timing_map"]["timing"] = timing;
      }
      for (auto& [type, power] : sr_summary.type_power_map) {
        sr_json["type_power_map"]["type"] = type;
        sr_json["type_power_map"]["power"] = power;
      }
      sr_json_list.push_back(sr_json);
    }
    json_rt["SR"] = sr_json_list;
  }

  {
    // TASummary
    json ta_json;
    for (auto& [routing_layer_idx, wire_length] : summary_irt.ta_summary.routing_wire_length_map) {
      ta_json["routing_wire_length_map"][std::to_string(routing_layer_idx)] = wire_length;
    }
    ta_json["total_wire_length"] = summary_irt.ta_summary.total_wire_length;
    for (auto& [routing_layer_idx, violation_num] : summary_irt.ta_summary.routing_violation_num_map) {
      ta_json["routing_violation_num_map"][std::to_string(routing_layer_idx)] = violation_num;
    }
    ta_json["total_violation_num"] = summary_irt.ta_summary.total_violation_num;
    json_rt["TA"] = ta_json;
  }

  {
    // DRSummary
    std::vector<json> dr_json_list;
    for (auto [iter, dr_summary] : summary_irt.iter_dr_summary_map) {
      json dr_json;
      dr_json["iter"] = iter;
      for (auto& [routing_layer_idx, wire_length] : dr_summary.routing_wire_length_map) {
        dr_json["routing_wire_length_map"][std::to_string(routing_layer_idx)] = wire_length;
      }
      dr_json["total_wire_length"] = dr_summary.total_wire_length;
      for (auto& [cut_layer_idx, via_num] : dr_summary.cut_via_num_map) {
        dr_json["cut_via_num_map"][std::to_string(cut_layer_idx)] = via_num;
      }
      dr_json["total_via_num"] = dr_summary.total_via_num;
      for (auto& [routing_layer_idx, patch_num] : dr_summary.routing_patch_num_map) {
        dr_json["routing_patch_num_map"][std::to_string(routing_layer_idx)] = patch_num;
      }
      dr_json["total_patch_num"] = dr_summary.total_patch_num;
      for (auto& [routing_layer_idx, violation_num] : dr_summary.routing_violation_num_map) {
        dr_json["routing_violation_num_map"][std::to_string(routing_layer_idx)] = violation_num;
      }
      dr_json["total_violation_num"] = dr_summary.total_violation_num;
      for (auto& [clock_name, timing] : dr_summary.clock_timing_map) {
        dr_json["clock_timing_map"]["clock_name"] = clock_name;
        dr_json["clock_timing_map"]["timing"] = timing;
      }
      for (auto& [type, power] : dr_summary.type_power_map) {
        dr_json["type_power_map"]["type"] = type;
        dr_json["type_power_map"]["power"] = power;
      }
      dr_json_list.push_back(dr_json);
    }
    json_rt["DR"] = dr_json_list;
  }

  {
    // VRSummary
    json vr_json;
    for (auto& [routing_layer_idx, wire_length] : summary_irt.vr_summary.routing_wire_length_map) {
      vr_json["routing_wire_length_map"][std::to_string(routing_layer_idx)] = wire_length;
    }
    vr_json["total_wire_length"] = summary_irt.vr_summary.total_wire_length;
    for (auto& [cut_layer_idx, via_num] : summary_irt.vr_summary.cut_via_num_map) {
      vr_json["cut_via_num_map"][std::to_string(cut_layer_idx)] = via_num;
    }
    vr_json["total_via_num"] = summary_irt.vr_summary.total_via_num;
    for (auto& [routing_layer_idx, patch_num] : summary_irt.vr_summary.routing_patch_num_map) {
      vr_json["routing_patch_num_map"][std::to_string(routing_layer_idx)] = patch_num;
    }
    vr_json["total_patch_num"] = summary_irt.vr_summary.total_patch_num;
    for (auto& [routing_layer_idx, violation_num] : summary_irt.vr_summary.within_net_routing_violation_num_map) {
      vr_json["within_net_routing_violation_num_map"][std::to_string(routing_layer_idx)] = violation_num;
    }
    vr_json["within_net_total_violation_num"] = summary_irt.vr_summary.within_net_total_violation_num;
    for (auto& [routing_layer_idx, violation_num] : summary_irt.vr_summary.among_net_routing_violation_num_map) {
      vr_json["among_net_routing_violation_num_map"][std::to_string(routing_layer_idx)] = violation_num;
    }
    vr_json["among_net_total_violation_num"] = summary_irt.vr_summary.among_net_total_violation_num;
    for (auto& [clock_name, timing] : summary_irt.vr_summary.clock_timing_map) {
      vr_json["clock_timing_map"]["clock_name"] = clock_name;
      vr_json["clock_timing_map"]["timing"] = timing;
    }
    for (auto& [type, power] : summary_irt.vr_summary.type_power_map) {
      vr_json["type_power_map"]["type"] = type;
      vr_json["type_power_map"]["power"] = power;
    }
    json_rt["VR"] = vr_json;
  }

  return json_rt;
}

json FeatureParser::buildSummaryPL(std::string step)
{
  json summary_pl;

  PlaceSummary& pl_summary = _summary->get_summary_ipl();

  if (step == "place") {
    summary_pl["gplace"]["place_density"] = pl_summary.gplace.place_density;
    summary_pl["gplace"]["HPWL"] = pl_summary.gplace.HPWL;
    summary_pl["gplace"]["STWL"] = pl_summary.gplace.STWL;

    summary_pl["dplace"]["place_density"] = pl_summary.dplace.place_density;
    summary_pl["dplace"]["HPWL"] = pl_summary.dplace.HPWL;
    summary_pl["dplace"]["STWL"] = pl_summary.dplace.STWL;

    summary_pl["instance_cnt"] = pl_summary.instance_cnt;
    summary_pl["fix_inst_cnt"] = pl_summary.fix_inst_cnt;
    summary_pl["net_cnt"] = pl_summary.net_cnt;
    summary_pl["total_pins"] = pl_summary.total_pins;
    summary_pl["bin_number"] = pl_summary.bin_number;
    summary_pl["bin_size_x"] = pl_summary.bin_size_x;
    summary_pl["bin_size_y"] = pl_summary.bin_size_y;
    summary_pl["overflow_number"] = pl_summary.overflow_number;
    summary_pl["overflow"] = pl_summary.overflow;
  }
  // 3: Data parameters required for legalization.
  else if (step == "legalization") {
    summary_pl["HPWL"] = pl_summary.lg_summary.pl_common_summary.HPWL;
    summary_pl["STWL"] = pl_summary.lg_summary.pl_common_summary.STWL;
    summary_pl["total_movement"] = pl_summary.lg_summary.lg_total_movement;
    summary_pl["max_movement"] = pl_summary.lg_summary.lg_max_movement;
  }

  return summary_pl;
}

json FeatureParser::buildSummaryCTS()
{
  json json_cts;

  CTSSummary& summary = _summary->get_summary_icts();

  json_cts["buffer_num"] = summary.buffer_num;
  json_cts["buffer_area"] = summary.buffer_area;
  json_cts["clock_path_min_buffer"] = summary.clock_path_min_buffer;
  json_cts["clock_path_max_buffer"] = summary.clock_path_max_buffer;
  json_cts["max_level_of_clock_tree"] = summary.max_level_of_clock_tree;
  json_cts["max_clock_wirelength"] = summary.max_clock_wirelength;
  json_cts["total_clock_wirelength"] = summary.total_clock_wirelength;

  json json_timing;
  for (int i = 0; i < (int) summary.clocks_timing.size(); ++i) {
    auto clock_timing = summary.clocks_timing[i];

    json_timing[i]["clock_name"] = clock_timing.clock_name;
    json_timing[i]["setup_tns"] = clock_timing.setup_tns;
    json_timing[i]["setup_wns"] = clock_timing.setup_wns;
    json_timing[i]["hold_tns"] = clock_timing.hold_tns;
    json_timing[i]["hold_wns"] = clock_timing.hold_wns;
    json_timing[i]["suggest_freq"] = clock_timing.suggest_freq;
  }

  json_cts["clocks_timing"] = json_timing;

  return json_cts;
}

json FeatureParser::buildSummaryNetOpt()
{
  json json_netopt;

  NetOptSummary& summary = _summary->get_summary_ino();

  json json_clock_timings;
  for (int i = 0; i < (int) summary.clock_timings.size(); ++i) {
    NOClockTimingCmp clock_timing = summary.clock_timings[i];

    json_clock_timings[i]["clock_name"] = clock_timing.clock_name;
    json_clock_timings[i]["origin_setup_tns"] = clock_timing.origin.setup_tns;
    json_clock_timings[i]["origin_setup_wns"] = clock_timing.origin.setup_wns;
    json_clock_timings[i]["origin_hold_tns"] = clock_timing.origin.hold_tns;
    json_clock_timings[i]["origin_hold_wns"] = clock_timing.origin.hold_wns;
    json_clock_timings[i]["origin_suggest_freq"] = clock_timing.origin.suggest_freq;
    json_clock_timings[i]["opt_setup_tns"] = clock_timing.opt.setup_tns;
    json_clock_timings[i]["opt_setup_wns"] = clock_timing.opt.setup_wns;
    json_clock_timings[i]["opt_hold_tns"] = clock_timing.opt.hold_tns;
    json_clock_timings[i]["opt_hold_wns"] = clock_timing.opt.hold_wns;
    json_clock_timings[i]["opt_suggest_freq"] = clock_timing.opt.suggest_freq;
    json_clock_timings[i]["delta_setup_tns"] = clock_timing.delta.setup_tns;
    json_clock_timings[i]["delta_setup_wns"] = clock_timing.delta.setup_wns;
    json_clock_timings[i]["delta_hold_tns"] = clock_timing.delta.hold_tns;
    json_clock_timings[i]["delta_hold_wns"] = clock_timing.delta.hold_wns;
    json_clock_timings[i]["delta_suggest_freq"] = clock_timing.delta.suggest_freq;
  }

  json_netopt["clocks_timing"] = json_clock_timings;

  return json_netopt;
}

json FeatureParser::buildSummaryTO(std::string step)
{
  auto step_summary = [](std::string step, FeatureSummary* summary) {
    if (step == "optDrv") {
      return summary->get_summary_ito_optdrv();
    } else if (step == "optHold") {
      return summary->get_summary_ito_opthold();
    } else {
      return summary->get_summary_ito_optsetup();
    }
  };

  json summary_to;

  TimingOptSummary summary = step_summary(step, _summary);

  summary_to["HPWL"] = summary.HPWL;
  summary_to["STWL"] = summary.STWL;

  json json_clock_timings;
  for (size_t i = 0; i < summary.clock_timings.size(); ++i) {
    TOClockTimingCmp clock_timing = summary.clock_timings[i];

    json_clock_timings[i]["clock_name"] = clock_timing.clock_name;
    json_clock_timings[i]["origin_tns"] = clock_timing.origin.tns;
    json_clock_timings[i]["origin_wns"] = clock_timing.origin.wns;
    json_clock_timings[i]["origin_suggest_freq"] = clock_timing.origin.suggest_freq;
    json_clock_timings[i]["opt_tns"] = clock_timing.opt.tns;
    json_clock_timings[i]["opt_wns"] = clock_timing.opt.wns;
    json_clock_timings[i]["opt_suggest_freq"] = clock_timing.opt.suggest_freq;
    json_clock_timings[i]["delta_tns"] = clock_timing.delta.tns;
    json_clock_timings[i]["delta_wns"] = clock_timing.delta.wns;
    json_clock_timings[i]["delta_suggest_freq"] = clock_timing.delta.suggest_freq;
  }

  summary_to["clocks_timing"] = json_clock_timings;

  return summary_to;
}

json FeatureParser::buildSummarySTA()
{
  json summary_sta;
  //   auto timing_engine = ista::TimingEngine::getOrCreateTimingEngine();
  //   auto* ista = timing_engine->get_ista();
  //   auto& all_clocks = ista->get_clocks();

  //   // iterate the clock group of all the clocks.
  //   for (unsigned id = 1; auto& clock_group : all_clocks) {
  //     json::value_type path_group;
  //     path_group["timing_path_group"]["id"] = id++;
  //     std::string group_name = clock_group->get_clock_name();
  //     path_group["timing_path_group"]["name"] = group_name;
  //     double wns = ista->getWNS(group_name.c_str(), ista::AnalysisMode::kMax);
  //     double tns = ista->getTNS(group_name.c_str(), ista::AnalysisMode::kMax);
  //     path_group["timing_path_group"]["WNS"] = wns;
  //     path_group["timing_path_group"]["TNS"] = tns;
  //     path_group["timing_path_group"]["NVP"] = 0;  // TBD for negative violated points.
  //     double freq = 1000.0 / (clock_group->getPeriodNs() - wns);
  //     path_group["timing_path_group"]["FREQ"] = freq;
  //     double hold_wns = ista->getWNS(group_name.c_str(), ista::AnalysisMode::kMin);
  //     double hold_tns = ista->getTNS(group_name.c_str(), ista::AnalysisMode::kMin);
  //     path_group["timing_path_group"]["hold_WNS"] = wns;
  //     path_group["timing_path_group"]["hold_TNS"] = tns;
  //     path_group["timing_path_group"]["hold_NVP"] = 0;  // TBD for hold negative violated points.

  //     FOREACH_MODE(mode)
  //     {
  //       json::value_type analysis_mode;
  //       analysis_mode["analysis_mode"] = (mode == AnalysisMode::kMax) ? "max_delay/setup" : "min_delay/hold";
  //       analysis_mode["levels_of_logic"] = 0;       // TBD
  //       analysis_mode["critical_path_length"] = 0;  // TBD
  //       analysis_mode["critical_path_slack"] = (mode == AnalysisMode::kMax) ? wns : hold_wns;
  //       analysis_mode["total_negative_slack"] = (mode == AnalysisMode::kMax) ? tns : hold_tns;
  //       path_group["timing_path_group"]["analysis_mode_infos"].push_back(analysis_mode);
  //     }
  //     summary_sta.push_back(path_group);
  //   }

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

}  // namespace ieda_feature
