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
 * @file		feature_builder.h
 * @date		13/05/2024
 * @version		0.1
 * @description


        build feature data
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "CTSAPI.hh"
#include "NoApi.hpp"
#include "PLAPI.hh"
#include "RTInterface.hpp"
#include "ToApi.hpp"
#include "congestion_api.h"
#include "density_api.h"
#include "feature_builder.h"
#include "idm.h"
#include "report_evaluator.h"
#include "route_builder.h"
#include "wirelength_api.h"

namespace ieda_feature {

EvalSummary FeatureBuilder::buildEvalSummary()
{
  EvalSummary summary;

  // Wirelength
  ieval::TotalWLSummary eval_total_wl_summary = WIRELENGTH_API_INST->totalWL();
  summary.total_wl_summary.HPWL = eval_total_wl_summary.HPWL;
  summary.total_wl_summary.FLUTE = eval_total_wl_summary.FLUTE;
  summary.total_wl_summary.HTree = eval_total_wl_summary.HTree;
  summary.total_wl_summary.VTree = eval_total_wl_summary.VTree;
  summary.total_wl_summary.GRWL = eval_total_wl_summary.GRWL;

  // Density
  int32_t grid_size = 2000;
  ieval::DensityMapSummary eval_density_map_summary = DENSITY_API_INST->densityMap(grid_size);
  summary.density_map_summary.cell_map_summary.macro_density = eval_density_map_summary.cell_map_summary.macro_density;
  summary.density_map_summary.cell_map_summary.stdcell_density = eval_density_map_summary.cell_map_summary.stdcell_density;
  summary.density_map_summary.cell_map_summary.allcell_density = eval_density_map_summary.cell_map_summary.allcell_density;
  summary.density_map_summary.pin_map_summary.macro_pin_density = eval_density_map_summary.pin_map_summary.macro_pin_density;
  summary.density_map_summary.pin_map_summary.stdcell_pin_density = eval_density_map_summary.pin_map_summary.stdcell_pin_density;
  summary.density_map_summary.pin_map_summary.allcell_pin_density = eval_density_map_summary.pin_map_summary.allcell_pin_density;
  summary.density_map_summary.net_map_summary.local_net_density = eval_density_map_summary.net_map_summary.local_net_density;
  summary.density_map_summary.net_map_summary.global_net_density = eval_density_map_summary.net_map_summary.global_net_density;
  summary.density_map_summary.net_map_summary.allnet_density = eval_density_map_summary.net_map_summary.allnet_density;

  // Congestion
  ieval::EGRMapSummary eval_egr_map_summary = CONGESTION_API_INST->egrMap();
  summary.egr_map_summary.horizontal_sum = eval_egr_map_summary.horizontal_sum;
  summary.egr_map_summary.vertical_sum = eval_egr_map_summary.vertical_sum;
  summary.egr_map_summary.union_sum = eval_egr_map_summary.union_sum;

  ieval::RUDYMapSummary eval_rudy_map_summary = CONGESTION_API_INST->rudyMap(grid_size);
  summary.rudy_map_summary.rudy_horizontal = eval_rudy_map_summary.rudy_horizontal;
  summary.rudy_map_summary.rudy_vertical = eval_rudy_map_summary.rudy_vertical;
  summary.rudy_map_summary.rudy_union = eval_rudy_map_summary.rudy_union;
  summary.rudy_map_summary.lutrudy_horizontal = eval_rudy_map_summary.lutrudy_horizontal;
  summary.rudy_map_summary.lutrudy_vertical = eval_rudy_map_summary.lutrudy_vertical;
  summary.rudy_map_summary.lutrudy_union = eval_rudy_map_summary.lutrudy_union;

  ieval::OverflowSummary eval_overflow_summary = CONGESTION_API_INST->egrOverflow();
  summary.overflow_summary.total_overflow_horizontal = eval_overflow_summary.total_overflow_horizontal;
  summary.overflow_summary.total_overflow_vertical = eval_overflow_summary.total_overflow_vertical;
  summary.overflow_summary.total_overflow_union = eval_overflow_summary.total_overflow_union;
  summary.overflow_summary.max_overflow_horizontal = eval_overflow_summary.max_overflow_horizontal;
  summary.overflow_summary.max_overflow_vertical = eval_overflow_summary.max_overflow_vertical;
  summary.overflow_summary.max_overflow_union = eval_overflow_summary.max_overflow_union;
  summary.overflow_summary.weighted_average_overflow_horizontal = eval_overflow_summary.weighted_average_overflow_horizontal;
  summary.overflow_summary.weighted_average_overflow_vertical = eval_overflow_summary.weighted_average_overflow_vertical;
  summary.overflow_summary.weighted_average_overflow_union = eval_overflow_summary.weighted_average_overflow_union;

  ieval::UtilizationSummary eval_utilization_summary = CONGESTION_API_INST->rudyUtilization(false);
  summary.rudy_utilization_summary.max_utilization_horizontal = eval_utilization_summary.max_utilization_horizontal;
  summary.rudy_utilization_summary.max_utilization_vertical = eval_utilization_summary.max_utilization_vertical;
  summary.rudy_utilization_summary.max_utilization_union = eval_utilization_summary.max_utilization_union;
  summary.rudy_utilization_summary.weighted_average_utilization_horizontal
      = eval_utilization_summary.weighted_average_utilization_horizontal;
  summary.rudy_utilization_summary.weighted_average_utilization_vertical = eval_utilization_summary.weighted_average_utilization_vertical;
  summary.rudy_utilization_summary.weighted_average_utilization_union = eval_utilization_summary.weighted_average_utilization_union;

  ieval::UtilizationSummary eval_lut_utilization_summary = CONGESTION_API_INST->rudyUtilization(true);
  summary.lutrudy_utilization_summary.max_utilization_horizontal = eval_lut_utilization_summary.max_utilization_horizontal;
  summary.lutrudy_utilization_summary.max_utilization_vertical = eval_lut_utilization_summary.max_utilization_vertical;
  summary.lutrudy_utilization_summary.max_utilization_union = eval_lut_utilization_summary.max_utilization_union;
  summary.lutrudy_utilization_summary.weighted_average_utilization_horizontal
      = eval_lut_utilization_summary.weighted_average_utilization_horizontal;
  summary.lutrudy_utilization_summary.weighted_average_utilization_vertical
      = eval_lut_utilization_summary.weighted_average_utilization_vertical;
  summary.lutrudy_utilization_summary.weighted_average_utilization_union = eval_lut_utilization_summary.weighted_average_utilization_union;

  return summary;
}

PlaceSummary FeatureBuilder::buildPLSummary(std::string step)
{
  PlaceSummary summary = iPLAPIInst.outputSummary(step);

  return summary;
}

RTSummary FeatureBuilder::buildRTSummary()
{
  RTSummary summary = RTI.outputSummary();

  return summary;
}

CTSSummary FeatureBuilder::buildCTSSummary()
{
  CTSSummary summary = CTSAPIInst.outputSummary();

  return summary;
}

NetOptSummary FeatureBuilder::buildNetOptSummary()
{
  NetOptSummary summary = NoApiInst.outputSummary();

  return summary;
}

TimingOptSummary FeatureBuilder::buildTimingOptSummary()
{
  TimingOptSummary summary = ToApiInst.outputSummary();

  // HPWL, STWL, Global_routing_WL, congestion
  auto& nets = dmInst->get_idb_design()->get_net_list()->get_net_list();
  auto wl_nets = iplf::EvalWrapper::parallelWrap<eval::WLNet>(nets, iplf::EvalWrapper::wrapWLNet);
  summary.HPWL = EvalInst.evalTotalWL("kHPWL", wl_nets);
  summary.STWL = EvalInst.evalTotalWL("kFlute", wl_nets);

  return summary;
}

bool FeatureBuilder::buildRouteData(RouteAnalyseData* data)
{
  RouteDataBuilder route_builder(data);

  return route_builder.buildRouteData();
}

}  // namespace ieda_feature