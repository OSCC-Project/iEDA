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
  int32_t grid_size = 2000;

  EvalSummary summary;
  // summary.total_wl_summary = WIRELENGTH_API_INST->totalWL();
  // summary.density_map_summary = DENSITY_API_INST->densityMap(grid_size);
  // summary.egr_map_summary = CONGESTION_API_INST->egrMap();
  // summary.overflow_summary = CONGESTION_API_INST->egrOverflow();
  // summary.rudy_map_summary = CONGESTION_API_INST->rudyMap(grid_size);
  // summary.rudy_utilization_summary = CONGESTION_API_INST->rudyUtilization(false);
  // summary.lutrudy_utilization_summary = CONGESTION_API_INST->rudyUtilization(true);

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