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
#include "feature_builder.h"
#include "idm.h"
#include "report_evaluator.h"
#include "route_builder.h"

namespace ieda_feature {

PlaceSummary FeatureBuilder::buildPLSummary(std::string step)
{
  PlaceSummary summary = iPLAPIInst.outputSummary(step);

  return summary;
}

RTSummary FeatureBuilder::buildRTSummary()
{
  RTSummary summary;

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
  

  return summary;
}

bool FeatureBuilder::buildRouteData(RouteAnalyseData* data)
{
  RouteDataBuilder route_builder(data);

  return route_builder.buildRouteData();
}

}  // namespace ieda_feature