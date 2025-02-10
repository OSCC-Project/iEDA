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

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "feature_db.h"
#include "feature_icts.h"
#include "feature_ieval.h"
#include "feature_ino.h"
#include "feature_ipl.h"
#include "feature_irt.h"
#include "feature_ito.h"

namespace ieda_feature {

class FeatureBuilder
{
 public:
  FeatureBuilder();
  ~FeatureBuilder();

  // builder
  DBSummary buildDBSummary();
  PlaceSummary buildPLSummary(std::string step);
  RTSummary buildRTSummary();
  CTSSummary buildCTSSummary();
  NetOptSummary buildNetOptSummary();
  TimingOptSummary buildTimingOptSummary();

  TotalWLSummary buildWirelengthEvalSummary();
  DensityMapSummary buildDensityEvalSummary(int32_t grid_size);
  CongestionSummary buildCongestionEvalSummary(int32_t grid_size);
  TimingEvalSummary buildTimingEvalSummary();
  TimingEvalSummary buildTimingUnionEvalSummary();
  void evalTiming(const std::string& routing_type, const bool& rt_done = false);

  bool initEvalTool();
  UnionEvalSummary buildUnionEvalSummary(int32_t grid_size, std::string stage);
  bool buildNetEval(std::string csv_path);
  bool destroyEvalTool();

  bool buildRouteData(RouteAnalyseData* data);

 private:
  SummaryInfo buildSummaryInfo();
  SummaryLayout buildSummaryLayout();
  SummaryStatis buildSummaryStatis();
  SummaryInstances buildSummaryInstances();
  SummaryNets buildSummaryNets();
  SummaryLayers buildSummaryLayers();
  SummaryPins buildSummaryPins();
};

}  // namespace ieda_feature
