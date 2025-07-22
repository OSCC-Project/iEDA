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
 * @project		iEDA
 * @file		feature_parser.h
 * @author		Yell
 * @date		10/08/2023
 * @version		0.1
 * @description


        feature parser
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <iostream>
#include <string>
#include <vector>

#include "json.hpp"
using json = nlohmann::json;

namespace idb {
class IdbDesign;
class IdbLayout;
}  // namespace idb

using namespace idb;

namespace ieda_feature {

using std::vector;

class FeatureSummary;
class RouteAnalyseData;

class FeatureParser
{
 public:
  FeatureParser();
  FeatureParser(FeatureSummary* summary);
  ~FeatureParser();

  // report
  bool buildSummary(std::string json_path);
  bool buildSummaryMap(std::string csv_path, int bin_cnt_x, int bin_cnt_y);
  bool buildTools(std::string json_path, std::string step);

  bool buildRouteData(std::string json_path, RouteAnalyseData* data);
  bool readRouteData(std::string json_path, RouteAnalyseData* data);

  bool buildSummaryEval(std::string json_path);
  bool buildSummaryTimingEval(std::string json_path);
  bool buildNetEval(std::string csv_path);
  bool buildCongMap(std::string stage, std::string csv_dir);

 private:
  IdbLayout* _layout = nullptr;
  IdbDesign* _design = nullptr;
  FeatureSummary* _summary = nullptr;

  json buildSummaryInfo();
  json buildSummaryLayout();
  json buildSummaryStatis();
  json buildSummaryInstances();
  json buildSummaryMacrosStatis();
  json buildSummaryMacros();
  json buildSummaryLayers();
  json buildSummaryNets();
  json buildSummaryPdn();
  json buildSummaryPins();

  json buildSummaryPL(std::string step);
  json buildSummaryCTS();
  json buildSummaryTO(std::string step = "");
  json buildSummaryNetOpt();
  json buildSummaryRT();
  json buildSummarySTA();
  json buildSummaryDRC();

  json buildSummaryWirelength();
  json buildSummaryDensity();
  json buildSummaryCongestion();
  json buildSummaryTiming();
  json buildSummaryPower();
};
}  // namespace ieda_feature
