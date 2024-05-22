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

#include "feature_parser.h"

#include "feature_summary.h"
#include "flow_config.h"
#include "idm.h"
#include "json_parser.h"

namespace ieda_feature {
FeatureParser::FeatureParser(FeatureSummary* summary)
{
  _layout = dmInst->getInstance()->get_idb_layout();
  _design = dmInst->getInstance()->get_idb_design();
  _summary = summary;
}

FeatureParser::~FeatureParser()
{
  _layout = nullptr;
  _design = nullptr;
}

bool FeatureParser::buildSummary(std::string json_path)
{
  std::ofstream& file_stream = ieda::getOutputFileStream(json_path);
  json root;

  root["Design Information"] = buildSummaryInfo();

  root["Design Layout"] = buildSummaryLayout();

  root["Design Statis"] = buildSummaryStatis();

  root["Instances"] = buildSummaryInstances();

  root["Macros Statis"] = buildSummaryMacrosStatis();

  root["Macros"] = buildSummaryMacros();

  root["Nets"] = buildSummaryNets();

  //   root["PDN"] = buildSummaryPdn();

  root["Layers"] = buildSummaryLayers();

  root["Pins"] = buildSummaryPins();

  file_stream << std::setw(4) << root;

  ieda::closeFileStream(file_stream);

  std::cout << std::endl << "Save feature json success, path = " << json_path << std::endl;
  return true;
}

/**
 * if step = "", only save idb summary
 */
bool FeatureParser::buildTools(std::string json_path, std::string step)
{
  if (json_path.empty() || step.empty()) {
    return false;
  }

  using SummaryBuilder = std::function<json()>;
  auto stepToBuilder = std::unordered_map<std::string, SummaryBuilder>{{"place", [this, step]() { return buildSummaryPL(step); }},
                                                                       {"legalization", [this, step]() { return buildSummaryPL(step); }},
                                                                       {"CTS", [this]() { return buildSummaryCTS(); }},
                                                                       {"fixFanout", [this]() { return buildSummaryNetOpt(); }},
                                                                       {"optDrv", [this, step]() { return buildSummaryTO(step); }},
                                                                       {"optHold", [this, step]() { return buildSummaryTO(step); }},
                                                                       {"optSetup", [this, step]() { return buildSummaryTO(step); }},
                                                                       {"sta", [this]() { return buildSummarySTA(); }},
                                                                       {"drc", [this]() { return buildSummaryDRC(); }},
                                                                       {"route", [this]() { return buildSummaryRT(); }}};

  std::ofstream& file_stream = ieda::getOutputFileStream(json_path);
  json root;

  root[step] = stepToBuilder[step]();

  file_stream << std::setw(4) << root;

  ieda::closeFileStream(file_stream);

  std::cout << std::endl << "Save feature json success, path = " << json_path << std::endl;

  return true;
}

}  // namespace ieda_feature
