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
FeatureParser::FeatureParser()
{
  _layout = dmInst->getInstance()->get_idb_layout();
  _design = dmInst->getInstance()->get_idb_design();
}

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
                                                                       {"filler", [this, step]() { return buildSummaryPL(step); }},
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

bool FeatureParser::buildRouteData(std::string json_path, RouteAnalyseData* data)
{
  std::ofstream& file_stream = ieda::getOutputFileStream(json_path);
  json root;

  for (auto [cellmaster_name, cell_master] : data->cell_master_list) {
    json json_cellmaster;
    for (auto [term_name, term_pa] : cell_master.term_list) {
      json json_term;

      for (size_t i = 0; i < term_pa.pa_list.size(); i++) {
        json json_pa;
        json_pa["layer"] = term_pa.pa_list[i].layer;
        json_pa["x"] = term_pa.pa_list[i].x;
        json_pa["y"] = term_pa.pa_list[i].y;
        json_pa["number"] = term_pa.pa_list[i].number;

        json_term[i] = json_pa;
      }

      json_cellmaster[term_name] = json_term;
    }

    root[cellmaster_name] = json_cellmaster;
  }

  /// build route data json
  file_stream << std::setw(4) << root;
  ieda::closeFileStream(file_stream);

  std::cout << std::endl << "Save feature json success, path = " << json_path << std::endl;
  return true;
}

bool FeatureParser::readRouteData(std::string json_path, RouteAnalyseData* data)
{
  auto json_file = std::ifstream(json_path);
  if (false == json_file.is_open()) {
    return false;
  }

  json root;
  json_file >> root;
  for (json::iterator item_cell = root.begin(); item_cell != root.end(); ++item_cell) {
    CellMasterPA master_pa;

    std::string cell_name = item_cell.key();
    auto json_cell = item_cell.value();

    for (json::iterator item_term = json_cell.begin(); item_term != json_cell.end(); ++item_term) {
      TermPA term_pa;

      std::string term_name = item_term.key();
      auto json_term = item_term.value();

      for (json::iterator item_pa = json_term.begin(); item_pa != json_term.end(); ++item_pa) {
        auto json_pa = item_pa.value();

        DbPinAccess pa;
        pa.layer = json_pa["layer"];
        pa.number = json_pa["number"];
        pa.x = json_pa["x"];
        pa.y = json_pa["y"];

        term_pa.pa_list.push_back(pa);
      }

      master_pa.term_list.insert(std::make_pair(term_name, term_pa));
    }

    master_pa.name = cell_name;

    data->cell_master_list.insert(std::make_pair(cell_name, master_pa));
  }

  return true;
}

bool FeatureParser::buildSummaryEval(std::string json_path)
{
  std::ofstream& file_stream = ieda::getOutputFileStream(json_path);
  json root;

  root["Wirelength"] = buildSummaryWirelength();

  root["Density"] = buildSummaryDensity();

  root["Congestion"] = buildSummaryCongestion();

  root["Timing"] = buildSummaryTiming();

  root["Power"] = buildSummaryPower();

  file_stream << std::setw(4) << root;

  ieda::closeFileStream(file_stream);

  std::cout << std::endl << "Save eval json success, path = " << json_path << std::endl;
  return true;
}

bool FeatureParser::buildSummaryEvalJsonl(std::string jsonl_path)
{
  std::ofstream& file_stream = ieda::getOutputFileStream(jsonl_path);

  json wirelength;
  wirelength["Wirelength"] = buildSummaryWirelength();
  file_stream << wirelength << std::endl;

  json density;
  density["Density"] = buildSummaryDensity();
  file_stream << density << std::endl;

  json congestion;
  congestion["Congestion"] = buildSummaryCongestion();
  file_stream << congestion << std::endl;

  // json timing;
  // timing["Timing"] = buildSummaryTiming();
  // file_stream << timing << std::endl;

  // json power;
  // power["Power"] = buildSummaryPower();
  // file_stream << power << std::endl;

  ieda::closeFileStream(file_stream);

  std::cout << std::endl << "Save eval jsonl success, path = " << jsonl_path << std::endl;
  return true;
}

bool FeatureParser::buildSummaryTimingEval(std::string json_path)
{
  std::ofstream& file_stream = ieda::getOutputFileStream(json_path);
  json root;

  root["clocks_timing"] = buildSummaryTiming();

  root["power_info"] = buildSummaryPower();

  file_stream << std::setw(4) << root;

  ieda::closeFileStream(file_stream);

  std::cout << std::endl << "Save eval json success, path = " << json_path << std::endl;
  return true;
}

}  // namespace ieda_feature
