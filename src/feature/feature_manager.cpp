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
#include "feature_manager.h"

#include "feature_builder.h"
#include "feature_parser.h"

namespace ieda_feature {
FeatureManager* FeatureManager::_instance = nullptr;

FeatureManager::FeatureManager()
{
  _summary = new FeatureSummary();
}
FeatureManager::~FeatureManager()
{
  if (_summary != nullptr) {
    delete _summary;
    _summary = nullptr;
  }
}

bool FeatureManager::save_summary(std::string path)
{
  FeatureBuilder builder;
  auto db_summary = builder.buildDBSummary();

  _summary->set_db(db_summary);

  FeatureParser feature_parser(_summary);
  return feature_parser.buildSummary(path);
}

bool FeatureManager::save_eval_summary(std::string path, int32_t grid_size)
{
  FeatureBuilder builder;

  auto wirelength_db = builder.buildWirelengthEvalSummary();
  auto density_db = builder.buildDensityEvalSummary(grid_size);
  auto congestion_db = builder.buildCongestionEvalSummary(grid_size);
  auto timing_db = builder.buildTimingEvalSummary();

  _summary->set_wirelength_eval(wirelength_db);
  _summary->set_density_eval(density_db);
  _summary->set_congestion_eval(congestion_db);
  _summary->set_timing_eval(timing_db);

  FeatureParser feature_parser(_summary);
  return feature_parser.buildSummaryEval(path);
}

bool FeatureManager::save_eval_union(std::string jsonl_path, std::string csv_path, int32_t grid_size)
{
  FeatureBuilder builder;

  bool is_init_eval_tool = builder.initEvalTool();
  if (!is_init_eval_tool) {
    return false;
  }

  // auto union_db = builder.buildUnionEvalSummary(grid_size);
  // _summary->set_wirelength_eval(union_db.total_wl_summary);
  // _summary->set_density_eval(union_db.density_map_summary);
  // _summary->set_congestion_eval(union_db.congestion_summary);

  bool csv_success = builder.buildNetEval(csv_path);
  // bool csv_success = true;

  // FeatureParser feature_parser(_summary);
  // bool jsonl_success = feature_parser.buildSummaryEval(jsonl_path);
  bool jsonl_success = true;

  builder.destroyEvalTool();

  return jsonl_success && csv_success;
}

bool FeatureManager::save_pl_eval(std::string json_path, int32_t grid_size)
{
  // EGR
  FeatureBuilder builder;

  bool is_init_eval_tool = builder.initEvalTool();
  if (!is_init_eval_tool) {
    return false;
  }

  std::string stage = "place";

  auto union_db = builder.buildUnionEvalSummary(grid_size, stage);
  _summary->set_wirelength_eval(union_db.total_wl_summary);
  _summary->set_density_eval(union_db.density_map_summary);
  _summary->set_congestion_eval(union_db.congestion_summary);

  // builder.evalTiming("EGR", true);
  // builder.evalTiming("HPWL");
  // builder.evalTiming("FLUTE");
  // builder.evalTiming("SALT");

  // auto union_timing_db = builder.buildTimingUnionEvalSummary();
  // _summary->set_timing_eval(union_timing_db);

  FeatureParser feature_parser(_summary);
  bool json_success = feature_parser.buildSummaryEval(json_path);

  builder.destroyEvalTool();

  return json_success;
}

bool FeatureManager::save_cts_eval(std::string json_path, int32_t grid_size)
{
  // EGR
  FeatureBuilder builder;

  bool is_init_eval_tool = builder.initEvalTool();
  if (!is_init_eval_tool) {
    return false;
  }

  std::string stage = "cts";

  auto union_db = builder.buildUnionEvalSummary(grid_size, stage);
  _summary->set_wirelength_eval(union_db.total_wl_summary);
  _summary->set_density_eval(union_db.density_map_summary);
  _summary->set_congestion_eval(union_db.congestion_summary);

  // builder.evalTiming("EGR", true);

  // builder.evalTiming("HPWL");
  // builder.evalTiming("FLUTE");
  // builder.evalTiming("SALT");

  // auto union_timing_db = builder.buildTimingUnionEvalSummary();
  // _summary->set_timing_eval(union_timing_db);

  FeatureParser feature_parser(_summary);
  bool json_success = feature_parser.buildSummaryEval(json_path);

  builder.destroyEvalTool();

  return json_success;
}

bool FeatureManager::save_timing_eval_summary(std::string path)
{
  FeatureBuilder builder;
  auto eval_db = builder.buildTimingEvalSummary();

  _summary->set_timing_eval(eval_db);

  FeatureParser feature_parser(_summary);
  return feature_parser.buildSummaryTimingEval(path);
}

bool FeatureManager::save_tools(std::string path, std::string step)
{
  FeatureBuilder builder;
  if (step == "fixFanout") {
    auto db = builder.buildNetOptSummary();

    _summary->set_ino(db);
  } else if (step == "place" || step == "legalization" || (step == "filler")) {
    auto db = builder.buildPLSummary(step);

    _summary->set_ipl(db);
  } else if (step == "CTS") {
    auto db = builder.buildCTSSummary();

    _summary->set_icts(db);
  } else if (step == "optDrv") {
    auto db = builder.buildTimingOptSummary();

    _summary->set_ito_optdrv(db);
  } else if (step == "optHold") {
    auto db = builder.buildTimingOptSummary();

    _summary->set_ito_opthold(db);
  } else if (step == "optSetup") {
    auto db = builder.buildTimingOptSummary();

    _summary->set_ito_optsetup(db);
  } else if (step == "route") {
    // skip
  } else {
  }

  FeatureParser feature_parser(_summary);
  return feature_parser.buildTools(path, step);
}

bool FeatureManager::save_eval_map(std::string path, int bin_cnt_x, int bin_cnt_y)
{
  FeatureParser feature_parser(_summary);
  return feature_parser.buildSummaryMap(path, bin_cnt_x, bin_cnt_y);
}

bool FeatureManager::save_net_eval(std::string path)
{
  FeatureParser feature_parser;
  return feature_parser.buildNetEval(path);
}

bool FeatureManager::save_route_data(std::string path)
{
  FeatureBuilder builder;
  builder.buildRouteData(&_route_data);

  FeatureParser feature_parser(_summary);
  return feature_parser.buildRouteData(path, &_route_data);
}

bool FeatureManager::read_route_data(std::string path)
{
  FeatureParser feature_parser;
  return feature_parser.readRouteData(path, &_route_data);
}

bool FeatureManager::save_cong_map(std::string stage, std::string csv_dir)
{
  FeatureParser feature_parser;
  return feature_parser.buildCongMap(stage, csv_dir);
}
}  // namespace ieda_feature
