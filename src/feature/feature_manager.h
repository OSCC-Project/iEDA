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
 * @File Name: feature_manager.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2023-0811
 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <vector>

#include "../database/interaction/RT_DRC/ids.hpp"
#include "feature_irt.h"
#include "feature_summary.h"

#define featureInst ieda_feature::FeatureManager::getInstance()

namespace ieda_feature {

class FeatureManager
{
 public:
  static FeatureManager* getInstance()
  {
    if (!_instance) {
      _instance = new FeatureManager;
    }
    return _instance;
  }

  ///
  FeatureSummary* get_summary() { return _summary; }
  RouteAnalyseData& get_route_data() { return _route_data; }
  std::map<std::string, std::map<std::string, std::vector<ids::Violation>>>& get_type_layer_violation_map()
  {
    return _type_layer_violation_map;
  }

  bool save_summary(std::string path);
  bool save_tools(std::string path, std::string step);
  bool save_eval_map(std::string path, int bin_cnt_x, int bin_cnt_y);
  bool save_cong_map(std::string stage, std::string csv_dir);
  bool save_net_eval(std::string path);
  // route data
  bool save_route_data(std::string path);
  bool read_route_data(std::string path);
  // evaluation
  bool save_eval_summary(std::string path, int32_t grid_size);
  bool save_timing_eval_summary(std::string path);
  bool save_eval_union(std::string jsonl_path, std::string csv_path, int32_t grid_size);
  bool save_pl_eval(std::string json_path, int32_t grid_size = 1);
  bool save_cts_eval(std::string json_path, int32_t grid_size = 1);

 private:
  static FeatureManager* _instance;

  FeatureSummary* _summary = nullptr;
  RouteAnalyseData _route_data;
  std::map<std::string, std::map<std::string, std::vector<ids::Violation>>> _type_layer_violation_map;

  FeatureManager();
  ~FeatureManager();
};

}  // namespace ieda_feature
