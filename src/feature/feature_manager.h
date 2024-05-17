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
  bool has_feature() { return _summary == nullptr ? false : true; }
  FeatureSummary* get_summary() { return _summary; }

  bool save_reportSummary(std::string path, std::string step);
  bool save_reportSummary_map(std::string path, int bin_cnt_x, int bin_cnt_y);

 private:
  static FeatureManager* _instance;

  FeatureSummary* _summary = nullptr;

  FeatureManager();
  ~FeatureManager();
};

}  // namespace ieda_feature
