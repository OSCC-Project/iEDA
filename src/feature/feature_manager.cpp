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
};

bool FeatureManager::save_layout(std::string path)
{
  FeatureParser feature_parser(_summary);
  return feature_parser.buildLayout(path);
}

bool FeatureManager::save_instances(std::string path)
{
  FeatureParser feature_parser(_summary);
  return feature_parser.buildInstances(path);
}

bool FeatureManager::save_nets(std::string path)
{
  FeatureParser feature_parser(_summary);
  return feature_parser.buildNets(path);
}

bool FeatureManager::save_reportSummary(std::string path, std::string step)
{
  FeatureParser feature_parser(_summary);
  return feature_parser.buildReportSummary(path, step);
}

bool FeatureManager::save_reportSummary_map(std::string path, int bin_cnt_x, int bin_cnt_y)
{
  FeatureParser feature_parser(_summary);
  return feature_parser.buildReportSummaryMap(path, bin_cnt_x, bin_cnt_y);
}

}  // namespace ieda_feature
