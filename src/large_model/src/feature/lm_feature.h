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
#include <string>

#include "lm_feature_drc.h"
#include "lm_feature_statis.h"
#include "lm_feature_timing.h"
#include "lm_layout.h"

namespace ilm {

class LmFeature
{
 public:
  LmFeature(LmLayout* layout, std::string dir)
  {
    _layout = layout;
    _dir = dir;  /// feature directory
  }
  ~LmFeature() {}

  void buildFeatureDrc(std::string drc_path = "");
  void buildFeatureTiming();
  void buildFeatureStatis();

 private:
  LmLayout* _layout;
  std::string _dir;
};

}  // namespace ilm