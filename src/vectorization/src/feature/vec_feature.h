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

#include "vec_feature_drc.h"
#include "vec_feature_statis.h"
#include "vec_feature_timing.h"
#include "vec_layout.h"

namespace ivec {

class VecFeature
{
 public:
  VecFeature(VecLayout* layout, VecPatchGrid* patch_grid, std::string dir, bool is_placement_mode = false)
  {
    _layout = layout;
    _patch_grid = patch_grid;
    _dir = dir;  /// feature directory
    _is_placement_mode = is_placement_mode;
  }
  ~VecFeature() {}

  void buildFeatureDrc(std::string drc_path = "");
  void buildFeatureTiming();
  void buildFeatureStatis();

 private:
  VecLayout* _layout;
  VecPatchGrid* _patch_grid;
  std::string _dir;
  bool _is_placement_mode;
};

}  // namespace ivec