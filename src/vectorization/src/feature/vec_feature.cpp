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

#include <filesystem>

#include "Log.hh"
#include "idm.h"
#include "vec_feature.h"

namespace ivec {

void VecFeature::buildFeatureDrc(std::string drc_path)
{
  if (drc_path == "") {
    drc_path = _dir + "/" + dmInst->get_idb_design()->get_design_name() + "_route_baseline_drc.json";
  }

  namespace fs = std::filesystem;
  if (false == fs::exists(drc_path)) {
    LOG_WARNING << "Drc file not exist, path : " << drc_path;
    return;
  }

  VecFeatureDrc feature_drc(_layout, drc_path);

  feature_drc.build();
}

void VecFeature::buildFeatureTiming()
{
  VecFeatureTiming feature_timing(_layout, _dir, _is_placement_mode, _sta_mode);

  feature_timing.build();
}

void VecFeature::buildFeatureStatis()
{
  VecFeatureStatis feature_statis(_layout, _patch_grid, _is_placement_mode);

  feature_statis.build();
}
}  // namespace ivec