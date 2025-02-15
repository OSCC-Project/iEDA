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

#include "Logger.hpp"

namespace irt {

enum class VRObsType
{
  kNone = 0,
  kPlanar = 1,
  kSpace = 2
};

struct GetVRObsTypeName
{
  std::string operator()(const VRObsType& vr_obs_type) const
  {
    std::string vr_obs_type_name;
    switch (vr_obs_type) {
      case VRObsType::kNone:
        vr_obs_type_name = "none";
        break;
      case VRObsType::kPlanar:
        vr_obs_type_name = "planar";
        break;
      case VRObsType::kSpace:
        vr_obs_type_name = "space";
        break;
      default:
        RTLOG.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return vr_obs_type_name;
  }
};

}  // namespace irt
