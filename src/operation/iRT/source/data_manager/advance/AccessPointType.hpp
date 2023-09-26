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

#include "Logger.hpp"

namespace irt {

enum class AccessPointType
{
  kNone = 0,
  kPrefTrackGrid = 1,
  kCurrTrackGrid = 2,
  kOnTrack = 3,
  kOnShape = 4
};

struct GetAccessPointTypeName
{
  std::string operator()(const AccessPointType& access_point_type) const
  {
    std::string access_point_name;
    switch (access_point_type) {
      case AccessPointType::kNone:
        access_point_name = "none";
        break;
      case AccessPointType::kPrefTrackGrid:
        access_point_name = "prefer_track_grid";
        break;
      case AccessPointType::kCurrTrackGrid:
        access_point_name = "curr_track_grid";
        break;
      case AccessPointType::kOnTrack:
        access_point_name = "on_track";
        break;
      case AccessPointType::kOnShape:
        access_point_name = "on_shape";
        break;
      default:
        LOG_INST.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return access_point_name;
  }
};

}  // namespace irt
