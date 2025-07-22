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

namespace idrc {

enum class Orientation
{
  kNone = 0,
  kEast = 1,
  kWest = 2,
  kSouth = 3,
  kNorth = 4,
  kAbove = 5,
  kBelow = 6,
  kOblique = 7
};

struct GetOrientationName
{
  std::string operator()(const Orientation& orientation) const
  {
    std::string orientation_name;
    switch (orientation) {
      case Orientation::kNone:
        orientation_name = "none";
        break;
      case Orientation::kEast:
        orientation_name = "east";
        break;
      case Orientation::kWest:
        orientation_name = "west";
        break;
      case Orientation::kSouth:
        orientation_name = "south";
        break;
      case Orientation::kNorth:
        orientation_name = "north";
        break;
      case Orientation::kAbove:
        orientation_name = "above";
        break;
      case Orientation::kBelow:
        orientation_name = "below";
        break;
      default:
        DRCLOG.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return orientation_name;
  }
};

}  // namespace idrc
