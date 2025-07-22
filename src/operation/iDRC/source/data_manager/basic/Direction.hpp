// ***************************************************************************************
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

namespace idrc {

enum class Direction
{
  kNone = 0,
  kHorizontal = 1,
  kVertical = 2,
  kOblique = 3,
  kProximal = 4
};

struct GetDirectionName
{
  std::string operator()(const Direction& direction) const
  {
    std::string direction_name;
    switch (direction) {
      case Direction::kNone:
        direction_name = "none";
        break;
      case Direction::kHorizontal:
        direction_name = "horizontal";
        break;
      case Direction::kVertical:
        direction_name = "vertical";
        break;
      case Direction::kOblique:
        direction_name = "oblique";
        break;
      case Direction::kProximal:
        direction_name = "proximal";
        break;
      default:
        DRCLOG.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return direction_name;
  }
};

}  // namespace idrc
