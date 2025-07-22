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

enum class Rotation
{
  kNone,
  kClockwise,
  kCounterclockwise
};

struct GetRotationName
{
  std::string operator()(const Rotation& rotation) const
  {
    std::string rotation_name;
    switch (rotation) {
      case Rotation::kNone:
        rotation_name = "none";
        break;
      case Rotation::kClockwise:
        rotation_name = "clock_wise";
        break;
      case Rotation::kCounterclockwise:
        rotation_name = "counter_clock_wise";
        break;
      default:
        DRCLOG.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return rotation_name;
  }
};

}  // namespace idrc
