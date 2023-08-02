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

enum class RoutingState
{
  kNone = 0,
  kRouted = 1,
  kUnrouted = 2
};

struct GetRoutingStateName
{
  std::string operator()(const RoutingState& connect_type) const
  {
    std::string connect_name;
    switch (connect_type) {
      case RoutingState::kNone:
        connect_name = "none";
        break;
      case RoutingState::kRouted:
        connect_name = "routed";
        break;
      case RoutingState::kUnrouted:
        connect_name = "unrouted";
        break;
      default:
        LOG_INST.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return connect_name;
  }
};

}  // namespace irt
