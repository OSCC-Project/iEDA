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

enum class DRRouteStrategy
{
  kNone,
  kFullyConsider,
  kIgnoringBlockAndPin
};

struct GetDRRouteStrategyName
{
  std::string operator()(const DRRouteStrategy& dr_route_strategy) const
  {
    std::string dr_route_strategy_name;
    switch (dr_route_strategy) {
      case DRRouteStrategy::kNone:
        dr_route_strategy_name = "none";
        break;
      case DRRouteStrategy::kFullyConsider:
        dr_route_strategy_name = "fully_consider";
        break;
      case DRRouteStrategy::kIgnoringBlockAndPin:
        dr_route_strategy_name = "ignoring_block_and_pin";
        break;
      default:
        LOG_INST.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return dr_route_strategy_name;
  }
};

}  // namespace irt
