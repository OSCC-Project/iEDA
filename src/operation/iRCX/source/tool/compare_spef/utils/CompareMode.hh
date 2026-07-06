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

#include "config/CompareSpefConfig.hh"

namespace ircx {
namespace compare_spef {
namespace compare_mode {

inline auto hasExplicitCompareMode(const Config& config) -> bool
{
  return config.compare_capacitance || config.compare_resistance || config.compare_delay;
}

inline auto compareCapacitance(const Config& config) -> bool
{
  return hasExplicitCompareMode(config) ? config.compare_capacitance : true;
}

inline auto compareResistance(const Config& config) -> bool
{
  return hasExplicitCompareMode(config) ? config.compare_resistance : true;
}

}  // namespace compare_mode
}  // namespace compare_spef
}  // namespace ircx
