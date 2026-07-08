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
/**
 * @file RoutingTerminal.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-03-24
 * @brief Shared routing terminal value type.
 */

#pragma once

#include <string>

#include "Point.hh"

namespace icts {

struct RoutingTerminal
{
  std::string name = "";
  Point<int> location = Point<int>(-1, -1);
};

struct ClockRoutingTerminal : public RoutingTerminal
{
  double pin_cap = 0.0;
  double insertion_delay = 0.0;
};

}  // namespace icts
