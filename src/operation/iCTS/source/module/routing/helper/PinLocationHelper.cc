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
 * @file PinLocationHelper.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-10
 * @brief Shared pin-location collection helpers for routing consumers.
 */

#include "PinLocationHelper.hh"

#include <vector>

#include "Pin.hh"
#include "Point.hh"

namespace icts {

auto CollectPinLocations(const std::vector<Pin*>& pins) -> std::vector<Point<int>>
{
  std::vector<Point<int>> points;
  points.reserve(pins.size());
  for (const auto* pin : pins) {
    if (pin == nullptr) {
      continue;
    }
    points.push_back(pin->get_location());
  }
  return points;
}

}  // namespace icts
