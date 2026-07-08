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
 * @file PinFactory.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Pin materialization helpers for synthetic iCTS test datasets.
 */

#include "common/data/pin_factory/PinFactory.hh"

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "common/dataset/TestDataset.hh"
#include "database/design/Pin.hh"
#include "database/spatial/Point.hh"

namespace icts_test::common::data::pin_factory {

auto BuildPinsFromPoints(const std::vector<icts::Point<int>>& points, CanvasSize canvas, const std::string& pin_prefix) -> GeneratedPins
{
  GeneratedPins result;
  result.width = canvas.width;
  result.height = canvas.height;
  result.storage.reserve(points.size());
  result.loads.reserve(points.size());
  for (std::size_t index = 0; index < points.size(); ++index) {
    std::ostringstream name;
    name << pin_prefix << index;
    auto pin = std::make_unique<icts::Pin>(name.str(), icts::PinType::kClock, points.at(index));
    result.loads.push_back(pin.get());
    result.storage.push_back(std::move(pin));
  }
  return result;
}

}  // namespace icts_test::common::data::pin_factory
