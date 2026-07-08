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
 * @file RouterLegalizationTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-03-17
 * @brief Unit tests for Router pin legalization API.
 */

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "database/design/Inst.hh"
#include "database/design/Pin.hh"
#include "database/spatial/Point.hh"
#include "database/spatial/Rect.hh"
#include "database/spatial/Region.hh"
#include "module/routing/router/Router.hh"

namespace icts_test {
namespace {

using Point = icts::Point<int>;
using Rect = icts::Rect<int>;
using Region = icts::Region<int>;
using Router = icts::Router;

TEST(RouterLegalizationTest, LegalizePinsWritesBackPinAndInstLocations)
{
  auto movable_inst_0 = std::make_unique<icts::Inst>("buf0", "BUF", icts::InstType::kBuffer, Point(0, 0));
  auto movable_inst_1 = std::make_unique<icts::Inst>("buf1", "BUF", icts::InstType::kBuffer, Point(0, 0));
  auto fixed_inst = std::make_unique<icts::Inst>("sink0", "FF", icts::InstType::kFlipFlop, Point(1, 0));

  auto movable_pin_0 = std::make_unique<icts::Pin>("buf0/clk", icts::PinType::kClock, Point(0, 0), movable_inst_0.get());
  auto movable_pin_1 = std::make_unique<icts::Pin>("buf1/clk", icts::PinType::kClock, Point(0, 0), movable_inst_1.get());
  auto fixed_pin = std::make_unique<icts::Pin>("sink0/clk", icts::PinType::kClock, Point(1, 0), fixed_inst.get());

  std::vector<icts::Pin*> movable_pins{movable_pin_0.get(), movable_pin_1.get()};
  const std::vector<icts::Pin*> fixed_pins{fixed_pin.get()};

  auto result = Router::legalizePins(movable_pins, fixed_pins, Region(Rect(0, 0, 3, 1)), Region{});
  ASSERT_TRUE(result.success);
  ASSERT_EQ(result.legalized_points.size(), 2U);

  EXPECT_EQ(movable_pin_0->get_location(), movable_inst_0->get_location());
  EXPECT_EQ(movable_pin_1->get_location(), movable_inst_1->get_location());
  EXPECT_NE(movable_pin_0->get_location(), movable_pin_1->get_location());
  EXPECT_NE(movable_pin_0->get_location(), fixed_pin->get_location());
  EXPECT_NE(movable_pin_1->get_location(), fixed_pin->get_location());
}

}  // namespace
}  // namespace icts_test
