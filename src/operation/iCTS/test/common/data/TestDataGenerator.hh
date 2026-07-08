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
 * @file TestDataGenerator.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Random pin data generation for synthetic iCTS tests.
 */

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "common/dataset/TestDataset.hh"

namespace icts {
template <typename T>
class Point;
}  // namespace icts

namespace icts_test::common::data {

auto MakeNormal(std::size_t count, CanvasSize canvas, unsigned seed) -> GeneratedPins;
auto MakeGaussianMixture(std::size_t count, CanvasSize canvas, unsigned seed) -> GeneratedPins;
auto MakeWeightedQuadrants(std::size_t count, CanvasSize canvas, unsigned seed, const std::array<double, 4>& weights) -> GeneratedPins;
auto BuildPinsFromPoints(const std::vector<icts::Point<int>>& points, CanvasSize canvas, const std::string& pin_prefix) -> GeneratedPins;

}  // namespace icts_test::common::data
