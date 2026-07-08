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
 * @file TestDataGenerator.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Random pin data generation for synthetic iCTS tests.
 */

#include "common/data/TestDataGenerator.hh"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "common/data/distribution/DistributionGenerators.hh"
#include "common/data/pin_factory/PinFactory.hh"
#include "common/dataset/TestDataset.hh"

namespace icts_test::common::data {

auto MakeNormal(std::size_t count, CanvasSize canvas, unsigned seed) -> GeneratedPins
{
  return distribution::MakeNormal(count, canvas, seed);
}

auto MakeGaussianMixture(std::size_t count, CanvasSize canvas, unsigned seed) -> GeneratedPins
{
  return distribution::MakeGaussianMixture(count, canvas, seed);
}

auto MakeWeightedQuadrants(std::size_t count, CanvasSize canvas, unsigned seed, const std::array<double, 4>& weights) -> GeneratedPins
{
  return distribution::MakeWeightedQuadrants(count, canvas, seed, weights);
}

auto BuildPinsFromPoints(const std::vector<icts::Point<int>>& points, CanvasSize canvas, const std::string& pin_prefix) -> GeneratedPins
{
  return pin_factory::BuildPinsFromPoints(points, canvas, pin_prefix);
}

}  // namespace icts_test::common::data
