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
 * @file TopologyGenCaseFactory.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Case construction and data generation for topology generation tests.
 */

#include <array>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

#include "common/data/TestDataGenerator.hh"
#include "common/dataset/TestDataset.hh"
#include "module/topology/topology_gen/fixture/TopologyGenCaseFixture.hh"
#include "module/topology/topology_gen/fixture/TopologyGenScenario.hh"

namespace icts_test::topology_gen {
namespace {

static_assert(std::is_same_v<decltype(TopologyCase::name), std::string>);

constexpr int kCanvasWidth = 10000;
constexpr int kCanvasHeight = 8000;
constexpr unsigned kSeedNormalSmall = 12345;
constexpr unsigned kSeedNormalLarge = 54321;
constexpr unsigned kSeedMixture = 2026;
constexpr unsigned kSeedQuadrantOne = 1001;
constexpr unsigned kSeedQuadrantUneven = 1002;
constexpr std::size_t kNormalSmallCount = 256;
constexpr std::size_t kNormalLargeCount = 1024;
constexpr std::size_t kMixtureCount = 512;
constexpr std::size_t kQuadrantOneCount = 256;
constexpr std::size_t kQuadrantUnevenCount = 768;
constexpr std::array<double, 4> kQuadrantOneWeights = {1.0, 0.0, 0.0, 0.0};
constexpr std::array<double, 4> kQuadrantUnevenWeights = {0.5, 0.2, 0.3, 0.0};

}  // namespace

namespace detail {

auto GenerateCase(const TopologyCase& test_case) -> GeneratedPins
{
  const CanvasSize canvas{.width = test_case.width, .height = test_case.height};
  switch (test_case.kind) {
    case DistKind::kNormal:
      return common::data::MakeNormal(test_case.count, canvas, test_case.seed);
    case DistKind::kMixture:
      return common::data::MakeGaussianMixture(test_case.count, canvas, test_case.seed);
    case DistKind::kQuadrants:
    default:
      return common::data::MakeWeightedQuadrants(test_case.count, canvas, test_case.seed, test_case.quadrant_weights);
  }
}

}  // namespace detail

auto BuildTopologyCases() -> std::vector<TopologyCase>
{
  return {
      TopologyCase{.name = "normal_small",
                   .kind = DistKind::kNormal,
                   .count = kNormalSmallCount,
                   .width = kCanvasWidth,
                   .height = kCanvasHeight,
                   .seed = kSeedNormalSmall},
      TopologyCase{.name = "normal_large",
                   .kind = DistKind::kNormal,
                   .count = kNormalLargeCount,
                   .width = kCanvasWidth,
                   .height = kCanvasHeight,
                   .seed = kSeedNormalLarge},
      TopologyCase{.name = "mixture",
                   .kind = DistKind::kMixture,
                   .count = kMixtureCount,
                   .width = kCanvasWidth,
                   .height = kCanvasHeight,
                   .seed = kSeedMixture},
      TopologyCase{.name = "quadrant_one",
                   .kind = DistKind::kQuadrants,
                   .count = kQuadrantOneCount,
                   .width = kCanvasWidth,
                   .height = kCanvasHeight,
                   .seed = kSeedQuadrantOne,
                   .quadrant_weights = kQuadrantOneWeights},
      TopologyCase{.name = "quadrant_three_uneven",
                   .kind = DistKind::kQuadrants,
                   .count = kQuadrantUnevenCount,
                   .width = kCanvasWidth,
                   .height = kCanvasHeight,
                   .seed = kSeedQuadrantUneven,
                   .quadrant_weights = kQuadrantUnevenWeights},
  };
}

}  // namespace icts_test::topology_gen
