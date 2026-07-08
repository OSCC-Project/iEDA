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
 * @file DistributionGenerators.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Statistical synthetic load generators for iCTS tests.
 */

#include "common/data/distribution/DistributionGenerators.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "Pin.hh"  // IWYU pragma: keep
#include "common/data/pin_factory/PinFactory.hh"
#include "database/spatial/Point.hh"

namespace icts_test::common::data::distribution {
namespace {

constexpr double kNormalMeanRatio = 0.5;
constexpr double kNormalSigmaRatio = 0.18;
constexpr std::array<double, 3> kGaussianMixtureWeights = {0.5, 0.3, 0.2};
constexpr double kFirstCenterXRatio = 0.25;
constexpr double kFirstCenterYRatio = 0.25;
constexpr double kSecondCenterXRatio = 0.75;
constexpr double kSecondCenterYRatio = 0.35;
constexpr double kThirdCenterXRatio = 0.55;
constexpr double kThirdCenterYRatio = 0.75;
constexpr double kMixtureSigmaRatio = 0.08;

auto ClampInt(int value, int lower, int upper) -> int
{
  return std::max(lower, std::min(value, upper));
}

template <typename Generator>
auto SampleAndClamp(Generator& generator, std::normal_distribution<double>& distribution, int lower, int upper) -> int
{
  const int sampled_value = static_cast<int>(std::lround(distribution(generator)));
  return ClampInt(sampled_value, lower, upper);
}

}  // namespace

auto MakeNormal(std::size_t count, CanvasSize canvas, unsigned seed) -> GeneratedPins
{
  if (count == 0 || canvas.width <= 0 || canvas.height <= 0) {
    return {};
  }

  std::mt19937 generator(seed);
  const double mean_x = static_cast<double>(canvas.width) * kNormalMeanRatio;
  const double mean_y = static_cast<double>(canvas.height) * kNormalMeanRatio;
  const double sigma_x = static_cast<double>(canvas.width) * kNormalSigmaRatio;
  const double sigma_y = static_cast<double>(canvas.height) * kNormalSigmaRatio;

  std::normal_distribution<double> dist_x(mean_x, sigma_x);
  std::normal_distribution<double> dist_y(mean_y, sigma_y);

  std::vector<icts::Point<int>> points;
  points.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const int x_coord = SampleAndClamp(generator, dist_x, 0, canvas.width);
    const int y_coord = SampleAndClamp(generator, dist_y, 0, canvas.height);
    points.emplace_back(x_coord, y_coord);
  }
  return pin_factory::BuildPinsFromPoints(points, canvas, "load_");
}

auto MakeGaussianMixture(std::size_t count, CanvasSize canvas, unsigned seed) -> GeneratedPins
{
  if (count == 0 || canvas.width <= 0 || canvas.height <= 0) {
    return {};
  }

  std::mt19937 generator(seed);
  std::discrete_distribution<int> pick(kGaussianMixtureWeights.begin(), kGaussianMixtureWeights.end());

  const std::array<icts::Point<double>, 3> centers = {
      icts::Point<double>(static_cast<double>(canvas.width) * kFirstCenterXRatio, static_cast<double>(canvas.height) * kFirstCenterYRatio),
      icts::Point<double>(static_cast<double>(canvas.width) * kSecondCenterXRatio,
                          static_cast<double>(canvas.height) * kSecondCenterYRatio),
      icts::Point<double>(static_cast<double>(canvas.width) * kThirdCenterXRatio, static_cast<double>(canvas.height) * kThirdCenterYRatio)};
  const double sigma_x = static_cast<double>(canvas.width) * kMixtureSigmaRatio;
  const double sigma_y = static_cast<double>(canvas.height) * kMixtureSigmaRatio;

  std::vector<icts::Point<int>> points;
  points.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const auto cluster_index = static_cast<std::size_t>(pick(generator));
    const auto& center = centers.at(cluster_index);
    std::normal_distribution<double> dist_x(center.get_x(), sigma_x);
    std::normal_distribution<double> dist_y(center.get_y(), sigma_y);
    const int x_coord = SampleAndClamp(generator, dist_x, 0, canvas.width);
    const int y_coord = SampleAndClamp(generator, dist_y, 0, canvas.height);
    points.emplace_back(x_coord, y_coord);
  }
  return pin_factory::BuildPinsFromPoints(points, canvas, "load_");
}

auto MakeWeightedQuadrants(std::size_t count, CanvasSize canvas, unsigned seed, const std::array<double, 4>& weights) -> GeneratedPins
{
  if (count == 0 || canvas.width <= 0 || canvas.height <= 0) {
    return {};
  }

  std::mt19937 generator(seed);
  std::array<double, 4> safe_weights = weights;
  const double sum = safe_weights.at(0) + safe_weights.at(1) + safe_weights.at(2) + safe_weights.at(3);
  if (sum <= 0.0) {
    safe_weights = {1.0, 1.0, 1.0, 1.0};
  }
  std::discrete_distribution<int> pick(safe_weights.begin(), safe_weights.end());

  const int mid_x = canvas.width / 2;
  const int mid_y = canvas.height / 2;

  std::vector<icts::Point<int>> points;
  points.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const int quadrant = pick(generator);
    int x_low = 0;
    int x_high = mid_x;
    int y_low = 0;
    int y_high = mid_y;
    switch (quadrant) {
      case 1:
        x_low = mid_x;
        x_high = canvas.width;
        y_low = 0;
        y_high = mid_y;
        break;
      case 2:
        x_low = 0;
        x_high = mid_x;
        y_low = mid_y;
        y_high = canvas.height;
        break;
      case 3:
        x_low = mid_x;
        x_high = canvas.width;
        y_low = mid_y;
        y_high = canvas.height;
        break;
      default:
        break;
    }
    std::uniform_int_distribution<int> dist_x(x_low, x_high);
    std::uniform_int_distribution<int> dist_y(y_low, y_high);
    points.emplace_back(dist_x(generator), dist_y(generator));
  }
  return pin_factory::BuildPinsFromPoints(points, canvas, "load_");
}

}  // namespace icts_test::common::data::distribution
