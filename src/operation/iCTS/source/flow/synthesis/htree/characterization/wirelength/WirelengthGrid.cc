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
 * @file WirelengthGrid.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-01
 * @brief H-tree characterization wirelength grid resolution implementation.
 */

#include "synthesis/htree/characterization/wirelength/WirelengthGrid.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <ostream>
#include <ranges>
#include <vector>

#include "Log.hh"
#include "Tree.hh"
#include "ValueLattice.hh"
#include "config/Config.hh"
#include "geometry/Geometry.hh"

namespace icts::htree {
namespace {

auto MakeCoveringLengthIndex(double length_um, double length_step_um) -> unsigned
{
  return UniformValueLattice(length_step_um, std::numeric_limits<unsigned>::max()).coveringIndex(length_um);
}

auto MakeDenseLengthIndices(unsigned max_length_idx) -> std::vector<unsigned>
{
  std::vector<unsigned> length_indices;
  length_indices.reserve(max_length_idx);
  for (unsigned length_idx = 1U; length_idx <= max_length_idx; ++length_idx) {
    length_indices.push_back(length_idx);
  }
  return length_indices;
}

}  // namespace

auto ToCharGridSourceName(CharGridSource source) -> const char*
{
  switch (source) {
    case CharGridSource::kNone:
      return "none";
    case CharGridSource::kRuntimeConfig:
      return "runtime_config";
    case CharGridSource::kAutoDerived:
      return "auto_derived";
  }
  return "none";
}

auto CountUniqueAlignedLengthBins(const std::vector<double>& requested_lengths_um, double length_step_um) -> unsigned
{
  if (requested_lengths_um.empty() || length_step_um <= 0.0) {
    return 0U;
  }

  std::vector<unsigned> aligned_bins;
  aligned_bins.reserve(requested_lengths_um.size());
  for (const double requested_length_um : requested_lengths_um) {
    const unsigned aligned_idx = MakeCoveringLengthIndex(requested_length_um, length_step_um);
    if (aligned_idx > 0U) {
      aligned_bins.push_back(aligned_idx);
    }
  }

  if (aligned_bins.empty()) {
    return 0U;
  }

  std::ranges::sort(aligned_bins);
  const auto unique_tail = std::ranges::unique(aligned_bins);
  aligned_bins.erase(unique_tail.begin(), unique_tail.end());
  return static_cast<unsigned>(aligned_bins.size());
}

auto CollectRequestedLevelLengthsUm(const Tree& topology, int32_t dbu_per_um) -> std::vector<double>
{
  LOG_FATAL_IF(dbu_per_um <= 0) << "HTree: DBU-per-micron must be positive when collecting requested level lengths.";
  std::vector<double> requested_lengths_um;
  const auto levels = topology.levels();
  if (levels.size() <= 1U) {
    return requested_lengths_um;
  }

  requested_lengths_um.reserve(levels.size() - 1U);
  for (std::size_t level = 1; level < levels.size(); ++level) {
    long long distance_sum = 0;
    std::size_t distance_count = 0;
    for (const auto node_id : levels.at(level)) {
      const auto* node = topology.get_node(node_id);
      if (node == nullptr || node->get_parent() == std::numeric_limits<std::size_t>::max()) {
        continue;
      }

      const auto* parent = topology.get_node(node->get_parent());
      if (parent == nullptr) {
        continue;
      }

      distance_sum += geometry::Manhattan(node->get_position(), parent->get_position());
      ++distance_count;
    }

    if (distance_count == 0U) {
      continue;
    }

    const int requested_length_dbu
        = static_cast<int>(std::llround(static_cast<double>(distance_sum) / static_cast<double>(distance_count)));
    const double requested_length_um = static_cast<double>(std::max(requested_length_dbu, 0)) / static_cast<double>(dbu_per_um);
    if (requested_length_um > 0.0) {
      requested_lengths_um.push_back(requested_length_um);
    }
  }

  return requested_lengths_um;
}

auto ResolveCharacterizationGridPlan(const Config& config, const std::vector<double>& requested_lengths_um) -> CharacterizationGridPlan
{
  CharBuilder::Config char_config;
  if (config.get_wirelength_unit_um() > 0.0) {
    char_config.wirelength_unit_um = config.get_wirelength_unit_um();
  }
  char_config.wirelength_iterations = config.get_wirelength_iterations();
  return ResolveCharacterizationGridPlan(char_config, requested_lengths_um);
}

auto ResolveCharacterizationGridPlan(const CharBuilder::Config& config, const std::vector<double>& requested_lengths_um)
    -> CharacterizationGridPlan
{
  CharacterizationGridPlan plan;
  if (requested_lengths_um.empty()) {
    return plan;
  }
  plan.requested_level_lengths = static_cast<unsigned>(requested_lengths_um.size());

  const double configured_unit_um = config.wirelength_unit_um.value_or(0.0);
  plan.configured_wirelength_iterations = std::max(1U, config.wirelength_iterations.value_or(1U));
  plan.configured_wirelength_unit_um = configured_unit_um;
  plan.configured_wirelength_missing = configured_unit_um <= 0.0;

  double effective_unit_um = configured_unit_um;
  if (effective_unit_um > 0.0) {
    plan.unique_level_bins = CountUniqueAlignedLengthBins(requested_lengths_um, effective_unit_um);
    plan.source = CharGridSource::kRuntimeConfig;
  }

  const double max_requested_length_um = *std::ranges::max_element(requested_lengths_um);
  const double auto_derived_unit_um = max_requested_length_um / static_cast<double>(requested_lengths_um.size());
  const bool grid_collapsed = configured_unit_um > 0.0 && requested_lengths_um.size() > 1U && plan.unique_level_bins <= 1U;
  plan.configured_grid_collapsed = grid_collapsed;
  if (plan.configured_wirelength_missing || grid_collapsed) {
    effective_unit_um = auto_derived_unit_um;
    plan.adapted = effective_unit_um > 0.0;
    plan.source = plan.adapted ? CharGridSource::kAutoDerived : CharGridSource::kNone;
    plan.auto_derived_wirelength_unit_um = effective_unit_um;
    plan.unique_level_bins = CountUniqueAlignedLengthBins(requested_lengths_um, effective_unit_um);
  }

  if (!plan.adapted || effective_unit_um <= 0.0) {
    return plan;
  }

  plan.wirelength_unit_um = effective_unit_um;
  plan.required_covering_iterations = std::max(1U, static_cast<unsigned>(std::ceil(max_requested_length_um / effective_unit_um)));
  plan.wirelength_iterations = std::min(plan.configured_wirelength_iterations, plan.required_covering_iterations);
  return plan;
}

auto ResolveCharacterizationGridPlan(const Config& config, const Tree& topology, int32_t dbu_per_um) -> CharacterizationGridPlan
{
  return ResolveCharacterizationGridPlan(config, CollectRequestedLevelLengthsUm(topology, dbu_per_um));
}

auto ResolveDirectCharacterizationLengthIndices(const Tree& topology, const CharacterizationGridPlan& char_grid_plan, int32_t dbu_per_um)
    -> std::vector<unsigned>
{
  return ResolveDirectCharacterizationLengthIndices(CollectRequestedLevelLengthsUm(topology, dbu_per_um), char_grid_plan);
}

auto ResolveDirectCharacterizationLengthIndices(const std::vector<double>& requested_lengths_um,
                                                const CharacterizationGridPlan& char_grid_plan) -> std::vector<unsigned>
{
  if (!char_grid_plan.adapted || char_grid_plan.wirelength_iterations == 0U) {
    return {};
  }

  if (char_grid_plan.required_covering_iterations > char_grid_plan.wirelength_iterations) {
    return MakeDenseLengthIndices(char_grid_plan.wirelength_iterations);
  }

  std::vector<unsigned> required_length_indices;
  required_length_indices.reserve(requested_lengths_um.size());
  for (const double requested_length_um : requested_lengths_um) {
    const unsigned length_idx = MakeCoveringLengthIndex(requested_length_um, char_grid_plan.wirelength_unit_um);
    if (length_idx > 0U) {
      required_length_indices.push_back(length_idx);
    }
  }
  std::ranges::sort(required_length_indices);
  const auto unique_tail = std::ranges::unique(required_length_indices);
  required_length_indices.erase(unique_tail.begin(), unique_tail.end());
  std::erase_if(required_length_indices, [&](unsigned length_idx) -> bool { return length_idx > char_grid_plan.wirelength_iterations; });
  return required_length_indices;
}

}  // namespace icts::htree
