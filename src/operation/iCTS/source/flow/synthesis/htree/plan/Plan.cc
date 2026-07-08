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
 * @file LevelPlan.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-01
 * @brief H-tree topology level length planning.
 */

#include "synthesis/htree/plan/Plan.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "Log.hh"
#include "PatternId.hh"
#include "Tree.hh"
#include "ValueLattice.hh"
#include "geometry/Geometry.hh"
#include "synthesis/htree/HTree.hh"

namespace icts::htree {
namespace {

auto MakeCoveringLengthIndex(double length_um, double length_step_um) -> unsigned
{
  return UniformValueLattice(length_step_um, std::numeric_limits<unsigned>::max()).coveringIndex(length_um);
}

}  // namespace

auto BuildLevelPlans(const Tree& topology, double length_step_um, int32_t dbu_per_um) -> std::vector<HTree::LevelPlan>
{
  LOG_FATAL_IF(dbu_per_um <= 0) << "HTree: DBU-per-micron must be positive when building level plans.";
  std::vector<HTree::LevelPlan> plans;
  const auto levels = topology.levels();
  if (levels.size() <= 1U || length_step_um <= 0.0) {
    return plans;
  }

  plans.reserve(levels.size() - 1U);
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
      plans.push_back({});
      continue;
    }

    const int requested_length_dbu
        = static_cast<int>(std::llround(static_cast<double>(distance_sum) / static_cast<double>(distance_count)));
    const double requested_length_um = static_cast<double>(std::max(requested_length_dbu, 0)) / static_cast<double>(dbu_per_um);
    const unsigned aligned_length_idx = MakeCoveringLengthIndex(requested_length_um, length_step_um);
    plans.push_back(HTree::LevelPlan{
        .requested_length_dbu = requested_length_dbu,
        .requested_length_um = requested_length_um,
        .aligned_length_idx = aligned_length_idx,
        .aligned_length_um = static_cast<double>(aligned_length_idx) * length_step_um,
        .is_leaf_level = (level + 1U == levels.size()),
        .selected_has_any_buffer = false,
        .selected_has_terminal_branch_buffer = false,
        .selected_leaf_buffer_cell_master = {},
        .selected_terminal_cell_master = {},
        .segment_pattern_id = PatternId::segment(0),
    });
  }

  return plans;
}

auto MakeCandidateLevelPlans(const std::vector<HTree::LevelPlan>& full_level_plans, unsigned depth) -> std::vector<HTree::LevelPlan>
{
  std::vector<HTree::LevelPlan> candidate_levels;
  if (depth == 0U || full_level_plans.empty()) {
    return candidate_levels;
  }

  const std::size_t level_count = std::min<std::size_t>(depth, full_level_plans.size());
  candidate_levels.reserve(level_count);
  for (std::size_t level_index = 0; level_index < level_count; ++level_index) {
    auto level = full_level_plans.at(level_index);
    level.is_leaf_level = (level_index + 1U == level_count);
    level.selected_has_any_buffer = false;
    level.selected_has_terminal_branch_buffer = false;
    level.selected_leaf_buffer_cell_master.clear();
    level.selected_terminal_cell_master.clear();
    level.segment_pattern_id = PatternId::segment(0);
    candidate_levels.push_back(std::move(level));
  }

  return candidate_levels;
}

auto CountCandidateLeafNodes(const Tree& topology, unsigned depth) -> std::size_t
{
  const auto topology_levels = topology.levels();
  if (depth == 0U || depth >= topology_levels.size()) {
    return 0U;
  }
  return topology_levels.at(depth).size();
}

auto ResolveDepthCandidates(unsigned max_depth, const HTree::Config& config) -> std::vector<unsigned>
{
  if (max_depth == 0U) {
    return {};
  }

  if (config.target_depth.has_value()) {
    const unsigned resolved_depth = std::clamp(*config.target_depth, 1U, max_depth);
    return {resolved_depth};
  }

  const unsigned requested_window = config.depth_explore_window;
  const unsigned resolved_window = std::max(1U, std::min(requested_window, max_depth));

  std::vector<unsigned> candidates;
  candidates.reserve(resolved_window);
  for (unsigned offset = 0U; offset < resolved_window; ++offset) {
    candidates.push_back(max_depth - offset);
  }
  return candidates;
}

}  // namespace icts::htree
