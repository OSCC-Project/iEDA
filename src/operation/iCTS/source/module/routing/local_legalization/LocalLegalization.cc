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
 * @file LocalLegalization.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-03-17
 * @brief Standalone point-based local legalization solver.
 */
#include "LocalLegalization.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <map>
#include <optional>
#include <ostream>
#include <utility>
#include <vector>

#include "Log.hh"
#include "geometry/Geometry.hh"

namespace icts {
namespace {

constexpr long long kForbiddenCost = std::numeric_limits<int>::max() / 4;

auto ContainsPoint(const std::vector<LocalLegalization::PointType>& points, const LocalLegalization::PointType& point) -> bool
{
  return std::ranges::find(points, point) != points.end();
}

struct HungarianColumnState
{
  std::vector<std::size_t>* matched_row_by_col = nullptr;
  std::vector<std::size_t>* predecessor_col_by_col = nullptr;
};

auto AdvanceAugmentingColumn(const std::vector<std::vector<long long>>& cost_matrix, std::vector<long long>& row_potential,
                             std::vector<long long>& col_potential, const HungarianColumnState& column_state, std::size_t col_count,
                             std::size_t& col0) -> void
{
  std::vector<long long> min_v(col_count + 1, std::numeric_limits<long long>::max());
  std::vector<bool> used(col_count + 1, false);

  while (true) {
    used.at(col0) = true;
    const std::size_t row0 = column_state.matched_row_by_col->at(col0);
    long long delta = std::numeric_limits<long long>::max();
    std::size_t next_col = 0;

    for (std::size_t col = 1; col <= col_count; ++col) {
      if (used.at(col)) {
        continue;
      }
      const long long current = cost_matrix.at(row0 - 1).at(col - 1) - row_potential.at(row0) - col_potential.at(col);
      if (current < min_v.at(col)) {
        min_v.at(col) = current;
        column_state.predecessor_col_by_col->at(col) = col0;
      }
      if (min_v.at(col) < delta) {
        delta = min_v.at(col);
        next_col = col;
      }
    }

    for (std::size_t col = 0; col <= col_count; ++col) {
      if (used.at(col)) {
        row_potential.at(column_state.matched_row_by_col->at(col)) += delta;
        col_potential.at(col) -= delta;
        continue;
      }
      min_v.at(col) -= delta;
    }

    col0 = next_col;
    if (column_state.matched_row_by_col->at(col0) == 0) {
      break;
    }
  }
}

auto ApplyAugmentingPath(std::vector<std::size_t>& matched_row_by_col, const std::vector<std::size_t>& predecessor_col_by_col,
                         std::size_t col0) -> void
{
  while (true) {
    const std::size_t prev_col = predecessor_col_by_col.at(col0);
    matched_row_by_col.at(col0) = matched_row_by_col.at(prev_col);
    col0 = prev_col;
    if (col0 == 0) {
      break;
    }
  }
}

auto HungarianSolve(const std::vector<std::vector<long long>>& cost_matrix) -> std::vector<std::size_t>
{
  const std::size_t row_count = cost_matrix.size();
  const std::size_t col_count = row_count == 0 ? 0 : cost_matrix.front().size();
  if (row_count == 0 || col_count == 0 || row_count > col_count) {
    return {};
  }

  std::vector<long long> row_potential(row_count + 1, 0);
  std::vector<long long> col_potential(col_count + 1, 0);
  std::vector<std::size_t> matched_row_by_col(col_count + 1, 0);
  std::vector<std::size_t> predecessor_col_by_col(col_count + 1, 0);

  const HungarianColumnState column_state{.matched_row_by_col = &matched_row_by_col, .predecessor_col_by_col = &predecessor_col_by_col};
  for (std::size_t row = 1; row <= row_count; ++row) {
    matched_row_by_col.at(0) = row;
    std::size_t col0 = 0;
    AdvanceAugmentingColumn(cost_matrix, row_potential, col_potential, column_state, col_count, col0);
    ApplyAugmentingPath(matched_row_by_col, predecessor_col_by_col, col0);
  }

  std::vector<std::size_t> assignment(row_count, std::numeric_limits<std::size_t>::max());
  for (std::size_t col = 1; col <= col_count; ++col) {
    if (matched_row_by_col.at(col) != 0) {
      assignment.at(matched_row_by_col.at(col) - 1) = col - 1;
    }
  }
  return assignment;
}

}  // namespace

auto LocalLegalization::buildLegalRegion(const Problem& problem) -> LocalLegalization::RegionType
{
  if (problem.feasible_region.empty()) {
    return problem.feasible_region;
  }
  auto legal_region = problem.feasible_region;
  legal_region.subtract(problem.block_region);
  return legal_region;
}

auto LocalLegalization::legalize(const Problem& problem) -> LocalLegalization::Output
{
  return legalize(problem, Config{});
}

auto LocalLegalization::legalize(const Problem& problem, const Config& config) -> LocalLegalization::Output
{
  Output result;
  result.legalized_points = problem.movable_points;
  if (problem.movable_points.empty()) {
    LOG_WARNING << "LocalLegalization skipped: movable point set is empty.";
    result.success = true;
    return result;
  }

  const auto legal_region = buildLegalRegion(problem);
  for (int round = 0; round < std::max(1, config.max_expansion_rounds); ++round) {
    const auto candidate_budget = std::max<std::size_t>(1, config.candidate_budget * static_cast<std::size_t>(round + 1));
    const int local_search_radius = std::max(0, config.local_search_radius * (round + 1));

    std::vector<std::vector<CandidateSite>> candidate_sets;
    candidate_sets.reserve(problem.movable_points.size());

    bool complete = true;
    for (const auto& origin : problem.movable_points) {
      auto candidates = generateCandidates(origin, legal_region, problem.fixed_points, candidate_budget, local_search_radius);
      if (candidates.empty()) {
        complete = false;
        break;
      }
      candidate_sets.push_back(std::move(candidates));
    }

    if (!complete) {
      LOG_WARNING << "LocalLegalization expansion round " << round << " generated incomplete candidate sets; retrying with wider search.";
      continue;
    }

    auto legalized_points = solveAssignment(problem.movable_points, candidate_sets);
    if (!legalized_points.empty()) {
      result.legalized_points = std::move(legalized_points);
      result.total_displacement = computeTotalDisplacement(problem.movable_points, result.legalized_points);
      result.success = true;
      return result;
    }
  }

  if (config.failure_policy == FailurePolicy::kKeepOriginal) {
    LOG_WARNING << "LocalLegalization failed to find a legal assignment; keeping original point locations.";
    result.legalized_points = problem.movable_points;
    return result;
  }

  LOG_ERROR << "LocalLegalization failed to find a legal assignment.";
  return result;
}

auto LocalLegalization::legalize(std::vector<PointType>& movable_points, const std::vector<PointType>& fixed_points,
                                 const RegionType& feasible_region, const RegionType& block_region) -> LocalLegalization::Output
{
  return legalize(movable_points, fixed_points, feasible_region, block_region, Config{});
}

auto LocalLegalization::legalize(std::vector<PointType>& movable_points, const std::vector<PointType>& fixed_points,
                                 const RegionType& feasible_region, const RegionType& block_region, const Config& config)
    -> LocalLegalization::Output
{
  Problem problem;
  problem.movable_points = movable_points;
  problem.fixed_points = fixed_points;
  problem.feasible_region = feasible_region;
  problem.block_region = block_region;

  auto result = legalize(problem, config);
  if (result.success || config.failure_policy == FailurePolicy::kKeepOriginal) {
    movable_points = result.legalized_points;
  }
  return result;
}

auto LocalLegalization::generateCandidates(const PointType& origin, const RegionType& legal_region,
                                           const std::vector<PointType>& fixed_points, std::size_t candidate_budget,
                                           int local_search_radius) -> std::vector<LocalLegalization::CandidateSite>
{
  std::vector<CandidateSite> candidates;
  candidates.reserve(candidate_budget);

  appendCandidate(candidates, origin, legal_region, fixed_points, candidate_budget);

  std::vector<PointType> seeds;
  if (legal_region.empty()) {
    seeds.push_back(origin);
  } else {
    if (auto nearest = geometry::ProjectNearest(legal_region, origin); nearest.has_value()) {
      seeds.push_back(*nearest);
      appendCandidate(candidates, *nearest, legal_region, fixed_points, candidate_budget);
    }
    for (const auto& rect : legal_region.rects()) {
      auto projected = rect.clamp(origin);
      seeds.push_back(projected);
      appendCandidate(candidates, projected, legal_region, fixed_points, candidate_budget);
    }
  }

  auto boundary_candidates = enumerateBoundaryBreakpoints(origin, legal_region, fixed_points, candidate_budget);
  for (const auto& candidate : boundary_candidates) {
    seeds.push_back(candidate.point);
    appendCandidate(candidates, candidate.point, legal_region, fixed_points, candidate_budget);
  }

  for (const auto& seed : seeds) {
    auto neighbor_candidates = enumerateProjectedNeighbors(seed, legal_region, fixed_points, local_search_radius, candidate_budget);
    for (const auto& candidate : neighbor_candidates) {
      appendCandidate(candidates, candidate.point, legal_region, fixed_points, candidate_budget);
    }
  }

  std::ranges::sort(candidates, [&](const CandidateSite& lhs, const CandidateSite& rhs) -> bool {
    const auto lhs_dist = geometry::Manhattan(origin, lhs.point);
    const auto rhs_dist = geometry::Manhattan(origin, rhs.point);
    if (lhs_dist != rhs_dist) {
      return lhs_dist < rhs_dist;
    }
    return lhs.point < rhs.point;
  });
  if (candidates.size() > candidate_budget) {
    candidates.resize(candidate_budget);
  }
  return candidates;
}

auto LocalLegalization::enumerateProjectedNeighbors(const PointType& seed, const RegionType& legal_region,
                                                    const std::vector<PointType>& fixed_points, int max_radius,
                                                    std::size_t candidate_budget) -> std::vector<LocalLegalization::CandidateSite>
{
  std::vector<CandidateSite> candidates;
  candidates.reserve(candidate_budget);

  for (int radius = 1; radius <= max_radius && candidates.size() < candidate_budget; ++radius) {
    for (int dx = -radius; dx <= radius && candidates.size() < candidate_budget; ++dx) {
      const int delta_y = radius - std::abs(dx);
      appendCandidate(candidates, PointType(seed.get_x() + dx, seed.get_y() + delta_y), legal_region, fixed_points, candidate_budget);
      if (delta_y != 0 && candidates.size() < candidate_budget) {
        appendCandidate(candidates, PointType(seed.get_x() + dx, seed.get_y() - delta_y), legal_region, fixed_points, candidate_budget);
      }
    }
  }

  return candidates;
}

auto LocalLegalization::enumerateBoundaryBreakpoints(const PointType& origin, const RegionType& legal_region,
                                                     const std::vector<PointType>& fixed_points, std::size_t candidate_budget)
    -> std::vector<LocalLegalization::CandidateSite>
{
  std::vector<CandidateSite> candidates;
  candidates.reserve(candidate_budget);

  if (legal_region.empty()) {
    return candidates;
  }

  for (const auto& rect : legal_region.rects()) {
    const int clamped_x = std::clamp(origin.get_x(), rect.get_min_x(), rect.get_max_x());
    const int clamped_y = std::clamp(origin.get_y(), rect.get_min_y(), rect.get_max_y());

    appendCandidate(candidates, PointType(rect.get_min_x(), rect.get_min_y()), legal_region, fixed_points, candidate_budget);
    appendCandidate(candidates, PointType(rect.get_min_x(), rect.get_max_y()), legal_region, fixed_points, candidate_budget);
    appendCandidate(candidates, PointType(rect.get_max_x(), rect.get_min_y()), legal_region, fixed_points, candidate_budget);
    appendCandidate(candidates, PointType(rect.get_max_x(), rect.get_max_y()), legal_region, fixed_points, candidate_budget);
    appendCandidate(candidates, PointType(clamped_x, rect.get_min_y()), legal_region, fixed_points, candidate_budget);
    appendCandidate(candidates, PointType(clamped_x, rect.get_max_y()), legal_region, fixed_points, candidate_budget);
    appendCandidate(candidates, PointType(rect.get_min_x(), clamped_y), legal_region, fixed_points, candidate_budget);
    appendCandidate(candidates, PointType(rect.get_max_x(), clamped_y), legal_region, fixed_points, candidate_budget);
  }

  return candidates;
}

auto LocalLegalization::solveAssignment(const std::vector<PointType>& movable_points,
                                        const std::vector<std::vector<CandidateSite>>& candidate_sets)
    -> std::vector<LocalLegalization::PointType>
{
  if (movable_points.empty()) {
    return {};
  }

  std::map<PointType, std::size_t> point_to_site;
  std::vector<PointType> site_points;
  for (const auto& candidate_set : candidate_sets) {
    for (const auto& candidate : candidate_set) {
      if (point_to_site.contains(candidate.point)) {
        continue;
      }
      point_to_site[candidate.point] = site_points.size();
      site_points.push_back(candidate.point);
    }
  }

  if (site_points.size() < movable_points.size()) {
    return {};
  }

  std::vector<std::vector<long long>> cost_matrix(movable_points.size(), std::vector<long long>(site_points.size(), kForbiddenCost));
  for (std::size_t i = 0; i < movable_points.size(); ++i) {
    for (const auto& candidate : candidate_sets.at(i)) {
      auto site_iter = point_to_site.find(candidate.point);
      if (site_iter == point_to_site.end()) {
        continue;
      }
      cost_matrix.at(i).at(site_iter->second) = geometry::Manhattan(movable_points.at(i), candidate.point);
    }
  }

  auto assignment = HungarianSolve(cost_matrix);
  if (assignment.size() != movable_points.size()) {
    return {};
  }

  std::vector<PointType> legalized_points(movable_points.size(), PointType(-1, -1));
  for (std::size_t i = 0; i < movable_points.size(); ++i) {
    if (assignment.at(i) >= site_points.size() || cost_matrix.at(i).at(assignment.at(i)) >= kForbiddenCost / 2) {
      return {};
    }
    legalized_points.at(i) = site_points.at(assignment.at(i));
  }
  return legalized_points;
}

auto LocalLegalization::computeTotalDisplacement(const std::vector<PointType>& original_points,
                                                 const std::vector<PointType>& legalized_points) -> long long
{
  long long total_displacement = 0;
  const auto count = std::min(original_points.size(), legalized_points.size());
  for (std::size_t i = 0; i < count; ++i) {
    total_displacement += static_cast<long long>(geometry::Manhattan(original_points.at(i), legalized_points.at(i)));
  }
  return total_displacement;
}

auto LocalLegalization::appendCandidate(std::vector<CandidateSite>& candidates, const PointType& point, const RegionType& legal_region,
                                        const std::vector<PointType>& fixed_points, std::size_t candidate_budget) -> void
{
  if (!legal_region.empty() && !legal_region.contains(point)) {
    return;
  }
  if (ContainsPoint(fixed_points, point)) {
    return;
  }
  const bool already_exists = std::ranges::any_of(candidates, [&](const auto& candidate) -> bool { return candidate.point == point; });
  if (already_exists) {
    return;
  }
  if (candidates.size() >= candidate_budget) {
    return;
  }
  candidates.push_back(CandidateSite{point});
}

}  // namespace icts
