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
 * @file OptimizationScalableCandidates.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief Scalable candidate generation helpers for CTS post-synthesis optimization.
 */

#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <compare>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "FastSta.hh"
#include "Log.hh"
#include "optimization/candidate/OptimizationCandidates.hh"
#include "optimization/model/ClockSizingOptimizationData.hh"
#include "optimization/policy/OptimizationPolicy.hh"
#include "optimization/state/OptimizationState.hh"

namespace icts::clock_sizing_optimization {

namespace {

auto MakeArrivalWindow(const ClockSizingTimingState& current, double target_skew_ns) -> ClockSizingArrivalWindow
{
  ClockSizingArrivalWindow window;
  if (!current.skew.valid) {
    return window;
  }
  window.center_ns = 0.5 * (current.skew.min_arrival_ns + current.skew.max_arrival_ns);
  window.staged_skew_ns = std::max(target_skew_ns, current.skew.skew_ns * DefaultOptimizationPolicy().target_window_shrink_ratio);
  const double half_window_ns = 0.5 * window.staged_skew_ns;
  window.lower_ns = window.center_ns - half_window_ns;
  window.upper_ns = window.center_ns + half_window_ns;
  return window;
}

auto AppendPostOrderFromRoot(FastStaNodeId root_id, const ClockSizingTopologyIndex& topology, std::vector<unsigned char>& visit_state,
                             std::vector<FastStaNodeId>& post_order) -> void
{
  if (root_id >= topology.children_by_node.size() || root_id >= visit_state.size()) {
    return;
  }
  std::vector<std::pair<FastStaNodeId, bool>> pending;
  pending.emplace_back(root_id, false);
  while (!pending.empty()) {
    const auto [node_id, expanded] = pending.back();
    pending.pop_back();
    if (node_id >= visit_state.size()) {
      continue;
    }
    if (expanded) {
      visit_state.at(node_id) = 2U;
      post_order.push_back(node_id);
      continue;
    }
    if (visit_state.at(node_id) != 0U) {
      continue;
    }
    visit_state.at(node_id) = 1U;
    pending.emplace_back(node_id, true);
    if (node_id >= topology.children_by_node.size()) {
      continue;
    }
    const auto& children = topology.children_by_node.at(node_id);
    for (const auto child_id : children) {
      if (child_id < visit_state.size() && visit_state.at(child_id) == 0U) {
        pending.emplace_back(child_id, false);
      }
    }
  }
}

auto CollectTopologyPostOrder(const ClockSizingTopologyIndex& topology) -> std::vector<FastStaNodeId>
{
  std::vector<FastStaNodeId> post_order;
  post_order.reserve(topology.parent_by_node.size());
  std::vector<unsigned char> visit_state(topology.parent_by_node.size(), 0U);
  AppendPostOrderFromRoot(topology.source_node_id, topology, visit_state, post_order);
  for (FastStaNodeId node_id = 0U; node_id < topology.parent_by_node.size(); ++node_id) {
    if (visit_state.at(node_id) == 0U) {
      AppendPostOrderFromRoot(node_id, topology, visit_state, post_order);
    }
  }
  return post_order;
}

auto AccumulateWindowStatsFromChild(ClockSizingTopologyWindowStats& stats, FastStaNodeId node_id, FastStaNodeId child_id) -> void
{
  if (node_id >= stats.sink_count_by_node.size() || child_id >= stats.sink_count_by_node.size()) {
    return;
  }
  stats.sink_count_by_node.at(node_id) += stats.sink_count_by_node.at(child_id);
  stats.late_count_by_node.at(node_id) += stats.late_count_by_node.at(child_id);
  stats.early_count_by_node.at(node_id) += stats.early_count_by_node.at(child_id);
  stats.late_violation_by_node.at(node_id) += stats.late_violation_by_node.at(child_id);
  stats.early_violation_by_node.at(node_id) += stats.early_violation_by_node.at(child_id);
  stats.min_arrival_by_node.at(node_id) = std::min(stats.min_arrival_by_node.at(node_id), stats.min_arrival_by_node.at(child_id));
  stats.max_arrival_by_node.at(node_id) = std::max(stats.max_arrival_by_node.at(node_id), stats.max_arrival_by_node.at(child_id));
}

auto BuildTopologyWindowStats(const FastSTA& fast_sta, FastStaClockId clock_id, const std::vector<ClockSizingBuffer>& buffers,
                              const ClockSizingTopologyIndex& topology, const ClockSizingArrivalWindow& window)
    -> ClockSizingTopologyWindowStats
{
  ClockSizingTopologyWindowStats stats;
  const auto graph_profile = fast_sta.queryClockGraphProfile(clock_id);
  if (!graph_profile.has_value()) {
    return stats;
  }

  const auto node_count = graph_profile->node_count;
  stats.sink_count_by_node.assign(node_count, 0U);
  stats.late_count_by_node.assign(node_count, 0U);
  stats.early_count_by_node.assign(node_count, 0U);
  stats.late_violation_by_node.assign(node_count, 0.0);
  stats.early_violation_by_node.assign(node_count, 0.0);
  stats.min_arrival_by_node.assign(node_count, std::numeric_limits<double>::infinity());
  stats.max_arrival_by_node.assign(node_count, -std::numeric_limits<double>::infinity());

  const auto post_order = CollectTopologyPostOrder(topology);
  std::unordered_map<FastStaNodeId, double> arrival_by_sink;
  for (const auto& sink_arrival : fast_sta.collectClockSinkArrivals(clock_id)) {
    arrival_by_sink[sink_arrival.node_id] = sink_arrival.arrival_ns;
  }
  for (const auto node_id : post_order) {
    if (node_id >= node_count) {
      continue;
    }
    const auto arrival_iter = arrival_by_sink.find(node_id);
    if (arrival_iter != arrival_by_sink.end()) {
      const double arrival_ns = arrival_iter->second;
      stats.sink_count_by_node.at(node_id) = 1U;
      stats.min_arrival_by_node.at(node_id) = arrival_ns;
      stats.max_arrival_by_node.at(node_id) = arrival_ns;
      if (arrival_ns > window.upper_ns + kClockSizingEpsilon) {
        stats.late_count_by_node.at(node_id) = 1U;
        stats.late_violation_by_node.at(node_id) = arrival_ns - window.upper_ns;
      } else if (arrival_ns < window.lower_ns - kClockSizingEpsilon) {
        stats.early_count_by_node.at(node_id) = 1U;
        stats.early_violation_by_node.at(node_id) = window.lower_ns - arrival_ns;
      }
    }
    if (node_id >= topology.children_by_node.size()) {
      continue;
    }
    for (const auto child_id : topology.children_by_node.at(node_id)) {
      AccumulateWindowStatsFromChild(stats, node_id, child_id);
    }
  }

  for (const auto& buffer : buffers) {
    const auto node_id = buffer.node_id;
    if (node_id >= stats.sink_count_by_node.size() || stats.sink_count_by_node.at(node_id) == 0U) {
      continue;
    }
    const auto late_count = stats.late_count_by_node.at(node_id);
    const auto early_count = stats.early_count_by_node.at(node_id);
    if (late_count == 0U && early_count == 0U) {
      ++stats.neutral_buffer_count;
    } else if (late_count > 0U && early_count == 0U) {
      ++stats.late_pure_buffer_count;
    } else if (early_count > 0U && late_count == 0U) {
      ++stats.early_pure_buffer_count;
    } else {
      ++stats.mixed_buffer_count;
    }
  }
  return stats;
}

auto ScoreWindowSizingEdit(const ClockSizingEdit& edit, const std::vector<ClockSizingBuffer>& buffers,
                           const ClockSizingTopologyWindowStats& stats, bool& mixed_rejected) -> double
{
  mixed_rejected = false;
  if (edit.drive_step == 0 || edit.buffer_index >= buffers.size()) {
    return 0.0;
  }
  const auto node_id = buffers.at(edit.buffer_index).node_id;
  if (node_id >= stats.sink_count_by_node.size() || stats.sink_count_by_node.at(node_id) == 0U) {
    return 0.0;
  }

  const bool late_edit = edit.drive_step > 0;
  const auto primary_count = late_edit ? stats.late_count_by_node.at(node_id) : stats.early_count_by_node.at(node_id);
  const auto opposite_count = late_edit ? stats.early_count_by_node.at(node_id) : stats.late_count_by_node.at(node_id);
  const double primary_violation = late_edit ? stats.late_violation_by_node.at(node_id) : stats.early_violation_by_node.at(node_id);
  const double opposite_violation = late_edit ? stats.early_violation_by_node.at(node_id) : stats.late_violation_by_node.at(node_id);
  if (primary_count == 0U || primary_violation <= kClockSizingEpsilon) {
    return 0.0;
  }

  const auto active_count = static_cast<double>(primary_count + opposite_count);
  const double purity = active_count <= 0.0 ? 0.0 : static_cast<double>(primary_count) / active_count;
  if (opposite_count > 0U
      && (purity < DefaultOptimizationPolicy().min_branch_purity
          || opposite_violation > primary_violation * DefaultOptimizationPolicy().max_opposite_violation_ratio + kClockSizingEpsilon)) {
    mixed_rejected = true;
    return 0.0;
  }

  const double drive_weight = 1.0 + 0.2 * static_cast<double>(std::max(0, std::abs(edit.drive_step) - 1));
  const double count_weight = 1.0 + std::log1p(static_cast<double>(primary_count));
  const double pure_bonus = opposite_count == 0U ? 1.2 : 1.0;
  const double area_cost = late_edit ? 1.0 + 0.005 * std::max(0.0, edit.area_delta_um2) : 1.0;
  return primary_violation * count_weight * drive_weight * pure_bonus * purity / area_cost;
}

auto ScoreScalableBatch(const std::vector<ClockSizingEdit>& edits, const std::vector<ClockSizingBuffer>& buffers,
                        const ClockSizingTopologyWindowStats& stats) -> double
{
  double score = 0.0;
  for (const auto& edit : edits) {
    bool mixed_rejected = false;
    score += ScoreWindowSizingEdit(edit, buffers, stats, mixed_rejected);
  }
  return score;
}

auto BatchKey(const std::vector<ClockSizingEdit>& edits) -> std::string
{
  auto sorted_edits = edits;
  std::ranges::sort(sorted_edits, [](const ClockSizingEdit& lhs, const ClockSizingEdit& rhs) -> bool {
    if (lhs.buffer_index != rhs.buffer_index) {
      return lhs.buffer_index < rhs.buffer_index;
    }
    return lhs.to_master < rhs.to_master;
  });

  std::string key;
  for (const auto& edit : sorted_edits) {
    key += std::to_string(edit.buffer_index);
    key += ':';
    key += edit.to_master;
    key += ';';
  }
  return key;
}

auto AppendScoredBatch(std::vector<ScoredClockSizingBatch>& candidates, std::unordered_set<std::string>& seen,
                       const std::vector<ClockSizingEdit>& edits, const std::vector<ClockSizingBuffer>& buffers,
                       const ClockSizingTopologyWindowStats& stats, std::size_t max_edit_count = 0U) -> void
{
  if (edits.empty()) {
    return;
  }

  const auto edit_count_limit = max_edit_count == 0U ? DefaultOptimizationPolicy().max_scalable_batch_edits : max_edit_count;
  std::vector<ClockSizingEdit> compact;
  compact.reserve(std::min(edits.size(), edit_count_limit));
  std::unordered_set<std::size_t> used_buffers;
  for (const auto& edit : edits) {
    if (used_buffers.contains(edit.buffer_index)) {
      continue;
    }
    used_buffers.insert(edit.buffer_index);
    compact.push_back(edit);
    if (compact.size() >= edit_count_limit) {
      break;
    }
  }
  if (compact.empty()) {
    return;
  }

  const auto key = BatchKey(compact);
  if (!seen.insert(key).second) {
    return;
  }
  const double score = ScoreScalableBatch(compact, buffers, stats);
  if (score <= 0.0) {
    return;
  }
  candidates.push_back(ScoredClockSizingBatch{.edits = std::move(compact), .score = score});
}

auto CollectScoredClockSizingEdits(const std::vector<ClockSizingBuffer>& buffers, const ClockSizingTopologyWindowStats& stats)
    -> ClockSizingEditScoreSet
{
  ClockSizingEditScoreSet collection;
  collection.edits.reserve(buffers.size() * 4U);
  const auto& rank_steps = DefaultOptimizationPolicy().rank_steps;
  for (std::size_t buffer_index = 0U; buffer_index < buffers.size(); ++buffer_index) {
    for (const auto rank_step : rank_steps) {
      for (const auto side : {ClockSizingFrontierSide::kLate, ClockSizingFrontierSide::kEarly}) {
        auto edit = MakeClockSizingEdit(buffers, buffer_index, side, rank_step);
        if (!edit.has_value()) {
          continue;
        }
        ++collection.raw_edit_count;
        bool mixed_rejected = false;
        const double score = ScoreWindowSizingEdit(*edit, buffers, stats, mixed_rejected);
        if (score > 0.0) {
          collection.edits.push_back(ScoredClockSizingEdit{.edit = std::move(*edit), .score = score});
        } else if (mixed_rejected) {
          ++collection.mixed_rejected_count;
        } else {
          ++collection.no_window_rejected_count;
        }
      }
    }
  }

  std::ranges::sort(collection.edits, [](const ScoredClockSizingEdit& lhs, const ScoredClockSizingEdit& rhs) -> bool {
    if (std::abs(lhs.score - rhs.score) > kClockSizingEpsilon) {
      return lhs.score > rhs.score;
    }
    if (std::abs(lhs.edit.area_delta_um2 - rhs.edit.area_delta_um2) > kClockSizingEpsilon) {
      return lhs.edit.area_delta_um2 < rhs.edit.area_delta_um2;
    }
    if (lhs.edit.buffer_index != rhs.edit.buffer_index) {
      return lhs.edit.buffer_index < rhs.edit.buffer_index;
    }
    return lhs.edit.to_master < rhs.edit.to_master;
  });
  return collection;
}

auto IsTopologyAncestor(const ClockSizingTopologyIndex& topology, FastStaNodeId ancestor_id, FastStaNodeId node_id) -> bool
{
  if (ancestor_id >= topology.parent_by_node.size() || node_id >= topology.parent_by_node.size()) {
    return false;
  }
  std::size_t step_count = 0U;
  while (node_id < topology.parent_by_node.size() && step_count <= topology.parent_by_node.size()) {
    if (node_id == ancestor_id) {
      return true;
    }
    node_id = topology.parent_by_node.at(node_id);
    ++step_count;
  }
  return false;
}

auto HasTopologySelectionConflict(const ClockSizingTopologyIndex& topology, const std::vector<ClockSizingBuffer>& buffers,
                                  const ClockSizingEdit& edit, const std::vector<FastStaNodeId>& selected_nodes) -> bool
{
  if (edit.buffer_index >= buffers.size()) {
    return true;
  }
  const auto candidate_node_id = buffers.at(edit.buffer_index).node_id;
  for (const auto selected_node_id : selected_nodes) {
    if (selected_node_id == candidate_node_id || IsTopologyAncestor(topology, selected_node_id, candidate_node_id)
        || IsTopologyAncestor(topology, candidate_node_id, selected_node_id)) {
      return true;
    }
  }
  return false;
}

auto SelectTopClockSizingEdits(const std::vector<ScoredClockSizingEdit>& scored_edits, const ClockSizingTopologyIndex& topology,
                               const std::vector<ClockSizingBuffer>& buffers, int drive_sign, std::size_t max_edit_count)
    -> std::vector<ClockSizingEdit>
{
  std::vector<ClockSizingEdit> edits;
  edits.reserve(max_edit_count);
  std::unordered_set<std::size_t> used_buffers;
  std::vector<FastStaNodeId> selected_nodes;
  selected_nodes.reserve(max_edit_count);
  for (const auto& scored_edit : scored_edits) {
    if (drive_sign > 0 && scored_edit.edit.drive_step <= 0) {
      continue;
    }
    if (drive_sign < 0 && scored_edit.edit.drive_step >= 0) {
      continue;
    }
    if (used_buffers.contains(scored_edit.edit.buffer_index)) {
      continue;
    }
    if (HasTopologySelectionConflict(topology, buffers, scored_edit.edit, selected_nodes)) {
      continue;
    }
    used_buffers.insert(scored_edit.edit.buffer_index);
    selected_nodes.push_back(buffers.at(scored_edit.edit.buffer_index).node_id);
    edits.push_back(scored_edit.edit);
    if (edits.size() >= max_edit_count) {
      break;
    }
  }
  return edits;
}

auto NormalizedBatchScore(const ScoredClockSizingBatch& candidate) -> double
{
  return candidate.edits.empty() ? 0.0 : candidate.score / std::sqrt(static_cast<double>(candidate.edits.size()));
}

}  // namespace

auto GenerateScalableClockSizingEditBatches(const FastSTA& fast_sta, FastStaClockId clock_id, const std::vector<ClockSizingBuffer>& buffers,
                                            const ClockSizingTopologyIndex& topology, const ClockSizingTimingState& current,
                                            double target_skew_ns) -> std::vector<ScoredClockSizingBatch>
{
  std::vector<ScoredClockSizingBatch> candidates;
  std::unordered_set<std::string> seen;
  const auto window = MakeArrivalWindow(current, target_skew_ns);
  const auto stats = BuildTopologyWindowStats(fast_sta, clock_id, buffers, topology, window);
  const auto scored_edit_collection = CollectScoredClockSizingEdits(buffers, stats);
  const auto& scored_edits = scored_edit_collection.edits;
  LOG_INFO << "Optimization: scalable target window lower=" << window.lower_ns << " ns, upper=" << window.upper_ns
           << " ns, staged_skew=" << window.staged_skew_ns << " ns, late_pure_buffers=" << stats.late_pure_buffer_count
           << ", early_pure_buffers=" << stats.early_pure_buffer_count << ", mixed_buffers=" << stats.mixed_buffer_count
           << ", neutral_buffers=" << stats.neutral_buffer_count << ", raw_edits=" << scored_edit_collection.raw_edit_count
           << ", scored_edits=" << scored_edits.size() << ", mixed_rejected_edits=" << scored_edit_collection.mixed_rejected_count << ".";

  const auto& batch_sizes = DefaultOptimizationPolicy().scalable_batch_sizes;
  for (const auto batch_size : batch_sizes) {
    AppendScoredBatch(candidates, seen, SelectTopClockSizingEdits(scored_edits, topology, buffers, 0, batch_size), buffers, stats);
    AppendScoredBatch(candidates, seen, SelectTopClockSizingEdits(scored_edits, topology, buffers, 1, batch_size), buffers, stats);
    AppendScoredBatch(candidates, seen, SelectTopClockSizingEdits(scored_edits, topology, buffers, -1, batch_size), buffers, stats);
  }

  std::ranges::sort(candidates, [](const ScoredClockSizingBatch& lhs, const ScoredClockSizingBatch& rhs) -> bool {
    const double lhs_normalized_score = NormalizedBatchScore(lhs);
    const double rhs_normalized_score = NormalizedBatchScore(rhs);
    if (std::abs(lhs_normalized_score - rhs_normalized_score) > kClockSizingEpsilon) {
      return lhs_normalized_score > rhs_normalized_score;
    }
    if (std::abs(lhs.score - rhs.score) > kClockSizingEpsilon) {
      return lhs.score > rhs.score;
    }
    if (lhs.edits.size() != rhs.edits.size()) {
      return lhs.edits.size() < rhs.edits.size();
    }
    return FirstClockSizingEditBufferIndex(lhs.edits) < FirstClockSizingEditBufferIndex(rhs.edits);
  });
  return candidates;
}

}  // namespace icts::clock_sizing_optimization
