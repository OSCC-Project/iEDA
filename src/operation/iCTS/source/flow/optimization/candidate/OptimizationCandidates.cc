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
 * @file OptimizationCandidates.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief Candidate generation helpers for CTS post-synthesis optimization.
 */

#include "optimization/candidate/OptimizationCandidates.hh"

#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <compare>
#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "FastSta.hh"
#include "Log.hh"
#include "optimization/model/ClockSizingOptimizationData.hh"
#include "optimization/policy/OptimizationPolicy.hh"
#include "optimization/preparation/OptimizationPreparation.hh"

namespace icts::clock_sizing_optimization {

namespace {

auto DriveStep(const ClockSizingBufferMaster& from, const ClockSizingBufferMaster& to) -> int
{
  return static_cast<int>(to.drive_rank) - static_cast<int>(from.drive_rank);
}

}  // namespace

auto BuildClockSizingTopologyIndex(const ClockSizingTopologyIndexInput& input) -> ClockSizingTopologyIndex
{
  LOG_FATAL_IF(input.fast_sta == nullptr) << "Optimization: topology-index build requires FastSTA.";
  LOG_FATAL_IF(input.buffers == nullptr) << "Optimization: topology-index build requires buffers.";
  const auto& fast_sta = *input.fast_sta;
  const auto& buffers = *input.buffers;
  ClockSizingTopologyIndex topology;
  const auto clock_tree_topology = fast_sta.queryClockTreeTopology(input.clock_id);
  if (!clock_tree_topology.has_value()) {
    return topology;
  }

  topology.source_node_id = clock_tree_topology->source_node_id;
  topology.parent_by_node = clock_tree_topology->parent_by_node;
  topology.children_by_node = clock_tree_topology->children_by_node;
  for (std::size_t buffer_index = 0U; buffer_index < buffers.size(); ++buffer_index) {
    topology.buffer_index_by_output_node[buffers.at(buffer_index).node_id] = buffer_index;
  }
  for (const auto& buffer : buffers) {
    if (buffer.node_id >= topology.parent_by_node.size()) {
      continue;
    }
    const auto buffer_input_id = topology.parent_by_node.at(buffer.node_id);
    if (buffer_input_id < topology.parent_by_node.size()) {
      topology.buffer_input_by_output[buffer.node_id] = buffer_input_id;
    }
  }
  return topology;
}

namespace {

auto CollectFrontierSinks(const FastSTA& fast_sta, FastStaClockId clock_id, const ClockSizingTimingState& current,
                          ClockSizingFrontierSide side) -> std::vector<FastStaNodeId>
{
  std::vector<std::pair<FastStaNodeId, double>> sinks;
  if (!current.skew.valid) {
    return {};
  }

  const auto sink_arrivals = fast_sta.collectClockSinkArrivals(clock_id);
  sinks.reserve(sink_arrivals.size());
  for (const auto& sink_arrival : sink_arrivals) {
    sinks.emplace_back(sink_arrival.node_id, sink_arrival.arrival_ns);
  }
  if (sinks.empty()) {
    const auto skew_extreme_sink_id
        = side == ClockSizingFrontierSide::kLate ? current.skew.max_sink_node_id : current.skew.min_sink_node_id;
    if (const auto arrival = fast_sta.queryClockNodeArrival(clock_id, skew_extreme_sink_id); arrival.has_value()) {
      sinks.emplace_back(skew_extreme_sink_id, *arrival);
    }
  }

  std::ranges::sort(sinks, [side](const auto& lhs, const auto& rhs) -> bool {
    if (std::abs(lhs.second - rhs.second) > kClockSizingEpsilon) {
      return side == ClockSizingFrontierSide::kLate ? lhs.second > rhs.second : lhs.second < rhs.second;
    }
    return lhs.first < rhs.first;
  });
  if (sinks.size() > DefaultOptimizationPolicy().max_frontier_sinks) {
    sinks.resize(DefaultOptimizationPolicy().max_frontier_sinks);
  }

  std::vector<FastStaNodeId> sink_ids;
  sink_ids.reserve(sinks.size());
  for (const auto& [node_id, _] : sinks) {
    sink_ids.push_back(node_id);
  }
  return sink_ids;
}

auto CollectPathBufferIndices(const ClockSizingTopologyIndex& topology, FastStaNodeId sink_node_id) -> std::vector<std::size_t>
{
  std::vector<std::size_t> path;
  FastStaNodeId node_id = sink_node_id;
  std::unordered_set<FastStaNodeId> visited;
  while (node_id < topology.parent_by_node.size() && !visited.contains(node_id)) {
    visited.insert(node_id);
    const auto buffer_iter = topology.buffer_index_by_output_node.find(node_id);
    if (buffer_iter != topology.buffer_index_by_output_node.end()) {
      path.push_back(buffer_iter->second);
    }
    node_id = topology.parent_by_node.at(node_id);
  }
  std::ranges::reverse(path);
  return path;
}

}  // namespace

auto MakeClockSizingEdit(const std::vector<ClockSizingBuffer>& buffers, std::size_t buffer_index, ClockSizingFrontierSide side,
                         unsigned rank_step) -> std::optional<ClockSizingEdit>
{
  if (buffer_index >= buffers.size() || rank_step == 0U) {
    return std::nullopt;
  }
  const auto& buffer = buffers.at(buffer_index);
  const auto* from = FindMasterInfo(buffer.candidates, buffer.current_master);
  if (from == nullptr) {
    return std::nullopt;
  }
  const auto from_rank = static_cast<int>(from->drive_rank);
  const auto target_rank
      = side == ClockSizingFrontierSide::kLate ? from_rank + static_cast<int>(rank_step) : from_rank - static_cast<int>(rank_step);
  if (target_rank < 0 || static_cast<std::size_t>(target_rank) >= buffer.candidates.size()) {
    return std::nullopt;
  }
  const auto& to = buffer.candidates.at(static_cast<std::size_t>(target_rank));
  if (to.cell_master == from->cell_master) {
    return std::nullopt;
  }
  return ClockSizingEdit{.buffer_index = buffer_index,
                         .from_master = from->cell_master,
                         .to_master = to.cell_master,
                         .drive_step = DriveStep(*from, to),
                         .area_delta_um2 = to.area_um2 - from->area_um2};
}

namespace {

auto AppendBatchCandidate(std::vector<std::vector<ClockSizingEdit>>& candidates, std::unordered_set<std::string>& seen,
                          std::vector<ClockSizingEdit> edits) -> void
{
  if (edits.empty() || candidates.size() >= DefaultOptimizationPolicy().max_batch_trials_per_iteration) {
    return;
  }
  std::ranges::sort(edits, [](const ClockSizingEdit& lhs, const ClockSizingEdit& rhs) -> bool {
    if (lhs.buffer_index != rhs.buffer_index) {
      return lhs.buffer_index < rhs.buffer_index;
    }
    return lhs.to_master < rhs.to_master;
  });

  std::vector<ClockSizingEdit> compact;
  compact.reserve(std::min(edits.size(), DefaultOptimizationPolicy().max_batch_edits));
  std::unordered_set<std::size_t> used_buffers;
  for (const auto& edit : edits) {
    if (used_buffers.contains(edit.buffer_index)) {
      continue;
    }
    used_buffers.insert(edit.buffer_index);
    compact.push_back(edit);
    if (compact.size() >= DefaultOptimizationPolicy().max_batch_edits) {
      break;
    }
  }
  if (compact.empty()) {
    return;
  }

  std::string key;
  for (const auto& edit : compact) {
    key += std::to_string(edit.buffer_index);
    key += ':';
    key += edit.to_master;
    key += ';';
  }
  if (!seen.insert(key).second) {
    return;
  }
  candidates.push_back(std::move(compact));
}

auto GeneratePathSegmentBatches(const std::vector<ClockSizingBuffer>& buffers, const std::vector<std::size_t>& path,
                                ClockSizingFrontierSide side, unsigned rank_step, std::vector<std::vector<ClockSizingEdit>>& candidates,
                                std::unordered_set<std::string>& seen) -> void
{
  const auto& segment_lengths = DefaultOptimizationPolicy().path_segment_lengths;
  for (std::size_t start = 0U; start < path.size() && candidates.size() < DefaultOptimizationPolicy().max_batch_trials_per_iteration;
       ++start) {
    for (const auto length : segment_lengths) {
      std::vector<ClockSizingEdit> edits;
      for (std::size_t offset = 0U; offset < length && start + offset < path.size(); ++offset) {
        auto edit = MakeClockSizingEdit(buffers, path.at(start + offset), side, rank_step);
        if (edit.has_value()) {
          edits.push_back(std::move(*edit));
        }
      }
      AppendBatchCandidate(candidates, seen, std::move(edits));
      if (candidates.size() >= DefaultOptimizationPolicy().max_batch_trials_per_iteration) {
        break;
      }
    }
  }
}

auto GenerateFrontierLevelBatches(const std::vector<ClockSizingBuffer>& buffers, const std::vector<std::vector<std::size_t>>& paths,
                                  ClockSizingFrontierSide side, unsigned rank_step, std::vector<std::vector<ClockSizingEdit>>& candidates,
                                  std::unordered_set<std::string>& seen) -> void
{
  std::size_t max_path_size = 0U;
  for (const auto& path : paths) {
    max_path_size = std::max(max_path_size, path.size());
  }
  for (std::size_t depth_index = 0U;
       depth_index < max_path_size && candidates.size() < DefaultOptimizationPolicy().max_batch_trials_per_iteration; ++depth_index) {
    std::vector<ClockSizingEdit> edits;
    for (const auto& path : paths) {
      if (depth_index >= path.size()) {
        continue;
      }
      auto edit = MakeClockSizingEdit(buffers, path.at(depth_index), side, rank_step);
      if (edit.has_value()) {
        edits.push_back(std::move(*edit));
      }
    }
    AppendBatchCandidate(candidates, seen, std::move(edits));
  }
}

auto GenerateFrontierPrefixBatches(const std::vector<ClockSizingBuffer>& buffers, const std::vector<std::vector<std::size_t>>& paths,
                                   ClockSizingFrontierSide side, unsigned rank_step, std::vector<std::vector<ClockSizingEdit>>& candidates,
                                   std::unordered_set<std::string>& seen) -> void
{
  const auto& path_counts = DefaultOptimizationPolicy().frontier_prefix_path_counts;
  const auto& prefix_lengths = DefaultOptimizationPolicy().frontier_prefix_lengths;
  for (const auto path_count : path_counts) {
    if (candidates.size() >= DefaultOptimizationPolicy().max_batch_trials_per_iteration) {
      break;
    }
    const auto selected_path_count = std::min(path_count, paths.size());
    if (selected_path_count == 0U) {
      continue;
    }
    for (const auto prefix_length : prefix_lengths) {
      std::vector<ClockSizingEdit> edits;
      for (std::size_t path_index = 0U; path_index < selected_path_count && edits.size() < DefaultOptimizationPolicy().max_batch_edits;
           ++path_index) {
        const auto& path = paths.at(path_index);
        for (std::size_t depth_index = 0U; depth_index < prefix_length && depth_index < path.size(); ++depth_index) {
          auto edit = MakeClockSizingEdit(buffers, path.at(depth_index), side, rank_step);
          if (edit.has_value()) {
            edits.push_back(std::move(*edit));
          }
        }
      }
      AppendBatchCandidate(candidates, seen, std::move(edits));
      if (candidates.size() >= DefaultOptimizationPolicy().max_batch_trials_per_iteration) {
        break;
      }
    }
  }
}

}  // namespace

auto GenerateClockSizingEditBatches(const FastSTA& fast_sta, FastStaClockId clock_id, const std::vector<ClockSizingBuffer>& buffers,
                                    const ClockSizingTopologyIndex& topology, const ClockSizingTimingState& current)
    -> std::vector<std::vector<ClockSizingEdit>>
{
  std::vector<std::vector<ClockSizingEdit>> candidates;
  std::unordered_set<std::string> seen;
  const auto& rank_steps = DefaultOptimizationPolicy().rank_steps;
  for (const auto side : {ClockSizingFrontierSide::kLate, ClockSizingFrontierSide::kEarly}) {
    const auto frontier_sinks = CollectFrontierSinks(fast_sta, clock_id, current, side);
    std::vector<std::vector<std::size_t>> paths;
    paths.reserve(frontier_sinks.size());
    for (const auto sink_id : frontier_sinks) {
      auto path = CollectPathBufferIndices(topology, sink_id);
      if (!path.empty()) {
        paths.push_back(std::move(path));
      }
    }
    for (const auto rank_step : rank_steps) {
      GenerateFrontierPrefixBatches(buffers, paths, side, rank_step, candidates, seen);
      GenerateFrontierLevelBatches(buffers, paths, side, rank_step, candidates, seen);
      for (const auto& path : paths) {
        GeneratePathSegmentBatches(buffers, path, side, rank_step, candidates, seen);
        if (candidates.size() >= DefaultOptimizationPolicy().max_batch_trials_per_iteration) {
          break;
        }
      }
      if (candidates.size() >= DefaultOptimizationPolicy().max_batch_trials_per_iteration) {
        break;
      }
    }
  }
  return candidates;
}

}  // namespace icts::clock_sizing_optimization
