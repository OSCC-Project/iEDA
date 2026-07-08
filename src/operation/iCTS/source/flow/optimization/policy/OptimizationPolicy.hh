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
 * @file OptimizationPolicy.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief Optimizer-owned policy parameters for CTS post-synthesis optimization.
 */

#pragma once

#include <array>
#include <cstddef>

namespace icts::clock_sizing_optimization {

struct OptimizationPolicy
{
  unsigned max_iterations = 64U;
  unsigned max_trials = 20000U;
  std::size_t max_frontier_sinks = 64U;
  std::size_t max_batch_edits = 16U;
  std::size_t max_batch_trials_per_iteration = 1600U;
  unsigned trial_progress_interval = 500U;
  unsigned initial_detailed_trials = 3U;
  double slow_trial_log_threshold_s = 5.0;
  bool stop_at_first_target_skew_batch = true;
  std::size_t scalable_node_threshold = 50000U;
  std::size_t scalable_buffer_threshold = 5000U;
  std::size_t max_scalable_batch_edits = 48U;
  std::size_t max_scalable_exact_trials_per_iteration = 8U;
  double target_window_shrink_ratio = 0.8;
  double min_branch_purity = 0.8;
  double max_opposite_violation_ratio = 0.25;
  std::array<unsigned, 3U> rank_steps = {1U, 2U, 3U};
  std::array<std::size_t, 6U> path_segment_lengths = {1U, 2U, 4U, 6U, 8U, 12U};
  std::array<std::size_t, 5U> frontier_prefix_path_counts = {4U, 8U, 16U, 32U, 64U};
  std::array<std::size_t, 6U> frontier_prefix_lengths = {1U, 2U, 3U, 4U, 6U, 8U};
  std::array<std::size_t, 8U> scalable_batch_sizes = {48U, 32U, 24U, 16U, 8U, 4U, 2U, 1U};
  std::size_t route_tree_progress_interval = 5000U;
  std::size_t route_tree_initial_detail_net_count = 5U;
  double route_tree_slow_net_threshold_s = 1.0;
};

auto DefaultOptimizationPolicy() -> const OptimizationPolicy&;
auto ValidateOptimizationPolicy(const OptimizationPolicy& policy) -> bool;

}  // namespace icts::clock_sizing_optimization
