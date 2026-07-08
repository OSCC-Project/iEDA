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
 * @file OptimizationPolicy.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief Optimizer-owned policy defaults for CTS post-synthesis optimization.
 */

#include "optimization/policy/OptimizationPolicy.hh"

namespace icts::clock_sizing_optimization {

auto DefaultOptimizationPolicy() -> const OptimizationPolicy&
{
  static const OptimizationPolicy policy;
  return policy;
}

auto ValidateOptimizationPolicy(const OptimizationPolicy& policy) -> bool
{
  return policy.max_iterations > 0U && policy.max_trials > 0U && policy.max_frontier_sinks > 0U && policy.max_batch_edits > 0U
         && policy.max_batch_trials_per_iteration > 0U && policy.trial_progress_interval > 0U && policy.initial_detailed_trials > 0U
         && policy.slow_trial_log_threshold_s > 0.0 && policy.max_scalable_batch_edits > 0U
         && policy.max_scalable_exact_trials_per_iteration > 0U && policy.target_window_shrink_ratio > 0.0
         && policy.target_window_shrink_ratio <= 1.0 && policy.min_branch_purity >= 0.0 && policy.min_branch_purity <= 1.0
         && policy.max_opposite_violation_ratio >= 0.0 && policy.route_tree_progress_interval > 0U
         && policy.route_tree_initial_detail_net_count > 0U && policy.route_tree_slow_net_threshold_s > 0.0;
}

}  // namespace icts::clock_sizing_optimization
