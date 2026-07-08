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
 * @file ClockSizingOptimizationData.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief Clock sizing optimization state, edit, legality, and metric data.
 */

#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "FastSta.hh"
#include "module/routing/router/Router.hh"

namespace icts {

class Inst;
class Net;

namespace clock_sizing_optimization {

// Numerical comparison tolerance used by the optimizer's timing and area comparisons.
inline constexpr double kClockSizingEpsilon = 1e-12;

struct ClockSizingBufferMaster
{
  std::string cell_master;
  double input_cap_pf = 0.0;
  double output_cap_limit_pf = 0.0;
  double area_um2 = 0.0;
  unsigned drive_rank = 0U;
};

struct ClockSizingBuffer
{
  FastStaNodeId node_id = kInvalidFastStaNodeId;
  Inst* inst = nullptr;
  std::string inst_name;
  std::string current_master;
  std::vector<ClockSizingBufferMaster> candidates;
};

struct ClockSizingCapLimit
{
  double load_cap_pf = 0.0;
  double max_cap_pf = 0.0;
  bool violated = false;
};

struct ClockSizingCapCheck
{
  bool legal = true;
  std::size_t violation_count = 0U;
};

struct ClockSizingSlewLimit
{
  double slew_ns = 0.0;
  double max_slew_ns = 0.0;
  FastStaSlewRole role = FastStaSlewRole::kUnknown;
  bool available = false;
  bool violated = false;
};

struct ClockSizingSlewCheck
{
  bool legal = true;
  std::size_t violation_count = 0U;
  std::size_t buffer_violation_count = 0U;
  std::size_t sink_violation_count = 0U;
};

struct ClockSizingTimingState
{
  bool valid = false;
  FastStaSkewSummary skew;
  FastStaPowerSummary power;
  ClockSizingCapCheck cap;
  ClockSizingSlewCheck slew;
};

struct ClockSizingEdit
{
  std::size_t buffer_index = 0U;
  std::string from_master;
  std::string to_master;
  int drive_step = 0;
  double area_delta_um2 = 0.0;
};

struct ClockSizingEditBatch
{
  bool valid = false;
  std::vector<ClockSizingEdit> edits;
  ClockSizingTimingState state;
};

struct ClockSizingAcceptedEdit
{
  std::string inst_name;
  std::string from_master;
  std::string to_master;
  double area_delta_um2 = 0.0;
};

struct ClockSizingRuntimeProfile
{
  double build_route_tree_cache_s = 0.0;
  double build_fast_sta_context_s = 0.0;
  double inject_route_trees_s = 0.0;
  double collect_optimizable_buffers_s = 0.0;
  double collect_cap_baseline_s = 0.0;
  double collect_slew_baseline_s = 0.0;
  double solve_clock_s = 0.0;
  double apply_accepted_edits_s = 0.0;
  double capture_initial_state_s = 0.0;
  double build_topology_index_s = 0.0;
  double generate_batch_candidates_s = 0.0;
  double batch_trial_eval_s = 0.0;
  double apply_accepted_batch_s = 0.0;
  std::size_t node_count = 0U;
  std::size_t net_count = 0U;
  std::size_t sink_count = 0U;
  std::size_t buffer_input_count = 0U;
  std::size_t buffer_output_count = 0U;
  std::size_t optimizable_buffer_count = 0U;
  std::size_t generated_candidate_count = 0U;
};

struct ClockSizingSummary
{
  bool valid = false;
  bool target_met = false;
  bool changed = false;
  std::string solve_mode;
  std::string stop_reason;
  ClockSizingTimingState before;
  ClockSizingTimingState after;
  unsigned iteration_count = 0U;
  unsigned trial_count = 0U;
  unsigned batch_trial_count = 0U;
  unsigned accepted_edit_count = 0U;
  unsigned accepted_batch_count = 0U;
  unsigned rejected_candidate_count = 0U;
  unsigned cap_rejected_count = 0U;
  unsigned slew_rejected_count = 0U;
  ClockSizingRuntimeProfile profile;
  std::vector<ClockSizingAcceptedEdit> accepted_edits;
};

struct ClockSizingTopologyIndex
{
  FastStaNodeId source_node_id = kInvalidFastStaNodeId;
  std::vector<FastStaNodeId> parent_by_node;
  std::vector<std::vector<FastStaNodeId>> children_by_node;
  std::unordered_map<FastStaNodeId, FastStaNodeId> buffer_input_by_output;
  std::unordered_map<FastStaNodeId, std::size_t> buffer_index_by_output_node;
};

struct ClockSizingArrivalWindow
{
  double center_ns = 0.0;
  double lower_ns = 0.0;
  double upper_ns = 0.0;
  double staged_skew_ns = 0.0;
};

struct ScoredClockSizingEdit
{
  ClockSizingEdit edit;
  double score = 0.0;
};

struct ClockSizingEditScoreSet
{
  std::vector<ScoredClockSizingEdit> edits;
  std::size_t raw_edit_count = 0U;
  std::size_t mixed_rejected_count = 0U;
  std::size_t no_window_rejected_count = 0U;
};

struct ScoredClockSizingBatch
{
  std::vector<ClockSizingEdit> edits;
  double score = 0.0;
};

struct ClockSizingTopologyWindowStats
{
  std::vector<std::size_t> sink_count_by_node;
  std::vector<std::size_t> late_count_by_node;
  std::vector<std::size_t> early_count_by_node;
  std::vector<double> late_violation_by_node;
  std::vector<double> early_violation_by_node;
  std::vector<double> min_arrival_by_node;
  std::vector<double> max_arrival_by_node;
  std::size_t late_pure_buffer_count = 0U;
  std::size_t early_pure_buffer_count = 0U;
  std::size_t mixed_buffer_count = 0U;
  std::size_t neutral_buffer_count = 0U;
};

enum class ClockSizingFrontierSide
{
  kLate,
  kEarly
};

using ClockSizingRouteTreeCache = std::unordered_map<const Net*, Router::ClockSteinerTreeType>;

}  // namespace clock_sizing_optimization
}  // namespace icts
