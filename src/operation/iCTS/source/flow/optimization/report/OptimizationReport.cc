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
 * @file OptimizationReport.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief Report helpers for CTS post-synthesis optimization.
 */

#include "optimization/report/OptimizationReport.hh"

#include <glog/logging.h>

#include <chrono>
#include <cstddef>
#include <map>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "FastSta.hh"
#include "Log.hh"
#include "design/Clock.hh"
#include "logger/LogFormat.hh"
#include "logger/Schema.hh"
#include "optimization/model/ClockSizingOptimizationData.hh"

namespace icts::clock_sizing_optimization {

auto ElapsedSeconds(std::chrono::steady_clock::time_point start_time) -> double
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
}

auto FormatNs(double value) -> std::string
{
  return logformat::FormatWithUnit(value, "ns");
}

auto FormatSeconds(double value) -> std::string
{
  return logformat::FormatWithUnit(value, "s");
}

namespace {

auto SummarizeTransitions(const std::vector<ClockSizingAcceptedEdit>& accepted_edits) -> std::map<std::string, std::size_t>
{
  std::map<std::string, std::size_t> counts;
  for (const auto& accepted_edit : accepted_edits) {
    ++counts[accepted_edit.from_master + " -> " + accepted_edit.to_master];
  }
  return counts;
}

}  // namespace

auto EmitClockSummary(SchemaWriter& reporter, const Clock& clock, const ClockSizingSummary& summary, double target_skew_ns,
                      double runtime_s) -> void
{
  KeyValueFields fields = {
      {"clock", clock.get_clock_name()},
      {"runtime", logformat::FormatWithUnit(runtime_s, "s")},
      {"target_skew", FormatNs(target_skew_ns)},
      {"initial_skew", FormatNs(summary.before.skew.skew_ns)},
      {"optimized_skew", FormatNs(summary.after.skew.skew_ns)},
      {"improvement", FormatNs(summary.before.skew.skew_ns - summary.after.skew.skew_ns)},
      {"initial_area", logformat::FormatWithUnit(summary.before.power.area_um2, "um^2")},
      {"optimized_area", logformat::FormatWithUnit(summary.after.power.area_um2, "um^2")},
      {"area_delta", logformat::FormatWithUnit(summary.after.power.area_um2 - summary.before.power.area_um2, "um^2")},
      {"iteration_count", std::to_string(summary.iteration_count)},
      {"trial_count", std::to_string(summary.trial_count)},
      {"batch_trial_count", std::to_string(summary.batch_trial_count)},
      {"accepted_batch_count", std::to_string(summary.accepted_batch_count)},
      {"accepted_edit_count", std::to_string(summary.accepted_edit_count)},
      {"rejected_candidate_count", std::to_string(summary.rejected_candidate_count)},
      {"cap_rejected_count", std::to_string(summary.cap_rejected_count)},
      {"slew_rejected_count", std::to_string(summary.slew_rejected_count)},
      {"cap_legal", summary.after.cap.legal ? "true" : "false"},
      {"initial_slew_violation_count", std::to_string(summary.before.slew.violation_count)},
      {"initial_buffer_slew_violation_count", std::to_string(summary.before.slew.buffer_violation_count)},
      {"initial_sink_slew_violation_count", std::to_string(summary.before.slew.sink_violation_count)},
      {"slew_legal", summary.after.slew.legal ? "true" : "false"},
      {"slew_violation_count", std::to_string(summary.after.slew.violation_count)},
      {"buffer_slew_violation_count", std::to_string(summary.after.slew.buffer_violation_count)},
      {"sink_slew_violation_count", std::to_string(summary.after.slew.sink_violation_count)},
      {"target_met", summary.target_met ? "true" : "false"},
      {"solve_mode", summary.solve_mode.empty() ? "n/a" : summary.solve_mode},
      {"stop_reason", summary.stop_reason.empty() ? "n/a" : summary.stop_reason},
  };
  EmitKeyValueTable(reporter, "CTS Optimization Clock Summary", fields);

  const auto transition_counts = SummarizeTransitions(summary.accepted_edits);
  TableRows rows;
  rows.reserve(transition_counts.size());
  for (const auto& [transition, count] : transition_counts) {
    rows.push_back({transition, std::to_string(count)});
  }
  if (!rows.empty()) {
    EmitTable(reporter, "CTS Optimization Master Transitions", {"Transition", "Count"}, rows);
  }
}

auto EmitClockProfile(const ClockSizingProfileReportInput& input) -> void
{
  LOG_FATAL_IF(input.reporter == nullptr) << "Optimization profile report requires reporter.";
  LOG_FATAL_IF(input.clock == nullptr) << "Optimization profile report requires clock.";
  LOG_FATAL_IF(input.profile == nullptr) << "Optimization profile report requires profile.";
  auto& reporter = *input.reporter;
  const auto& clock = *input.clock;
  const auto& profile = *input.profile;
  EmitKeyValueTable(reporter, "CTS Optimization Clock Graph Profile",
                    {{"clock", clock.get_clock_name()},
                     {"node_count", std::to_string(profile.node_count)},
                     {"net_count", std::to_string(profile.net_count)},
                     {"sink_count", std::to_string(profile.sink_count)},
                     {"buffer_input_count", std::to_string(profile.buffer_input_count)},
                     {"buffer_output_count", std::to_string(profile.buffer_output_count)},
                     {"optimizable_buffer_count", std::to_string(profile.optimizable_buffer_count)},
                     {"generated_candidate_count", std::to_string(profile.generated_candidate_count)}});

  TableRows rows = {
      {"build_route_tree_cache", FormatSeconds(profile.build_route_tree_cache_s)},
      {"build_fast_sta_context", FormatSeconds(profile.build_fast_sta_context_s)},
      {"inject_route_trees", FormatSeconds(profile.inject_route_trees_s)},
      {"collect_optimizable_buffers", FormatSeconds(profile.collect_optimizable_buffers_s)},
      {"collect_cap_baseline", FormatSeconds(profile.collect_cap_baseline_s)},
      {"collect_slew_baseline", FormatSeconds(profile.collect_slew_baseline_s)},
      {"solve_clock_total", FormatSeconds(profile.solve_clock_s)},
      {"capture_initial_state", FormatSeconds(profile.capture_initial_state_s)},
      {"build_topology_index", FormatSeconds(profile.build_topology_index_s)},
      {"generate_batch_candidates", FormatSeconds(profile.generate_batch_candidates_s)},
      {"batch_trial_eval", FormatSeconds(profile.batch_trial_eval_s)},
      {"apply_accepted_batch", FormatSeconds(profile.apply_accepted_batch_s)},
      {"apply_accepted_edits", FormatSeconds(profile.apply_accepted_edits_s)},
  };
  EmitTable(reporter, "CTS Optimization Clock Runtime Profile", {"Stage", "Runtime"}, rows);
}

}  // namespace icts::clock_sizing_optimization
