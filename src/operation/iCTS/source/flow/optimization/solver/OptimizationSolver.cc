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
 * @file OptimizationSolver.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief Solver loops and trial evaluation for CTS post-synthesis optimization.
 */

#include "optimization/solver/OptimizationSolver.hh"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "FastSta.hh"
#include "Log.hh"
#include "optimization/candidate/OptimizationCandidates.hh"
#include "optimization/model/ClockSizingOptimizationData.hh"
#include "optimization/policy/OptimizationPolicy.hh"
#include "optimization/report/OptimizationReport.hh"
#include "optimization/state/OptimizationState.hh"

namespace icts::clock_sizing_optimization {

namespace {

auto ClockSizingEditDriveMagnitude(const std::vector<ClockSizingEdit>& edits) -> int
{
  int magnitude = 0;
  for (const auto& edit : edits) {
    magnitude += std::abs(edit.drive_step);
  }
  return magnitude;
}

auto PreferTrial(const ClockSizingEditBatch& candidate, const ClockSizingEditBatch& incumbent, const ClockSizingTimingState& current,
                 double target_skew_ns) -> bool
{
  if (!candidate.valid) {
    return false;
  }
  if (!incumbent.valid) {
    return true;
  }

  const bool current_met = TargetMet(current, target_skew_ns);
  const bool candidate_met = TargetMet(candidate.state, target_skew_ns);
  const bool incumbent_met = TargetMet(incumbent.state, target_skew_ns);

  if (!current_met && candidate_met != incumbent_met) {
    return candidate_met;
  }
  if (current_met || candidate_met) {
    if (candidate.state.power.area_um2 < incumbent.state.power.area_um2 - kClockSizingEpsilon) {
      return true;
    }
    if (std::abs(candidate.state.power.area_um2 - incumbent.state.power.area_um2) > kClockSizingEpsilon) {
      return false;
    }
  }

  if (candidate.state.skew.skew_ns < incumbent.state.skew.skew_ns - kClockSizingEpsilon) {
    return true;
  }
  if (std::abs(candidate.state.skew.skew_ns - incumbent.state.skew.skew_ns) > kClockSizingEpsilon) {
    return false;
  }
  if (candidate.state.power.area_um2 < incumbent.state.power.area_um2 - kClockSizingEpsilon) {
    return true;
  }
  if (std::abs(candidate.state.power.area_um2 - incumbent.state.power.area_um2) > kClockSizingEpsilon) {
    return false;
  }
  if (candidate.edits.size() != incumbent.edits.size()) {
    return candidate.edits.size() < incumbent.edits.size();
  }
  const auto candidate_drive_magnitude = ClockSizingEditDriveMagnitude(candidate.edits);
  const auto incumbent_drive_magnitude = ClockSizingEditDriveMagnitude(incumbent.edits);
  if (candidate_drive_magnitude != incumbent_drive_magnitude) {
    return candidate_drive_magnitude < incumbent_drive_magnitude;
  }
  return FirstClockSizingEditBufferIndex(candidate.edits) < FirstClockSizingEditBufferIndex(incumbent.edits);
}

auto ChangeFastStaMasters(FastSTA& fast_sta, FastStaClockId clock_id, const std::vector<FastStaBufferMasterChange>& changes) -> bool
{
  if (!fast_sta.changeBufferMasters(clock_id, changes)) {
    return false;
  }
  const auto analysis_status = fast_sta.queryClockAnalysisStatus(clock_id);
  return analysis_status.has_value() && analysis_status->timing_valid && analysis_status->power_valid;
}

auto ChangeFastStaMastersTimingOnly(FastSTA& fast_sta, FastStaClockId clock_id, const std::vector<FastStaBufferMasterChange>& changes)
    -> bool
{
  if (!fast_sta.changeBufferMastersTimingOnly(clock_id, changes)) {
    return false;
  }
  const auto analysis_status = fast_sta.queryClockAnalysisStatus(clock_id);
  return analysis_status.has_value() && analysis_status->timing_valid;
}

auto BuildMasterChanges(const std::vector<ClockSizingBuffer>& buffers, const std::vector<ClockSizingEdit>& edits, bool restore)
    -> std::vector<FastStaBufferMasterChange>
{
  std::vector<FastStaBufferMasterChange> changes;
  changes.reserve(edits.size());
  for (const auto& edit : edits) {
    if (edit.buffer_index >= buffers.size()) {
      continue;
    }
    changes.push_back(FastStaBufferMasterChange{
        .node_id = buffers.at(edit.buffer_index).node_id,
        .cell_master = restore ? edit.from_master : edit.to_master,
    });
  }
  return changes;
}

auto ClockSizingEditAreaDelta(const std::vector<ClockSizingEdit>& edits) -> double
{
  double area_delta_um2 = 0.0;
  for (const auto& edit : edits) {
    area_delta_um2 += edit.area_delta_um2;
  }
  return area_delta_um2;
}

auto EvaluateClockSizingEditBatch(FastSTA& fast_sta, FastStaClockId clock_id, const std::vector<ClockSizingBuffer>& buffers,
                                  const std::vector<ClockSizingEdit>& edits, const ClockSizingTimingState& current,
                                  const std::vector<ClockSizingCapLimit>& cap_baseline,
                                  const std::vector<ClockSizingSlewLimit>& slew_baseline, double target_skew_ns) -> ClockSizingEditBatch
{
  ClockSizingEditBatch trial;
  trial.edits = edits;
  if (trial.edits.empty()) {
    return trial;
  }

  if (!ChangeFastStaMasters(fast_sta, clock_id, BuildMasterChanges(buffers, trial.edits, false))) {
    return trial;
  }
  trial.state = CaptureState(fast_sta, clock_id, cap_baseline, slew_baseline);
  trial.valid = StateImproves(current, trial.state, target_skew_ns);
  if (!ChangeFastStaMasters(fast_sta, clock_id, BuildMasterChanges(buffers, trial.edits, true))) {
    LOG_FATAL << "Optimization: failed to restore fast STA batch trial.";
  }
  return trial;
}

auto EvaluateClockSizingEditBatchTimingOnly(FastSTA& fast_sta, FastStaClockId clock_id, const std::vector<ClockSizingBuffer>& buffers,
                                            const std::vector<ClockSizingEdit>& edits, const ClockSizingTimingState& current,
                                            const std::vector<ClockSizingCapLimit>& cap_baseline,
                                            const std::vector<ClockSizingSlewLimit>& slew_baseline, double target_skew_ns)
    -> ClockSizingEditBatch
{
  ClockSizingEditBatch trial;
  trial.edits = edits;
  if (trial.edits.empty()) {
    return trial;
  }

  if (!ChangeFastStaMastersTimingOnly(fast_sta, clock_id, BuildMasterChanges(buffers, trial.edits, false))) {
    return trial;
  }
  trial.state = CaptureStateWithArea(fast_sta, clock_id, cap_baseline, slew_baseline,
                                     current.power.area_um2 + ClockSizingEditAreaDelta(trial.edits));
  trial.valid = StateImproves(current, trial.state, target_skew_ns);
  if (!ChangeFastStaMastersTimingOnly(fast_sta, clock_id, BuildMasterChanges(buffers, trial.edits, true))) {
    LOG_FATAL << "Optimization: failed to restore fast STA timing-only batch trial.";
  }
  return trial;
}

auto FindBestClockSizingEditBatch(FastSTA& fast_sta, FastStaClockId clock_id, const std::vector<ClockSizingBuffer>& buffers,
                                  const ClockSizingTopologyIndex& topology, const ClockSizingTimingState& current,
                                  const std::vector<ClockSizingCapLimit>& cap_baseline,
                                  const std::vector<ClockSizingSlewLimit>& slew_baseline, double target_skew_ns,
                                  ClockSizingSummary& summary) -> ClockSizingEditBatch
{
  ClockSizingEditBatch best;
  const auto candidate_start = std::chrono::steady_clock::now();
  const auto candidates = GenerateClockSizingEditBatches(fast_sta, clock_id, buffers, topology, current);
  summary.profile.generate_batch_candidates_s += ElapsedSeconds(candidate_start);
  summary.profile.generated_candidate_count += candidates.size();
  LOG_INFO << "Optimization: solve iteration " << (summary.iteration_count + 1U) << " starts with current_skew=" << current.skew.skew_ns
           << " ns, current_area=" << current.power.area_um2 << " um^2, candidates=" << candidates.size()
           << ", total_trials=" << summary.trial_count << ".";
  const auto iteration_start = std::chrono::steady_clock::now();
  for (std::size_t candidate_index = 0U; candidate_index < candidates.size(); ++candidate_index) {
    if (summary.trial_count >= DefaultOptimizationPolicy().max_trials) {
      break;
    }
    const auto& edits = candidates.at(candidate_index);
    ++summary.trial_count;
    ++summary.batch_trial_count;
    if (summary.trial_count <= DefaultOptimizationPolicy().initial_detailed_trials) {
      LOG_INFO << "Optimization: start batch trial " << summary.trial_count << ", candidate=" << (candidate_index + 1U) << "/"
               << candidates.size() << ", edit_count=" << edits.size() << ".";
    }
    const auto trial_start = std::chrono::steady_clock::now();
    auto trial = EvaluateClockSizingEditBatch(fast_sta, clock_id, buffers, edits, current, cap_baseline, slew_baseline, target_skew_ns);
    const double trial_runtime_s = ElapsedSeconds(trial_start);
    summary.profile.batch_trial_eval_s += trial_runtime_s;
    if (!trial.state.cap.legal) {
      ++summary.cap_rejected_count;
    }
    if (!trial.state.slew.legal) {
      ++summary.slew_rejected_count;
    }
    if (!trial.valid) {
      ++summary.rejected_candidate_count;
      continue;
    }
    if (PreferTrial(trial, best, current, target_skew_ns)) {
      best = std::move(trial);
    }
    if (DefaultOptimizationPolicy().stop_at_first_target_skew_batch && !TargetMet(current, target_skew_ns)
        && TargetMet(best.state, target_skew_ns)) {
      LOG_INFO << "Optimization: target skew reached by exact batch trial, iteration=" << (summary.iteration_count + 1U)
               << ", candidate=" << (candidate_index + 1U) << "/" << candidates.size() << ", total_trials=" << summary.trial_count
               << ", trial_runtime=" << trial_runtime_s << " s, iteration_runtime=" << ElapsedSeconds(iteration_start)
               << " s, target_skew=" << target_skew_ns << " ns, candidate_skew=" << best.state.skew.skew_ns
               << " ns, candidate_area=" << best.state.power.area_um2 << " um^2.";
      return best;
    }
    if (summary.trial_count <= DefaultOptimizationPolicy().initial_detailed_trials
        || summary.trial_count % DefaultOptimizationPolicy().trial_progress_interval == 0U
        || trial_runtime_s >= DefaultOptimizationPolicy().slow_trial_log_threshold_s) {
      LOG_INFO << "Optimization: batch trial progress, iteration=" << (summary.iteration_count + 1U)
               << ", candidate=" << (candidate_index + 1U) << "/" << candidates.size() << ", total_trials=" << summary.trial_count
               << ", trial_runtime=" << trial_runtime_s << " s, iteration_runtime=" << ElapsedSeconds(iteration_start)
               << " s, best_skew=" << (best.valid ? best.state.skew.skew_ns : current.skew.skew_ns) << " ns.";
    }
  }
  return best;
}

auto FindBestScalableClockSizingEditBatch(FastSTA& fast_sta, FastStaClockId clock_id, const std::vector<ClockSizingBuffer>& buffers,
                                          const ClockSizingTopologyIndex& topology, const ClockSizingTimingState& current,
                                          const std::vector<ClockSizingCapLimit>& cap_baseline,
                                          const std::vector<ClockSizingSlewLimit>& slew_baseline, double target_skew_ns,
                                          ClockSizingSummary& summary) -> ClockSizingEditBatch
{
  ClockSizingEditBatch best;
  const auto candidate_start = std::chrono::steady_clock::now();
  const auto candidates = GenerateScalableClockSizingEditBatches(fast_sta, clock_id, buffers, topology, current, target_skew_ns);
  summary.profile.generate_batch_candidates_s += ElapsedSeconds(candidate_start);
  summary.profile.generated_candidate_count += candidates.size();
  LOG_INFO << "Optimization: scalable solve iteration " << (summary.iteration_count + 1U)
           << " starts with current_skew=" << current.skew.skew_ns << " ns, current_area=" << current.power.area_um2
           << " um^2, scored_batches=" << candidates.size()
           << ", exact_trial_limit=" << DefaultOptimizationPolicy().max_scalable_exact_trials_per_iteration
           << ", total_trials=" << summary.trial_count << ".";

  const auto iteration_start = std::chrono::steady_clock::now();
  const auto exact_trial_count = std::min(DefaultOptimizationPolicy().max_scalable_exact_trials_per_iteration, candidates.size());
  for (std::size_t candidate_index = 0U; candidate_index < exact_trial_count; ++candidate_index) {
    if (summary.trial_count >= DefaultOptimizationPolicy().max_trials) {
      break;
    }
    const auto& candidate = candidates.at(candidate_index);
    ++summary.trial_count;
    ++summary.batch_trial_count;
    LOG_INFO << "Optimization: scalable batch trial " << summary.trial_count << ", candidate=" << (candidate_index + 1U) << "/"
             << candidates.size() << ", edit_count=" << candidate.edits.size() << ", score=" << candidate.score << ".";
    const auto trial_start = std::chrono::steady_clock::now();
    auto trial = EvaluateClockSizingEditBatchTimingOnly(fast_sta, clock_id, buffers, candidate.edits, current, cap_baseline, slew_baseline,
                                                        target_skew_ns);
    const double trial_runtime_s = ElapsedSeconds(trial_start);
    summary.profile.batch_trial_eval_s += trial_runtime_s;
    if (!trial.state.cap.legal) {
      ++summary.cap_rejected_count;
    }
    if (!trial.state.slew.legal) {
      ++summary.slew_rejected_count;
    }
    if (!trial.valid) {
      ++summary.rejected_candidate_count;
      const char* reject_reason = "no_improvement";
      if (!trial.state.cap.legal) {
        reject_reason = "cap";
      } else if (!trial.state.slew.legal) {
        reject_reason = "slew";
      } else if (!trial.state.skew.valid) {
        reject_reason = "timing";
      }
      LOG_INFO << "Optimization: scalable batch trial rejected, reason=" << reject_reason
               << ", candidate_skew=" << (trial.state.skew.valid ? trial.state.skew.skew_ns : 0.0)
               << " ns, candidate_area=" << trial.state.power.area_um2 << " um^2, trial_runtime=" << trial_runtime_s << " s.";
      continue;
    }
    if (PreferTrial(trial, best, current, target_skew_ns)) {
      best = std::move(trial);
    }
    LOG_INFO << "Optimization: scalable batch trial finished, iteration=" << (summary.iteration_count + 1U)
             << ", candidate=" << (candidate_index + 1U) << "/" << candidates.size() << ", total_trials=" << summary.trial_count
             << ", trial_runtime=" << trial_runtime_s << " s, iteration_runtime=" << ElapsedSeconds(iteration_start)
             << " s, best_skew=" << (best.valid ? best.state.skew.skew_ns : current.skew.skew_ns) << " ns.";
  }
  return best;
}

}  // namespace

auto SolveClock(FastSTA& fast_sta, FastStaClockId clock_id, std::vector<ClockSizingBuffer>& buffers,
                const std::vector<ClockSizingCapLimit>& cap_baseline, const std::vector<ClockSizingSlewLimit>& slew_baseline,
                double target_skew_ns) -> ClockSizingSummary
{
  ClockSizingSummary summary;
  summary.solve_mode = "exact_full_power_batch";
  auto stage_start = std::chrono::steady_clock::now();
  summary.before = CaptureState(fast_sta, clock_id, cap_baseline, slew_baseline);
  summary.profile.capture_initial_state_s = ElapsedSeconds(stage_start);
  if (!summary.before.valid) {
    if (!summary.before.skew.valid) {
      summary.stop_reason = "initial_skew_unavailable";
    } else if (!summary.before.cap.legal) {
      summary.stop_reason = "initial_cap_worse_than_baseline";
    } else {
      summary.stop_reason = "initial_slew_worse_than_baseline";
    }
    summary.after = summary.before;
    return summary;
  }

  stage_start = std::chrono::steady_clock::now();
  const auto topology = BuildClockSizingTopologyIndex(ClockSizingTopologyIndexInput{
      .fast_sta = &fast_sta,
      .clock_id = clock_id,
      .buffers = &buffers,
  });
  summary.profile.build_topology_index_s = ElapsedSeconds(stage_start);
  auto current = summary.before;
  if (DefaultOptimizationPolicy().stop_at_first_target_skew_batch && TargetMet(current, target_skew_ns)) {
    summary.after = current;
    summary.valid = true;
    summary.target_met = true;
    summary.stop_reason = "target_met";
    return summary;
  }
  while (summary.iteration_count < DefaultOptimizationPolicy().max_iterations
         && summary.trial_count < DefaultOptimizationPolicy().max_trials) {
    auto best = FindBestClockSizingEditBatch(fast_sta, clock_id, buffers, topology, current, cap_baseline, slew_baseline, target_skew_ns,
                                             summary);
    if (!best.valid) {
      summary.stop_reason = summary.trial_count >= DefaultOptimizationPolicy().max_trials ? "trial_limit" : "no_improving_candidate";
      break;
    }
    stage_start = std::chrono::steady_clock::now();
    if (!ChangeFastStaMasters(fast_sta, clock_id, BuildMasterChanges(buffers, best.edits, false))) {
      summary.stop_reason = "accepted_edit_apply_failed";
      break;
    }
    current = CaptureState(fast_sta, clock_id, cap_baseline, slew_baseline);
    summary.profile.apply_accepted_batch_s += ElapsedSeconds(stage_start);
    for (const auto& edit : best.edits) {
      if (edit.buffer_index >= buffers.size()) {
        continue;
      }
      auto& buffer = buffers.at(edit.buffer_index);
      summary.accepted_edits.push_back(ClockSizingAcceptedEdit{.inst_name = buffer.inst_name,
                                                               .from_master = edit.from_master,
                                                               .to_master = edit.to_master,
                                                               .area_delta_um2 = edit.area_delta_um2});
      buffer.current_master = edit.to_master;
    }
    summary.accepted_edit_count += static_cast<unsigned>(best.edits.size());
    ++summary.accepted_batch_count;
    ++summary.iteration_count;
    LOG_INFO << "Optimization: accepted batch " << summary.accepted_batch_count << ", edit_count=" << best.edits.size()
             << ", skew=" << current.skew.skew_ns << " ns, area=" << current.power.area_um2 << " um^2, total_trials=" << summary.trial_count
             << ".";
    if (DefaultOptimizationPolicy().stop_at_first_target_skew_batch && TargetMet(current, target_skew_ns)) {
      summary.stop_reason = "target_met";
      break;
    }
  }

  summary.after = current;
  summary.valid = summary.before.valid && summary.after.valid;
  summary.changed = !summary.accepted_edits.empty();
  summary.target_met = TargetMet(summary.after, target_skew_ns);
  if (summary.stop_reason.empty()) {
    summary.stop_reason = summary.target_met ? "target_met" : "iteration_limit";
  }
  return summary;
}

auto SolveClockScalable(FastSTA& fast_sta, FastStaClockId clock_id, std::vector<ClockSizingBuffer>& buffers,
                        const std::vector<ClockSizingCapLimit>& cap_baseline, const std::vector<ClockSizingSlewLimit>& slew_baseline,
                        double target_skew_ns) -> ClockSizingSummary
{
  ClockSizingSummary summary;
  summary.solve_mode = "scalable_timing_only_batch";
  auto stage_start = std::chrono::steady_clock::now();
  summary.before = CaptureState(fast_sta, clock_id, cap_baseline, slew_baseline);
  summary.profile.capture_initial_state_s = ElapsedSeconds(stage_start);
  if (!summary.before.valid) {
    if (!summary.before.skew.valid) {
      summary.stop_reason = "initial_skew_unavailable";
    } else if (!summary.before.cap.legal) {
      summary.stop_reason = "initial_cap_worse_than_baseline";
    } else {
      summary.stop_reason = "initial_slew_worse_than_baseline";
    }
    summary.after = summary.before;
    return summary;
  }

  stage_start = std::chrono::steady_clock::now();
  const auto topology = BuildClockSizingTopologyIndex(ClockSizingTopologyIndexInput{
      .fast_sta = &fast_sta,
      .clock_id = clock_id,
      .buffers = &buffers,
  });
  summary.profile.build_topology_index_s = ElapsedSeconds(stage_start);
  auto current = summary.before;
  while (summary.iteration_count < DefaultOptimizationPolicy().max_iterations
         && summary.trial_count < DefaultOptimizationPolicy().max_trials) {
    auto best = FindBestScalableClockSizingEditBatch(fast_sta, clock_id, buffers, topology, current, cap_baseline, slew_baseline,
                                                     target_skew_ns, summary);
    if (!best.valid) {
      summary.stop_reason = summary.trial_count >= DefaultOptimizationPolicy().max_trials ? "trial_limit" : "no_improving_candidate";
      break;
    }

    stage_start = std::chrono::steady_clock::now();
    if (!ChangeFastStaMastersTimingOnly(fast_sta, clock_id, BuildMasterChanges(buffers, best.edits, false))) {
      summary.stop_reason = "accepted_edit_apply_failed";
      break;
    }
    current = CaptureStateWithArea(fast_sta, clock_id, cap_baseline, slew_baseline,
                                   current.power.area_um2 + ClockSizingEditAreaDelta(best.edits));
    summary.profile.apply_accepted_batch_s += ElapsedSeconds(stage_start);
    if (!current.valid) {
      summary.stop_reason = !current.cap.legal ? "accepted_edit_cap_violation" : "accepted_edit_slew_violation";
      break;
    }

    for (const auto& edit : best.edits) {
      if (edit.buffer_index >= buffers.size()) {
        continue;
      }
      auto& buffer = buffers.at(edit.buffer_index);
      summary.accepted_edits.push_back(ClockSizingAcceptedEdit{.inst_name = buffer.inst_name,
                                                               .from_master = edit.from_master,
                                                               .to_master = edit.to_master,
                                                               .area_delta_um2 = edit.area_delta_um2});
      buffer.current_master = edit.to_master;
    }
    summary.accepted_edit_count += static_cast<unsigned>(best.edits.size());
    ++summary.accepted_batch_count;
    ++summary.iteration_count;
    LOG_INFO << "Optimization: accepted scalable batch " << summary.accepted_batch_count << ", edit_count=" << best.edits.size()
             << ", skew=" << current.skew.skew_ns << " ns, tracked_area=" << current.power.area_um2
             << " um^2, total_trials=" << summary.trial_count << ".";
  }

  if (!fast_sta.updatePower(clock_id)) {
    summary.after = current;
    summary.valid = false;
    if (summary.stop_reason.empty()) {
      summary.stop_reason = "final_power_update_failed";
    }
    return summary;
  }

  summary.after = CaptureState(fast_sta, clock_id, cap_baseline, slew_baseline);
  summary.valid = summary.before.valid && summary.after.valid;
  summary.changed = !summary.accepted_edits.empty();
  summary.target_met = TargetMet(summary.after, target_skew_ns);
  if (summary.stop_reason.empty()) {
    summary.stop_reason = summary.target_met ? "target_met" : "iteration_limit";
  }
  return summary;
}

auto ShouldUseScalableSolver(const ScalableSolverDecisionInput& input) -> bool
{
  LOG_FATAL_IF(input.fast_sta == nullptr) << "Optimization: scalable solver decision requires FastSTA.";
  LOG_FATAL_IF(input.buffers == nullptr) << "Optimization: scalable solver decision requires buffers.";
  const auto graph_profile = input.fast_sta->queryClockGraphProfile(input.clock_id);
  if (!graph_profile.has_value()) {
    return false;
  }
  return graph_profile->node_count >= DefaultOptimizationPolicy().scalable_node_threshold
         || input.buffers->size() >= DefaultOptimizationPolicy().scalable_buffer_threshold;
}

}  // namespace icts::clock_sizing_optimization
