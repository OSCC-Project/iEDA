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
 * @file SolutionReport.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-01
 * @brief Selected H-tree solution report assembly.
 */

#include "synthesis/htree/solution/report/SolutionReport.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <ostream>
#include <ranges>
#include <string>
#include <vector>

#include "HTreeTopologyChar.hh"
#include "Log.hh"
#include "LogFormat.hh"
#include "PatternId.hh"
#include "logger/Schema.hh"
#include "synthesis/htree/HTree.hh"
#include "synthesis/htree/constraint/Constraint.hh"
#include "synthesis/htree/diagnostic/HTreeDiagnostic.hh"
#include "synthesis/htree/plan/DepthPlan.hh"
#include "synthesis/htree/topology_pruning/TopologyPruning.hh"

namespace icts::htree {

namespace {

auto FormatDelayPower(double delay_ns, double power_w) -> std::string
{
  return logformat::FormatWithUnit(delay_ns, "ns") + " / " + logformat::FormatPowerW(power_w);
}

auto FormatRootLoadDetail(const htree::RootDriverCompensationReport& compensation) -> std::string
{
  if (compensation.load_cap_pf <= 0.0) {
    return "physical root-closure load unavailable";
  }

  std::string detail = "source=" + (compensation.load_source.empty() ? std::string{"none"} : compensation.load_source);
  detail += ", route=" + (compensation.route_estimator.empty() ? std::string{"none"} : compensation.route_estimator);
  detail += ", bucket=" + std::to_string(compensation.load_bucket_idx);
  detail += ", source_boundary_bucket=" + std::to_string(compensation.source_boundary_bucket_idx);
  detail += ", source_boundary_branches=" + std::to_string(compensation.source_boundary_branch_count);
  detail += ", terminals=" + std::to_string(compensation.terminal_count);
  detail += ", pin_cap=" + logformat::FormatWithUnit(compensation.terminal_pin_cap_pf, "pF");
  detail += ", wire_cap=" + logformat::FormatWithUnit(compensation.wire_cap_pf, "pF");
  return detail;
}

auto FormatBucketDelta(unsigned physical_bucket_idx, unsigned raw_bucket_idx) -> std::string
{
  const auto delta = physical_bucket_idx >= raw_bucket_idx ? static_cast<std::int64_t>(physical_bucket_idx - raw_bucket_idx)
                                                           : -static_cast<std::int64_t>(raw_bucket_idx - physical_bucket_idx);
  return std::to_string(delta);
}

auto FormatLevelBufferCounts(const std::vector<HTree::LevelPlan>& levels, bool weighted) -> std::string
{
  std::string summary;
  for (std::size_t level_index = 0U; level_index < levels.size(); ++level_index) {
    if (!summary.empty()) {
      summary += ",";
    }
    const auto& level = levels.at(level_index);
    summary += "L" + std::to_string(level_index) + "="
               + std::to_string(weighted ? level.selected_weighted_buffer_count : level.selected_buffer_count);
  }
  return summary.empty() ? "none" : summary;
}

auto FormatLevelBufferAreas(const std::vector<HTree::LevelPlan>& levels, bool weighted) -> std::string
{
  std::string summary;
  for (std::size_t level_index = 0U; level_index < levels.size(); ++level_index) {
    if (!summary.empty()) {
      summary += ",";
    }
    const auto& level = levels.at(level_index);
    const double area_um2 = weighted ? level.selected_weighted_buffer_area_um2 : level.selected_buffer_area_um2;
    summary += "L" + std::to_string(level_index) + "=" + logformat::FormatWithUnit(area_um2, "um^2");
  }
  return summary.empty() ? "none" : summary;
}

auto IsDefaultSynthesisSummaryField(const std::string& field) -> bool
{
  return field == "levels" || field == "depth_candidates" || field == "selected_depth" || field == "selection_engine"
         || field == "analytical_failure_reason" || field == "selected_topology_pattern_id" || field == "selected_level_segment_pattern_ids"
         || field == "selected_terminal_branch_buffered_levels" || field == "final_frontier_count" || field == "inserted_insts"
         || field == "inserted_nets" || field == "pruned_leaf_single_load_buffers" || field == "power" || field == "delay"
         || field == "root_driver_compensation" || field == "root_driver_clock_period" || field == "compensated_htree_metric"
         || field == "selected_root_driver_cell_master" || field == "root_driver_sizing_enabled" || field == "used_boundary_relaxation"
         || field == "boundary_relaxation_score" || field == "htree_load_group_count" || field == "htree_load_cap_min"
         || field == "htree_load_cap_max" || field == "htree_load_cap_mean" || field == "htree_load_cap_median";
}

auto BuildDefaultSynthesisSummaryFields(SchemaWriter& reporter, const logformat::TableRows& synthesis_summary_rows) -> KeyValueFields
{
  KeyValueFields default_fields;
  default_fields.reserve(synthesis_summary_rows.size());
  for (const auto& row : synthesis_summary_rows) {
    if (row.size() < 2U || !IsDefaultSynthesisSummaryField(row.at(0))) {
      continue;
    }
    default_fields.emplace_back(row.at(0), row.at(1));
  }
  default_fields.emplace_back("detail_report", reporter.getDetailPath().empty() ? "unavailable" : reporter.getDetailPath().string());
  return default_fields;
}

}  // namespace

auto LogSynthesisSummary(SchemaWriter& reporter, const htree::DiagnosticBuild& result, const CandidateBuildEvaluation& selected_evaluation,
                         const DepthSummary& selected_summary) -> void
{
  const bool selected_has_boundary_constraints = HasBoundaryConstraints(selected_evaluation.boundary_constraints);
  if (!result.output.best_char.has_value()) {
    LOG_WARNING << "HTree: synthesis summary skipped because no selected topology char is available.";
    return;
  }
  const auto& best_char = *result.output.best_char;
  const auto selected_terminal_branch_buffered_levels = static_cast<std::size_t>(std::ranges::count_if(
      result.output.levels, [](const HTree::LevelPlan& level) -> bool { return level.selected_has_terminal_branch_buffer; }));
  const auto& root_compensation_report = result.diagnostics.root_driver_compensation;
  std::string selected_level_segment_pattern_ids;
  for (const auto& level : result.output.levels) {
    if (!selected_level_segment_pattern_ids.empty()) {
      selected_level_segment_pattern_ids += ",";
    }
    selected_level_segment_pattern_ids += std::to_string(level.segment_pattern_id.local_id);
  }
  std::string selection_engine_detail;
  if (!result.diagnostics.analytical_mode_enabled) {
    selection_engine_detail = "analytical mode disabled";
  } else if (result.diagnostics.analytical_mode_selected) {
    selection_engine_detail = "mathematical analytical optimization produced the embedded H-tree";
  } else {
    selection_engine_detail = "analytical mode was enabled but did not select a candidate";
  }

  std::string selection_policy;
  if (result.diagnostics.analytical_mode_selected) {
    selection_policy = "math_milp_normalized_delay_power";
  } else if (result.summary.used_boundary_relaxation) {
    selection_policy = "global_boundary_relaxation";
  } else {
    selection_policy = "global_frontier_pareto_power_median";
  }

  std::string selection_policy_detail;
  if (result.summary.used_boundary_relaxation) {
    selection_policy_detail
        = "the global strict-feasible pool across all depth candidates is empty; relaxed selection uses the global "
          "candidate post-compensation frontier pool with delay-power Pareto power-median ordering";
  } else if (result.diagnostics.analytical_mode_selected) {
    selection_policy_detail
        = "the affine MILP solves minimum-delay and minimum-power anchors, then selects the normalized delay/power tradeoff solution";
  } else {
    selection_policy_detail
        = "the global feasible post-compensation frontier pool is Pareto filtered and the lower power-ordered median entry is selected";
  }

  reporter.emitSection("### H-Tree Selection");
  logformat::TableRows synthesis_summary_rows = {
      {"clock_name", result.diagnostics.log_context.clock_name.empty() ? "n/a" : result.diagnostics.log_context.clock_name,
       "context for repeated H-tree/top-level sections"},
      {"clock_net_name", result.diagnostics.log_context.clock_net_name.empty() ? "n/a" : result.diagnostics.log_context.clock_net_name,
       "context for repeated H-tree/top-level sections"},
      {"sink_domain", result.diagnostics.log_context.sink_domain.empty() ? "n/a" : result.diagnostics.log_context.sink_domain,
       "context for repeated H-tree/top-level sections"},
      {"stage", result.diagnostics.log_context.stage.empty() ? "n/a" : result.diagnostics.log_context.stage,
       "context for repeated H-tree/top-level sections"},
      {"object_name_prefix", result.diagnostics.object_name_prefix.empty() ? "n/a" : result.diagnostics.object_name_prefix,
       "context for inserted object names"},
      {"levels", std::to_string(result.output.levels.size()), "selected H-tree levels"},
      {"depth_candidates", std::to_string(result.diagnostics.depth_candidate_count), "evaluated descending depth candidates"},
      {"selected_depth", result.summary.selected_depth.has_value() ? std::to_string(*result.summary.selected_depth) : "none",
       "global winner across all evaluated depth candidates"},
      {"selection_engine", result.diagnostics.analytical_mode_selected ? "analytical" : "discrete", selection_engine_detail},
      {"analytical_failure_reason",
       result.diagnostics.analytical_failure_reason.empty() ? "none" : result.diagnostics.analytical_failure_reason,
       result.diagnostics.analytical_mode_enabled ? "analytical failure reason; discrete solver is not used in analytical mode"
                                                  : "analytical mode disabled"},
      {"analytical_model_sets", std::to_string(result.diagnostics.analytical_model_set_count),
       result.diagnostics.analytical_mode_enabled ? "complete F/D/P/W model sets built from segment characterization" : "not evaluated"},
      {"analytical_rejected_fits", std::to_string(result.diagnostics.analytical_rejected_fit_count),
       result.diagnostics.analytical_mode_enabled ? "fit failures while building analytical model catalog" : "not evaluated"},
      {"analytical_candidates", std::to_string(result.diagnostics.analytical_generated_candidate_count),
       result.diagnostics.analytical_mode_enabled ? "materializable analytical candidates generated before discrete validation"
                                                  : "not evaluated"},
      {"analytical_validated_candidates", std::to_string(result.diagnostics.analytical_validated_candidate_count),
       result.diagnostics.analytical_mode_enabled ? "analytical candidates accepted by discrete legality/compensation validation"
                                                  : "not evaluated"},
      {"analytical_validated_pareto_candidates", std::to_string(result.diagnostics.analytical_validated_pareto_count),
       result.diagnostics.analytical_mode_enabled ? "validated analytical candidates remaining on the delay-power Pareto frontier"
                                                  : "not evaluated"},
      {"analytical_selected_pareto_power_rank", std::to_string(result.diagnostics.analytical_selected_pareto_power_rank),
       result.diagnostics.analytical_mode_selected ? "1-based selected rank by power after delay-first analytical Pareto selection"
                                                   : "not selected analytically"},
      {"analytical_validated_delay_range",
       FormatDelayPower(result.diagnostics.analytical_validated_delay_min_ns, result.diagnostics.analytical_validated_power_min_w) + " .. "
           + FormatDelayPower(result.diagnostics.analytical_validated_delay_median_ns,
                              result.diagnostics.analytical_validated_power_median_w)
           + " .. "
           + FormatDelayPower(result.diagnostics.analytical_validated_delay_max_ns, result.diagnostics.analytical_validated_power_max_w),
       result.diagnostics.analytical_mode_enabled
           ? "min / median / max delay with matching marginal power distribution values over validated candidates"
           : "not evaluated"},
      {"analytical_solver_backend",
       result.diagnostics.analytical_solver_backend.empty() ? "none" : result.diagnostics.analytical_solver_backend,
       result.diagnostics.analytical_mode_enabled ? "mathematical analytical solver backend" : "not evaluated"},
      {"analytical_solver_status",
       result.diagnostics.analytical_solver_status.empty() ? "none" : result.diagnostics.analytical_solver_status,
       result.diagnostics.analytical_mode_enabled ? "mathematical analytical solver status" : "not evaluated"},
      {"analytical_solver_model_size",
       std::to_string(result.diagnostics.analytical_solver_variable_count) + " vars / "
           + std::to_string(result.diagnostics.analytical_solver_binary_variable_count) + " binaries / "
           + std::to_string(result.diagnostics.analytical_solver_continuous_variable_count) + " continuous / "
           + std::to_string(result.diagnostics.analytical_solver_constraint_count) + " constraints",
       result.diagnostics.analytical_mode_enabled ? "MILP model size" : "not evaluated"},
      {"analytical_solver_runtime", logformat::FormatWithUnit(result.diagnostics.analytical_solver_wall_time_ms / 1000.0, "s"),
       result.diagnostics.analytical_mode_enabled ? "external solver wall time" : "not evaluated"},
      {"analytical_solver_objective",
       std::to_string(result.diagnostics.analytical_solver_objective_value)
           + " gap=" + std::to_string(result.diagnostics.analytical_solver_optimality_gap),
       result.diagnostics.analytical_mode_enabled ? "normalized delay/power objective and parsed optimality gap" : "not evaluated"},
      {"analytical_solver_anchors",
       FormatDelayPower(result.diagnostics.analytical_solver_min_delay_anchor_ns, result.diagnostics.analytical_solver_min_power_anchor_w),
       result.diagnostics.analytical_mode_enabled ? "minimum delay anchor / minimum power anchor" : "not evaluated"},
      {"analytical_solver_totals",
       FormatDelayPower(result.diagnostics.analytical_solver_total_delay_ns, result.diagnostics.analytical_solver_total_power_w),
       result.diagnostics.analytical_mode_enabled ? "solver-selected total delay / total power before discrete validation"
                                                  : "not evaluated"},
      {"selected_topology_pattern_id", std::to_string(best_char.get_pattern_id().local_id),
       result.summary.used_boundary_relaxation ? "selected relaxed topology pattern from candidate frontier selection entries"
                                               : "selected strict-feasible topology pattern from the global feasible frontier pool"},
      {"selected_level_segment_pattern_ids", selected_level_segment_pattern_ids.empty() ? "none" : selected_level_segment_pattern_ids,
       "root-to-leaf selected segment pattern ids for the chosen topology pattern"},
      {"selected_level_buffer_counts", FormatLevelBufferCounts(result.output.levels, false),
       "unweighted selected buffer count per H-tree level"},
      {"selected_weighted_level_buffer_counts", FormatLevelBufferCounts(result.output.levels, true),
       "selected buffer count multiplied by H-tree level multiplicity; this explains physical H-tree buffer pressure"},
      {"selected_level_buffer_area", FormatLevelBufferAreas(result.output.levels, false),
       "unweighted selected buffer area per H-tree level"},
      {"selected_weighted_level_buffer_area", FormatLevelBufferAreas(result.output.levels, true),
       "selected buffer area multiplied by H-tree level multiplicity"},
      {"selection_policy", selection_policy, selection_policy_detail},
      {"final_frontier_count", std::to_string(selected_summary.final_frontier_count),
       "selected-depth post-compensation frontier size before boundary filtering and sink-load-region legality filtering"},
      {"candidate_solutions", std::to_string(selected_summary.candidate_solution_count),
       selected_has_boundary_constraints ? "selected-depth post-compensation frontier entries after full topology assembly"
                                         : "not assembled on unrestricted builds"},
      {"candidate_frontier_entry_count", std::to_string(selected_summary.candidate_frontier_entry_count),
       selected_has_boundary_constraints ? "selected-depth sink-load-region-legal candidate frontier entries before feasible filtering"
                                         : "not assembled on unrestricted builds"},
      {"feasible_solutions", std::to_string(selected_summary.feasible_solution_count),
       selected_has_boundary_constraints ? "selected-depth strict-feasible post-compensation frontier entries after boundary filtering"
                                         : "same as post-compensation frontier"},
      {"feasible_frontier_entry_count", std::to_string(selected_summary.feasible_frontier_entry_count),
       "selected-depth sink-load-region-legal frontier entries after feasible filtering"},
      {"inserted_insts", std::to_string(result.output.inserted_insts.size()), "built CTS buffer instances"},
      {"inserted_nets", std::to_string(result.output.inserted_nets.size()), "built CTS nets"},
      {"pruned_leaf_single_load_buffers", std::to_string(result.diagnostics.pruned_leaf_single_load_buffers),
       "post-clock-tree object construction redundant leaf buffers removed when a leaf buffer directly drove one external load"},
      {"power", logformat::FormatPowerW(best_char.get_power()), "selected pattern metric (total power)"},
      {"delay", logformat::FormatWithUnit(best_char.get_delay(), "ns"), "selected pattern metric"},
      {"raw_htree_char_metric", FormatDelayPower(root_compensation_report.raw_delay_ns, root_compensation_report.raw_power_w),
       "characterization-only H-tree delay / power before root-driver compensation"},
      {"root_driver_compensation", FormatDelayPower(root_compensation_report.cell_delay_ns, root_compensation_report.cell_power_w),
       "direct Liberty root cell delay / internal+leakage power; root output net switching power is not added"},
      {"root_driver_clock_period", logformat::FormatWithUnit(root_compensation_report.clock_period_ns, "ns"),
       root_compensation_report.clock_period_source.empty() ? "source=unknown" : "source=" + root_compensation_report.clock_period_source},
      {"compensated_htree_metric",
       FormatDelayPower(root_compensation_report.compensated_delay_ns, root_compensation_report.compensated_power_w),
       root_compensation_report.enabled ? "selected H-tree metric after root-driver compensation" : "root driver compensation disabled"},
      {"selected_physical_root_load", logformat::FormatWithUnit(root_compensation_report.load_cap_pf, "pF"),
       FormatRootLoadDetail(root_compensation_report)},
      {"selected_physical_root_source_boundary_load", logformat::FormatWithUnit(root_compensation_report.source_boundary_load_cap_pf, "pF"),
       "physical root closure load projected to one root H-tree source branch; this is the cap bucket matched against raw char"},
      {"root_driver_input_slew", logformat::FormatWithUnit(root_compensation_report.input_slew_ns, "ns"),
       "H-tree root input margin used for the root-driver Liberty lookup"},
      {"root_driver_output_slew", logformat::FormatWithUnit(root_compensation_report.output_slew_ns, "ns"),
       "resolved root-driver output slew at the raw H-tree top boundary"},
      {"raw_char_source_boundary_cap_idx", std::to_string(best_char.get_driven_cap_idx()),
       "raw H-tree char top/source boundary cap bucket"},
      {"physical_root_load_bucket_idx", std::to_string(root_compensation_report.load_bucket_idx),
       "total physical root-closure output load bucket used by root-driver compensation"},
      {"physical_root_source_boundary_bucket_idx", std::to_string(root_compensation_report.source_boundary_bucket_idx),
       "per-root-branch physical source-boundary bucket used to close the raw H-tree top boundary"},
      {"root_cap_bucket_delta", FormatBucketDelta(root_compensation_report.source_boundary_bucket_idx, best_char.get_driven_cap_idx()),
       "physical_root_source_boundary_bucket_idx - raw_char_source_boundary_cap_idx"},
      {"raw_char_top_input_slew_idx", std::to_string(best_char.get_input_slew_idx()), "raw H-tree char top/source input slew bucket"},
      {"root_output_slew_bucket_idx", std::to_string(root_compensation_report.output_slew_bucket_idx),
       "root-driver output slew bucket used to close the raw H-tree top boundary"},
      {"root_slew_bucket_delta", FormatBucketDelta(root_compensation_report.output_slew_bucket_idx, best_char.get_input_slew_idx()),
       "root_output_slew_bucket_idx - raw_char_top_input_slew_idx"},
      {"leaf_load_cap_idx", std::to_string(best_char.get_leaf_load_cap_idx()), "selected pattern metric"},
      {"leaf_output_slew_idx", std::to_string(best_char.get_output_slew_idx()), "selected pattern metric"},
      {"raw_char_downstream_load_cap_idx", std::to_string(best_char.get_load_cap_idx()), "raw H-tree downstream/leaf-side load cap bucket"},
      {"selected_terminal_branch_buffered_levels",
       std::to_string(selected_terminal_branch_buffered_levels) + "/" + std::to_string(result.output.levels.size()),
       "actual selected H-tree levels whose segment pattern includes a terminal branch buffer"},
      {"raw_top_input_slew_constraint_idx",
       selected_evaluation.boundary_constraints.top_input_slew_covering_idx.has_value()
           ? std::to_string(*selected_evaluation.boundary_constraints.top_input_slew_covering_idx)
           : "none",
       selected_evaluation.boundary_constraints.min_top_input_slew_ns.has_value()
           ? logformat::FormatWithUnit(*selected_evaluation.boundary_constraints.min_top_input_slew_ns, "ns")
           : "unconstrained"},
      {"htree_load_group_count", std::to_string(selected_summary.htree_load_group_count),
       "selected H-tree external-load groups driven by the bottom-most buffered segments"},
      {"htree_load_cap_min", logformat::FormatWithUnit(selected_summary.htree_load_cap_min_pf, "pF"),
       "selected H-tree external-load total-cap minimum across real driven-load groups"},
      {"htree_load_cap_max", logformat::FormatWithUnit(selected_summary.htree_load_cap_max_pf, "pF"),
       "selected H-tree external-load total-cap maximum across real driven-load groups"},
      {"htree_load_cap_mean", logformat::FormatWithUnit(selected_summary.htree_load_cap_mean_pf, "pF"),
       "selected H-tree external-load total-cap mean across real driven-load groups"},
      {"htree_load_cap_median", logformat::FormatWithUnit(selected_summary.htree_load_cap_median_pf, "pF"),
       "selected H-tree external-load total-cap median across real driven-load groups"},
      {"selected_root_driver_cell_master",
       result.diagnostics.selected_root_driver_cell_master.empty() ? "none" : result.diagnostics.selected_root_driver_cell_master,
       "root driver master applied to the input root-net driver inst"},
      {"root_driver_sizing_enabled", logformat::FormatBool(result.diagnostics.root_driver_sizing_enabled),
       result.diagnostics.root_driver_sizing_enabled ? "root driver may be resized when the root is a CTS-owned buffer"
                                                     : "root driver sizing is disabled for immutable top-level clock source"},
      {"used_boundary_relaxation", logformat::FormatBool(result.summary.used_boundary_relaxation),
       result.summary.used_boundary_relaxation ? result.diagnostics.boundary_relaxation_reason
                                               : "constraints satisfied without relaxation"},
      {"boundary_relaxation_score",
       result.diagnostics.boundary_relaxation_score.has_value() ? std::to_string(*result.diagnostics.boundary_relaxation_score) : "none",
       result.summary.used_boundary_relaxation ? "diagnostic normalized active-boundary score of the selected relaxed candidate"
                                               : "not used"},
  };
  {
    const auto is_duplicate_frontier_row = [&](const auto& row) -> bool {
      if (row.empty()) {
        return false;
      }
      const auto& field = row.front();
      if (!selected_has_boundary_constraints
          && (field == "candidate_solutions" || field == "candidate_frontier_entry_count" || field == "feasible_frontier_entry_count")) {
        return true;
      }
      if (!selected_has_boundary_constraints && field == "feasible_solutions"
          && selected_summary.feasible_solution_count == selected_summary.final_frontier_count) {
        return true;
      }
      if (field == "candidate_solutions" && selected_summary.candidate_solution_count == selected_summary.final_frontier_count) {
        return true;
      }
      if (field == "candidate_frontier_entry_count"
          && selected_summary.candidate_frontier_entry_count == selected_summary.candidate_solution_count) {
        return true;
      }
      if (field == "feasible_frontier_entry_count"
          && selected_summary.feasible_frontier_entry_count == selected_summary.feasible_solution_count) {
        return true;
      }
      return false;
    };
    auto duplicate_rows = std::ranges::remove_if(synthesis_summary_rows, is_duplicate_frontier_row);
    synthesis_summary_rows.erase(duplicate_rows.begin(), duplicate_rows.end());
  }
  reporter.emitKeyValueTableTo("HTree Synthesis Overview", BuildDefaultSynthesisSummaryFields(reporter, synthesis_summary_rows),
                               ReportSink::kDefault);
  reporter.emitTableTo("HTree Synthesis Detail", {"Item", "Value", "Detail"}, synthesis_summary_rows, ReportSink::kDetail);
}

}  // namespace icts::htree
