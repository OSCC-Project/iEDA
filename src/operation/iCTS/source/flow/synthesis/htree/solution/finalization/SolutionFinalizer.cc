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
 * @file SolutionFinalizer.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-26
 * @brief Shared H-tree selected-solution finalization implementation.
 */

#include "synthesis/htree/solution/finalization/SolutionFinalizer.hh"

#include <glog/logging.h>

#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "HTreeTopologyChar.hh"
#include "HTreeTopologyPattern.hh"
#include "Inst.hh"
#include "Log.hh"
#include "logger/Schema.hh"
#include "synthesis/htree/constraint/Constraint.hh"
#include "synthesis/htree/diagnostic/HTreeDiagnostic.hh"
#include "synthesis/htree/embedding/Embedding.hh"
#include "synthesis/htree/segment_pruning/TopologyPatternLibrary.hh"
#include "synthesis/htree/solution/report/SolutionReport.hh"
#include "synthesis/htree/solution/report/StageReport.hh"
#include "synthesis/htree/solution/selection/SolutionSelection.hh"
#include "synthesis/htree/synthesis_state/SynthesisState.hh"

namespace icts::htree {

auto ToStageValue(HTreeSelectionEngine engine) -> std::string
{
  switch (engine) {
    case HTreeSelectionEngine::kDiscrete:
      return "discrete";
    case HTreeSelectionEngine::kAnalytical:
      return "analytical";
  }
  return "unknown";
}

auto FinalizeSelectedHTreeSolution(HTreeSynthesisState& state, SchemaWriter::StageScope& build_stage,
                                   const HTreeSelectedSolution& selected_solution) -> bool
{
  LOG_FATAL_IF(state.input == nullptr) << "HTree selected-solution finalization requires synthesis input.";
  LOG_FATAL_IF(state.config == nullptr) << "HTree selected-solution finalization requires synthesis config.";
  auto& result = state.result;
  const auto& input = *state.input;
  const auto& config = *state.config;
  auto& segment_pattern_library = state.segmentPatterns();

  LOG_FATAL_IF(input.design == nullptr) << "HTree selected-solution finalization requires explicit Design dependency.";
  LOG_FATAL_IF(input.wrapper == nullptr) << "HTree selected-solution finalization requires explicit Wrapper dependency.";
  LOG_FATAL_IF(input.reporter == nullptr) << "HTree selected-solution finalization requires explicit reporter dependency.";

  auto& design = *input.design;
  auto& wrapper = *input.wrapper;
  auto& reporter = *input.reporter;
  const auto& selected_evaluation = selected_solution.evaluation;
  const auto& selected_summary = selected_solution.summary;
  const std::string selection_engine = ToStageValue(selected_solution.engine);

  if (!selected_evaluation.best_char.has_value()) {
    result.summary.failure_reason = "missing_selected_best_char";
    build_stage.failed({{"reason", result.summary.failure_reason}, {"selection_engine", selection_engine}});
    return false;
  }

  result.summary.selected_depth = selected_evaluation.depth;
  result.output.best_char = *selected_evaluation.best_char;
  ApplyRootDriverCompensationSummary(result, selected_solution.compensation_stats, selected_solution.compensation_detail,
                                     *result.output.best_char);
  result.diagnostics.root_driver_compensation.clock_period_source = selected_solution.root_driver_clock_period_source;
  result.output.levels = selected_evaluation.levels;
  result.diagnostics.selected_final_frontier_count = selected_summary.final_frontier_count;
  result.diagnostics.selected_candidate_solution_count = selected_summary.candidate_solution_count;
  result.diagnostics.selected_candidate_frontier_entry_count = selected_summary.candidate_frontier_entry_count;
  result.diagnostics.selected_feasible_solution_count = selected_summary.feasible_solution_count;
  result.diagnostics.selected_feasible_frontier_entry_count = selected_summary.feasible_frontier_entry_count;
  result.diagnostics.min_top_input_slew_ns = selected_evaluation.boundary_constraints.min_top_input_slew_ns;
  result.diagnostics.top_input_slew_covering_idx = selected_evaluation.boundary_constraints.top_input_slew_covering_idx;
  result.diagnostics.htree_load_group_count = selected_summary.htree_load_group_count;
  result.diagnostics.htree_load_cap_min_pf = selected_summary.htree_load_cap_min_pf;
  result.diagnostics.htree_load_cap_max_pf = selected_summary.htree_load_cap_max_pf;
  result.diagnostics.htree_load_cap_mean_pf = selected_summary.htree_load_cap_mean_pf;
  result.diagnostics.htree_load_cap_median_pf = selected_summary.htree_load_cap_median_pf;

  if (selected_solution.used_boundary_relaxation) {
    result.summary.used_boundary_relaxation = true;
    result.diagnostics.boundary_relaxation_reason = selected_solution.boundary_relaxation_reason;
    result.diagnostics.boundary_relaxation_score = selected_solution.boundary_relaxation_score;
    EmitDiagnostic(reporter, DiagnosticLevel::kWarning, "HTree",
                   "boundary relaxation is enabled; selected a relaxed solution from the global candidate pool.",
                   {
                       {"reason", result.diagnostics.boundary_relaxation_reason},
                       {"selected_depth", std::to_string(result.summary.selected_depth.value_or(0U))},
                       {"relaxation_score", std::to_string(result.diagnostics.boundary_relaxation_score.value_or(0.0))},
                       {"selected_top_input_slew_idx", std::to_string(result.output.best_char->get_input_slew_idx())},
                       {"selected_leaf_load_cap_idx", std::to_string(result.output.best_char->get_leaf_load_cap_idx())},
                       {"selection_engine", selection_engine},
                   });
  }

  result.output.best_pattern = selected_evaluation.topology_pattern_library.materialize(result.output.best_char->get_pattern_id());
  ApplySelectedPatternToLevelPlans(wrapper, result, segment_pattern_library);
  const std::string selected_root_driver_cell_master = ResolveSelectedRootDriverCellMaster(result.output.levels);
  if (config.enable_root_driver_sizing && !ValidateRootDriverSizing(design, wrapper, result, selected_root_driver_cell_master)) {
    result.summary.failure_reason = "root_driver_sizing_precheck_failed";
    build_stage.failed({{"reason", result.summary.failure_reason}, {"selection_engine", selection_engine}});
    return false;
  }

  {
    auto embedding_stage = reporter.beginStage("HTree", "Build selected embedding",
                                               {
                                                   {"selected_depth", std::to_string(result.summary.selected_depth.value_or(0U))},
                                                   {"selected_levels", std::to_string(result.output.levels.size())},
                                                   {"selection_engine", selection_engine},
                                               },
                                               DetailStageReportOptions());
    BuildEmbedding(design, wrapper, result, segment_pattern_library);
    result.summary.success = result.summary.failure_reason.empty() && result.output.best_char.has_value()
                             && result.output.best_pattern.has_value() && result.output.root_output_pin != nullptr
                             && result.output.root_net != nullptr;
    if (result.summary.success && config.enable_root_driver_sizing) {
      LOG_FATAL_IF(!ApplyRootDriverSizing(design, wrapper, result, selected_root_driver_cell_master))
          << "HTree: prevalidated root-driver sizing failed during embedding construction.";
    } else if (result.summary.success && result.output.root_inst != nullptr) {
      result.diagnostics.selected_root_driver_cell_master = result.output.root_inst->get_cell_master();
    }
    if (result.summary.success) {
      embedding_stage.finished({
          {"inserted_insts", std::to_string(result.output.inserted_insts.size())},
          {"inserted_nets", std::to_string(result.output.inserted_nets.size())},
          {"pruned_leaf_single_load_buffers", std::to_string(result.diagnostics.pruned_leaf_single_load_buffers)},
      });
    } else {
      embedding_stage.failed(
          {{"reason", result.summary.failure_reason.empty() ? "incomplete_embedding_build" : result.summary.failure_reason}});
    }
  }

  {
    auto summary_stage = reporter.beginStage("HTree", "Emit synthesis summary", {}, StageReportOptions{.emit_success_summary = false});
    LogSynthesisSummary(reporter, result, selected_evaluation, selected_summary);
    summary_stage.finished();
  }
  if (result.summary.success) {
    build_stage.finished({{"selection_engine", selection_engine}});
  } else {
    build_stage.failed({{"reason", result.summary.failure_reason.empty() ? "incomplete_embedding_build" : result.summary.failure_reason},
                        {"selection_engine", selection_engine}});
  }
  return result.summary.success;
}

}  // namespace icts::htree
