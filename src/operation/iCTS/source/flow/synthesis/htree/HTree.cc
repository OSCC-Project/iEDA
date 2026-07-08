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
 * @file HTree.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-14
 * @brief H-tree topology-family synthesis entry implementation.
 */

#include "synthesis/htree/HTree.hh"

#include <glog/logging.h>

#include <ostream>
#include <string>
#include <utility>

#include "Log.hh"
#include "Net.hh"
#include "Pin.hh"
#include "Point.hh"
#include "Tree.hh"
#include "logger/Schema.hh"
#include "synthesis/htree/diagnostic/HTreeDiagnostic.hh"
#include "synthesis/htree/solution/Solution.hh"
#include "synthesis/htree/synthesis_state/SynthesisState.hh"

namespace icts {

namespace {

class HTreeBuilder
{
 public:
  HTreeBuilder(const HTree::Input& input, const HTree::Config& config) : _input(&input), _config(&config) {}

  auto build() -> htree::DiagnosticBuild;

 private:
  const HTree::Input* _input = nullptr;
  const HTree::Config* _config = nullptr;
};

auto ExtractProductionBuild(htree::DiagnosticBuild diagnostic_build) -> HTree::Build
{
  HTree::Build build;
  build.output = std::move(diagnostic_build.output);
  build.summary = std::move(diagnostic_build.summary);
  return build;
}

auto InitializeRootResult(const HTree::Input& input, htree::DiagnosticBuild& result) -> void
{
  LOG_FATAL_IF(input.root_net == nullptr) << "HTree build requires an explicit root net.";
  auto& root_net = *input.root_net;
  result.diagnostics.log_context = input.log_context;
  result.diagnostics.object_name_prefix = input.object_name_prefix;
  result.output.root_net = &root_net;
  result.output.root_output_pin = root_net.get_driver();
  result.output.root_inst = result.output.root_output_pin == nullptr ? nullptr : result.output.root_output_pin->get_inst();
}

}  // namespace

auto HTreeBuilder::build() -> htree::DiagnosticBuild
{
  const auto& input = *_input;
  const auto& config = *_config;
  htree::DiagnosticBuild result;
  InitializeRootResult(input, result);
  auto& root_net = *input.root_net;
  if (result.output.root_output_pin == nullptr) {
    result.summary.failure_reason = "missing_root_driver_pin";
    LOG_WARNING << "HTree: build skipped because root net " << root_net.get_name() << " has no driver pin.";
    return result;
  }

  const auto loads = root_net.get_loads();
  if (loads.empty()) {
    result.summary.failure_reason = "empty_root_net_loads";
    LOG_WARNING << "HTree: build skipped because root net " << root_net.get_name() << " has no loads.";
    return result;
  }
  if (loads.size() == 1U) {
    const auto root = result.output.topology.create_node();
    result.output.topology.set_root(root);
    if (auto* root_node = result.output.topology.get_node(root); root_node != nullptr) {
      root_node->get_position() = input.fixed_topology_root_location.value_or(loads.front()->get_location());
    }
    result.summary.selected_depth = 0U;
    result.diagnostics.root_driver_sizing_enabled = config.enable_root_driver_sizing;
    result.diagnostics.target_depth = config.target_depth;
    result.summary.success = true;
    if (input.reporter != nullptr) {
      auto build_stage = input.reporter->beginStage("HTree", "build", {}, StageReportOptions{.emit_success_summary = false});
      build_stage.finished({
          {"reason", "trivial_single_load"},
          {"selected_depth", "0"},
          {"inserted_insts", "0"},
          {"inserted_nets", "0"},
      });
    }
    return result;
  }

  LOG_FATAL_IF(input.reporter == nullptr) << "HTree build requires an explicit reporter.";
  auto& reporter = *input.reporter;
  auto build_stage = reporter.beginStage("HTree", "build", {}, StageReportOptions{.emit_success_summary = false});

  auto state_build = htree::AssembleHTreeSynthesisState(input, config, build_stage);
  if (state_build.status != htree::HTreeSynthesisStateStatus::kReady) {
    return std::move(state_build.state.result);
  }

  auto selection = htree::SelectHTreeSolution(state_build.state);
  if (selection.selected) {
    htree::FinalizeSelectedHTreeSolution(state_build.state, build_stage, selection.selected_solution);
    return std::move(state_build.state.result);
  }

  auto& state = state_build.state;
  const std::string selection_engine = htree::ToStageValue(selection.engine);
  if (!selection.failure_reason.empty()) {
    state.result.summary.failure_reason = selection.failure_reason;
  } else if (selection.engine == htree::HTreeSelectionEngine::kAnalytical) {
    state.result.summary.failure_reason = "analytical_candidate_unavailable";
  } else {
    state.result.summary.failure_reason = "missing_best_char";
  }
  build_stage.failed({{"reason", state.result.summary.failure_reason},
                      {"depth_candidates", std::to_string(state.depth_candidates.size())},
                      {"selection_engine", selection_engine}});
  return std::move(state.result);
}

auto HTree::build(const Input& input, const Config& config) -> Build
{
  return ExtractProductionBuild(htree::BuildWithDiagnostics(input, config));
}

auto htree::BuildWithDiagnostics(const HTree::Input& input, const HTree::Config& config) -> htree::DiagnosticBuild
{
  HTreeBuilder builder(input, config);
  return builder.build();
}

}  // namespace icts
