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
 * @file SynthesisState.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-27
 * @brief Shared H-tree synthesis-state assembly implementation.
 */

#include "synthesis/htree/synthesis_state/SynthesisState.hh"

#include <glog/logging.h>
#include <stdint.h>

#include <algorithm>
#include <limits>
#include <ostream>
#include <string>
#include <utility>

#include "BufferingPattern.hh"
#include "ClockRouteSegmentRc.hh"
#include "Inst.hh"
#include "Log.hh"
#include "Net.hh"
#include "Pin.hh"
#include "Point.hh"
#include "TopologyConfig.hh"
#include "TopologyGen.hh"
#include "Tree.hh"
#include "characterization/Characterization.hh"
#include "io/Wrapper.hh"
#include "logger/Schema.hh"
#include "synthesis/htree/characterization/Characterization.hh"
#include "synthesis/htree/plan/Plan.hh"

namespace icts::htree {
namespace {

auto InitializeRootResult(const HTree::Input& input, DiagnosticBuild& result) -> void
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

auto HTreeSynthesisState::charLibrary() -> CharacterizationLibrary&
{
  LOG_FATAL_IF(input == nullptr) << "HTree synthesis state is missing input.";
  if (input->characterization_library != nullptr) {
    return *input->characterization_library;
  }
  LOG_FATAL_IF(!local_char_library.has_value()) << "HTree synthesis state is missing local characterization library.";
  return *local_char_library;
}

auto HTreeSynthesisState::charBuilder() const -> const CharBuilder&
{
  LOG_FATAL_IF(input == nullptr) << "HTree synthesis state is missing input.";
  const auto* char_library = input->characterization_library;
  if (char_library == nullptr) {
    LOG_FATAL_IF(!local_char_library.has_value()) << "HTree synthesis state is missing local characterization library.";
    char_library = &*local_char_library;
  }
  return char_library->getCharBuilder();
}

auto HTreeSynthesisState::segmentPatterns() -> BufferPatternLibrary&
{
  LOG_FATAL_IF(!segment_pattern_library.has_value()) << "HTree synthesis state is missing segment pattern library.";
  return *segment_pattern_library;
}

auto AssembleHTreeSynthesisState(const HTree::Input& input, const HTree::Config& config, SchemaWriter::StageScope& build_stage)
    -> HTreeSynthesisStateBuild
{
  HTreeSynthesisStateBuild state_build;
  auto& state = state_build.state;
  state.input = &input;
  state.config = &config;
  InitializeRootResult(input, state.result);

  LOG_FATAL_IF(input.design == nullptr) << "HTree build requires an explicit design.";
  LOG_FATAL_IF(input.wrapper == nullptr) << "HTree build requires an explicit Wrapper.";
  LOG_FATAL_IF(input.reporter == nullptr) << "HTree build requires an explicit reporter.";
  LOG_FATAL_IF(input.characterization_input.fast_sta == nullptr) << "HTree build requires explicit FastSTA characterization context.";
  LOG_FATAL_IF(!input.characterization_input.dbu_per_um.has_value() || *input.characterization_input.dbu_per_um <= 0)
      << "HTree build requires explicit positive DBU-per-micron input.";

  auto& reporter = *input.reporter;
  auto& root_net = *input.root_net;
  const auto loads = root_net.get_loads();
  const int32_t dbu_per_um = *input.characterization_input.dbu_per_um;
  LOG_FATAL_IF(dbu_per_um <= 0) << "HTree: build failed because DBU-per-micron is unavailable.";

  BiPartitionConfig topology_config;
  topology_config.htree_topology_tolerance = std::max(0.0, config.topology_tolerance);
  topology_config.max_leaf_load_count = config.max_fanout;
  state.result.output.topology = TopologyGen::build(
      loads,
      TopologyGen::Input{
          .fixed_root_location = input.fixed_topology_root_location,
          .dbu_per_um = dbu_per_um,
          .load_count_kind
          = input.load_role == HTree::LoadRole::kLocalBuffer ? TopologyGen::LoadCountKind::kLocalBuffer : TopologyGen::LoadCountKind::kSink,
          .clock_name = input.log_context.clock_name,
          .clock_net_name = input.log_context.clock_net_name,
          .sink_domain = input.log_context.sink_domain,
          .stage = input.log_context.stage,
          .reporter = &reporter,
      },
      TopologyGen::Config{.partition_config = topology_config});
  const auto levels = state.result.output.topology.levels();
  if (levels.size() <= 1U) {
    LOG_WARNING << "HTree: topology has no H-tree levels after generation.";
    state_build.failure_reason = "no_h_tree_levels";
    build_stage.skip({{"reason", state_build.failure_reason}});
    return state_build;
  }

  build_stage.markRunning("characterization");
  if (input.characterization_library == nullptr) {
    state.local_char_library.emplace();
  }
  auto& char_library = state.charLibrary();
  const auto char_flow = RunCharacterizationFlow(state.result.output.topology, dbu_per_um, input.characterization_input,
                                                 input.characterization_config, state.result, char_library, input, config);
  if (!char_flow.success) {
    state_build.failure_reason = char_flow.failure_reason;
    build_stage.failed({{"reason", char_flow.failure_reason}});
    return state_build;
  }
  state.char_length_step_um = char_flow.length_step_um;

  const auto& char_builder = state.charBuilder();
  state.base_boundary_constraints = ResolveBoundaryConstraints(config, char_builder);
  state.result.diagnostics.force_branch_buffer = state.base_boundary_constraints.force_branch_buffer;
  state.result.diagnostics.root_driver_sizing_enabled = config.enable_root_driver_sizing;
  state.result.diagnostics.target_depth = config.target_depth;
  state.strict_root_boundary_closure = config.enable_root_driver_sizing;
  state.search_boundary_constraints
      = ResolvePatternSearchBoundaryConstraints(state.base_boundary_constraints, state.strict_root_boundary_closure);

  state.full_level_plans = BuildLevelPlans(state.result.output.topology, char_flow.length_step_um, dbu_per_um);
  if (state.full_level_plans.empty()) {
    LOG_WARNING << "HTree: failed to derive H-tree level plans from topology.";
    state_build.failure_reason = "empty_level_plans";
    build_stage.failed({{"reason", state_build.failure_reason}});
    return state_build;
  }

  state.max_depth = static_cast<unsigned>(state.full_level_plans.size());
  state.depth_candidates = ResolveDepthCandidates(state.max_depth, config);
  if (state.depth_candidates.empty()) {
    LOG_WARNING << "HTree: no depth candidates were resolved from topology.";
    state_build.failure_reason = "empty_depth_candidates";
    build_stage.failed({{"reason", state_build.failure_reason}});
    return state_build;
  }
  state.result.diagnostics.depth_explore_window = static_cast<unsigned>(state.depth_candidates.size());

  state.segment_pattern_library.emplace(*input.wrapper);
  for (const auto& pattern : char_builder.get_buffering_patterns()) {
    state.segmentPatterns().add(pattern);
  }

  const auto [root_driver_clock_period_ns, root_driver_clock_period_source] = ResolveRootDriverClockPeriod(input);
  state.root_driver_clock_period_source = root_driver_clock_period_source;
  state.root_driver_compensation_input = RootDriverCompensationInput{
      .enabled = config.enable_root_driver_sizing,
      .wrapper = input.wrapper,
      .input_slew_ns = ResolveRootDriverCompensationInputSlewNs(config, char_builder.get_max_slew()),
      .clock_period_ns = root_driver_clock_period_ns,
      .cap_lattice = char_builder.get_cap_lattice(),
      .slew_lattice = char_builder.get_slew_lattice(),
      .default_cell_master = state.result.output.root_inst != nullptr ? state.result.output.root_inst->get_cell_master() : "",
      .routing_layer = config.routing_layer,
      .wire_width_um = config.wire_width_um,
      .dbu_per_um = dbu_per_um,
      .reporter = &reporter,
      .strict_boundary_closure = state.strict_root_boundary_closure,
  };
  state.fanout_pruning_config = HTreeFanoutPruningConfig{
      .max_fanout = config.max_fanout,
      .allow_boundary_relaxation = config.allow_boundary_relaxation,
  };
  state.sink_load_region_input = SinkLoadRegionLegalityInput{
      .wrapper = input.wrapper,
      .max_fanout = config.max_fanout,
      .has_max_cap = config.has_max_cap,
      .max_cap_pf = config.has_max_cap ? config.max_cap_pf : std::numeric_limits<double>::infinity(),
      .clock_route_segment_rc = char_builder.get_clock_route_segment_rc(),
  };
  state.result.diagnostics.analytical_mode_enabled = config.enable_analytical_solver;

  state_build.status = HTreeSynthesisStateStatus::kReady;
  return state_build;
}

}  // namespace icts::htree
