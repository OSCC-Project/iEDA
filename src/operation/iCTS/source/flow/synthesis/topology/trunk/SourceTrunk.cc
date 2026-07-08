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
 * @file SourceTrunk.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief Implements Topology source-to-root segment or HTree dispatch.
 */

#include "synthesis/topology/trunk/SourceTrunk.hh"

#include <glog/logging.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "Log.hh"
#include "Pin.hh"
#include "Point.hh"
#include "characterization/Characterization.hh"
#include "config/Config.hh"
#include "geometry/Geometry.hh"
#include "io/Wrapper.hh"
#include "logger/Schema.hh"
#include "synthesis/htree/HTree.hh"
#include "synthesis/htree/characterization/library/CharacterizationLibrary.hh"
#include "synthesis/topology/buffer/BufferInsertion.hh"
#include "synthesis/topology/trunk/SourceTrunkSegment.hh"
#include "synthesis/trace/topology_build/TopologyBuildTrace.hh"

namespace icts {
class Net;
}  // namespace icts

namespace icts::topology {

namespace {

auto ResolveSourceDriveCap(const SourceTrunkInput& input, Pin* clock_source) -> double
{
  const auto& config = *input.config;
  auto& wrapper = *input.wrapper;
  return wrapper.queryClockSourceDriveCapLimit(config, clock_source);
}

auto ResolveRoutingLayer(const Config& config) -> int
{
  const auto& routing_layers = config.get_routing_layers();
  LOG_FATAL_IF(routing_layers.empty() || routing_layers.front() == 0U) << "Topology: routing layer must be configured for HTree.";
  return static_cast<int>(routing_layers.front());
}

auto ResolveWireWidth(const Config& config) -> std::optional<double>
{
  const double wire_width_um = config.get_wire_width();
  return wire_width_um > 0.0 ? std::optional<double>(wire_width_um) : std::nullopt;
}

auto ApplyMinTopInputSlew(const Config& config, HTree::Config& htree_config) -> void
{
  htree_config.min_top_input_slew_ns = config.get_root_input_slew();
}

auto BuildTopSegmentConfig(const Config& config) -> SourceTrunkSegment::Config
{
  return SourceTrunkSegment::Config{
      .min_input_slew_ns = config.get_root_input_slew(),
  };
}

auto BuildTopSegmentInput(const SourceTrunkInput& input, Pin* clock_source, Pin* root_input) -> SourceTrunkSegment::Input
{
  const auto& config = *input.config;
  auto& wrapper = *input.wrapper;
  auto& fast_sta = *input.fast_sta;
  auto& reporter = *input.reporter;
  HTree::LogContext log_context = input.log_context;
  log_context.stage = "top_segment";
  SourceTrunkSegment::Input segment_input{
      .source_net = input.source_net,
      .source = clock_source,
      .sink = root_input,
      .characterization_library = input.characterization_library,
      .wrapper = &wrapper,
      .characterization_input = {},
      .characterization_config = {},
      .reporter = &reporter,
      .dbu_per_um = 0,
      .required_load_cap_pf = 0.0,
      .source_drive_cap_pf = 0.0,
      .object_name_prefix = input.object_name_prefix,
      .log_context = log_context,
  };
  const int distance_dbu = geometry::Manhattan(clock_source->get_location(), root_input->get_location());
  if (distance_dbu <= 0) {
    return segment_input;
  }

  segment_input.characterization_input = CharacterizationLibrary::buildRuntimeInput(CharacterizationRuntimeInput{
      .config = &config,
      .wrapper = &wrapper,
      .fast_sta = &fast_sta,
      .reporter = &reporter,
  });
  segment_input.characterization_config = CharacterizationLibrary::buildRuntimeConfig(config);
  segment_input.dbu_per_um = wrapper.queryDbUnit();
  segment_input.required_load_cap_pf = wrapper.queryPinCapacitance(root_input);
  segment_input.source_drive_cap_pf = ResolveSourceDriveCap(input, clock_source);
  return segment_input;
}

auto BuildTopHtreeInput(const SourceTrunkInput& input, Net& source_net, Pin* clock_source) -> HTree::Input
{
  const auto& config = *input.config;
  auto& design = *input.design;
  auto& wrapper = *input.wrapper;
  auto& fast_sta = *input.fast_sta;
  auto& reporter = *input.reporter;
  HTree::LogContext log_context = input.log_context;
  log_context.stage = "top_htree";
  return HTree::Input{
      .root_net = &source_net,
      .design = &design,
      .wrapper = &wrapper,
      .reporter = &reporter,
      .characterization_library = input.characterization_library,
      .characterization_input = CharacterizationLibrary::buildRuntimeInput(CharacterizationRuntimeInput{
          .config = &config,
          .wrapper = &wrapper,
          .fast_sta = &fast_sta,
          .reporter = &reporter,
      }),
      .characterization_config = CharacterizationLibrary::buildRuntimeConfig(config),
      .additional_characterization_lengths_um = {},
      .fixed_topology_root_location = FindRenderableLocation(clock_source),
      .clock_period_ns = input.clock_period_ns,
      .clock_period_source = input.clock_period_source,
      .log_context = log_context,
      .object_name_prefix = input.object_name_prefix,
      .load_role = HTree::LoadRole::kSink,
  };
}

auto BuildTopHtreeConfig(const Config& config) -> HTree::Config
{
  HTree::Config htree_config{
      .force_branch_buffer = config.is_force_branch_buffer(),
      .depth_explore_window = std::max(1U, config.get_htree_depth_explore_window()),
      .topology_tolerance = config.get_htree_topology_tolerance(),
      .max_fanout = config.get_max_fanout(),
      .has_max_cap = config.has_max_cap(),
      .max_cap_pf = config.has_max_cap() ? config.get_max_cap() : 0.0,
      .enable_root_driver_sizing = false,
      .allow_boundary_relaxation = false,
      .enable_analytical_solver = config.is_enable_analytical_htree(),
      .routing_layer = ResolveRoutingLayer(config),
      .wire_width_um = ResolveWireWidth(config),
  };
  ApplyMinTopInputSlew(config, htree_config);
  return htree_config;
}

auto DetailStageReportOptions() -> StageReportOptions
{
  return StageReportOptions{.context_sink = ReportSink::kDetail, .summary_sink = ReportSink::kDetail};
}

}  // namespace

auto BuildSourceTrunkTree(const SourceTrunkInput& input) -> SourceTrunkBuild
{
  LOG_FATAL_IF(input.config == nullptr) << "Topology source trunk build requires an explicit config.";
  LOG_FATAL_IF(input.design == nullptr) << "Topology source trunk build requires an explicit design.";
  LOG_FATAL_IF(input.wrapper == nullptr) << "Topology source trunk build requires an explicit wrapper.";
  LOG_FATAL_IF(input.fast_sta == nullptr) << "Topology source trunk build requires an explicit FastSTA.";
  LOG_FATAL_IF(input.reporter == nullptr) << "Topology source trunk build requires an explicit reporter.";
  LOG_FATAL_IF(input.source_net == nullptr) << "Topology source trunk build requires an explicit source net.";
  const auto& flow_config = *input.config;
  auto& reporter = *input.reporter;
  auto& source_net = *input.source_net;
  auto* clock_source = input.clock_source;
  const auto& root_inputs = input.root_inputs;

  SourceTrunkBuild result;
  if (clock_source == nullptr) {
    result.summary.failure_reason = "clock_source_is_null";
    LOG_ERROR << "Topology: top-level source-to-root synthesis failed because clock source is null.";
    return result;
  }

  std::vector<Pin*> valid_root_inputs;
  valid_root_inputs.reserve(root_inputs.size());
  for (auto* root_input : root_inputs) {
    if (root_input != nullptr) {
      valid_root_inputs.push_back(root_input);
    }
  }
  if (valid_root_inputs.empty()) {
    result.summary.failure_reason = "empty_root_inputs";
    LOG_ERROR << "Topology: top-level source-to-root synthesis failed because no root inputs are available.";
    return result;
  }

  SourceNetSideEffectGuard source_net_side_effects(source_net, clock_source, valid_root_inputs);
  ReconnectExistingNet(TopologyNetConnectionInput{
      .net = &source_net,
      .driver = clock_source,
      .sinks = valid_root_inputs,
  });
  auto dispatch_stage = reporter.beginStage("SourceTrunk", "Dispatch source trunk synthesis",
                                            {
                                                {"root_inputs", std::to_string(valid_root_inputs.size())},
                                                {"dispatch", valid_root_inputs.size() == 1U ? "top_segment" : "top_htree"},
                                            },
                                            DetailStageReportOptions());
  if (valid_root_inputs.size() == 1U) {
    result.summary.stage = SourceTrunkStage::kSegment;
    auto segment_input = BuildTopSegmentInput(input, clock_source, valid_root_inputs.front());
    auto segment_config = BuildTopSegmentConfig(flow_config);
    auto segment_build = SourceTrunkSegment::build(segment_input, segment_config);
    if (!segment_build.summary.success) {
      result.summary.failure_reason
          = segment_build.summary.failure_reason.empty() ? "top_segment_failed" : segment_build.summary.failure_reason;
      result.summary.used_boundary_relaxation = segment_build.summary.used_boundary_relaxation;
      source_net_side_effects.restore();
      dispatch_stage.failed({{"reason", result.summary.failure_reason}});
      return result;
    }
    RecordTopSegmentBuild(result, segment_build);
    dispatch_stage.finished({
        {"stage", ToString(result.summary.stage)},
        {"inserted_insts", std::to_string(result.output.inserted_insts.size())},
        {"inserted_nets", std::to_string(result.output.inserted_nets.size())},
        {"used_boundary_relaxation", result.summary.used_boundary_relaxation ? "true" : "false"},
    });
    return result;
  }

  result.summary.stage = SourceTrunkStage::kHTree;
  auto htree_input = BuildTopHtreeInput(input, source_net, clock_source);
  auto htree_config = BuildTopHtreeConfig(flow_config);
  auto htree_build = HTree::build(htree_input, htree_config);
  if (!htree_build.summary.success) {
    result.summary.failure_reason = htree_build.summary.failure_reason.empty() ? "top_htree_failed" : htree_build.summary.failure_reason;
    source_net_side_effects.restore();
    dispatch_stage.failed({{"reason", result.summary.failure_reason}});
    return result;
  }

  const auto selected_depth = htree_build.summary.selected_depth.value_or(0U);
  RecordTopHtreeBuild(result, std::move(htree_build));
  dispatch_stage.finished({
      {"stage", ToString(result.summary.stage)},
      {"selected_depth", std::to_string(selected_depth)},
      {"inserted_insts", std::to_string(result.output.inserted_insts.size())},
      {"inserted_nets", std::to_string(result.output.inserted_nets.size())},
      {"used_boundary_relaxation", result.summary.used_boundary_relaxation ? "true" : "false"},
  });
  return result;
}

}  // namespace icts::topology
