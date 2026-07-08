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
 * @file Topology.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-01
 * @brief CTS topology formation entry for sink branches and source trunk.
 */

#include "synthesis/topology/Topology.hh"

#include <glog/logging.h>

#include <algorithm>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "Log.hh"
#include "config/Config.hh"
#include "design/Clock.hh"
#include "design/ClockLayout.hh"
#include "design/Design.hh"
#include "design/Net.hh"
#include "design/Pin.hh"
#include "geometry/Geometry.hh"
#include "io/Wrapper.hh"
#include "logger/Schema.hh"
#include "synthesis/distribution/ClockDistribution.hh"
#include "synthesis/realization/ClockTreeRealization.hh"
#include "synthesis/topology/SourceTrunkStage.hh"
#include "synthesis/topology/sink/SinkBranch.hh"
#include "synthesis/topology/trunk/SourceTrunk.hh"
#include "synthesis/trace/SynthesisTrace.hh"
#include "synthesis/trace/domain_status/DomainStatus.hh"
#include "synthesis/trace/layout/ClockLayoutAdapter.hh"
#include "synthesis/trace/layout/ClockLayoutBuilder.hh"

namespace icts {
namespace {

constexpr std::size_t kMinSynthesisSinkCount = 2U;

auto recordSynthesisBuild(SynthesisTraceSummary& summary, const Topology::Build& build) -> void
{
  summary.selected_htree_level_count = std::max(summary.selected_htree_level_count, build.summary.selected_htree_level_count);
  if (build.summary.selected_htree_depth.has_value()) {
    summary.selected_htree_depth = std::max(summary.selected_htree_depth, *build.summary.selected_htree_depth);
  }
  summary.htree_inserted_buffer_count += build.summary.htree_inserted_buffer_count;
  summary.htree_inserted_net_count += build.summary.htree_inserted_net_count;
}

auto recordSourceTrunkBuild(SynthesisTraceSummary& summary, const topology::SourceTrunkBuild& build) -> void
{
  if (build.summary.selected_depth.has_value()) {
    summary.selected_htree_depth = std::max(summary.selected_htree_depth, *build.summary.selected_depth);
  }
  summary.selected_htree_level_count = std::max(summary.selected_htree_level_count, build.summary.selected_level_count);
  summary.htree_inserted_buffer_count += build.summary.inserted_buffer_count;
  summary.htree_inserted_net_count += build.summary.inserted_net_count;
}

auto sourceTrunkSynthesisPhase(SourceTrunkStage stage) -> ClockLayoutPhase
{
  switch (stage) {
    case SourceTrunkStage::kSegment:
      return ClockLayoutPhase::kSourceToRootSegment;
    case SourceTrunkStage::kHTree:
      return ClockLayoutPhase::kSourceToRootHTree;
    case SourceTrunkStage::kUnknown:
      return ClockLayoutPhase::kUnknown;
  }
  return ClockLayoutPhase::kUnknown;
}

auto makeLogContext(const Clock& clock, const std::string& sink_domain, const std::string& stage, const std::string& object_name_prefix)
    -> HTree::LogContext
{
  return HTree::LogContext{
      .clock_name = clock.get_clock_name(),
      .clock_net_name = clock.get_clock_net_name(),
      .sink_domain = sink_domain,
      .stage = stage,
      .object_name_prefix = object_name_prefix,
  };
}

auto DetailStageReportOptions() -> StageReportOptions
{
  return StageReportOptions{.context_sink = ReportSink::kDetail, .summary_sink = ReportSink::kDetail};
}

auto clearClockCtsMembership(Design& design, Clock& clock) -> void
{
  design.removeClockMembershipObjects(clock);
  clock.clearMembership();
}

auto collectRootInputs(const std::vector<ClockDistributionContext>& sink_domains) -> std::vector<Pin*>
{
  std::vector<Pin*> root_inputs;
  root_inputs.reserve(sink_domains.size());
  for (const auto& context : sink_domains) {
    if (context.root_input != nullptr) {
      root_inputs.push_back(context.root_input);
    }
  }
  return root_inputs;
}

auto collectSourceTrunkLengthsUm(Wrapper& wrapper, Pin* clock_source, const std::vector<Pin*>& root_inputs) -> std::vector<double>
{
  std::vector<double> lengths_um;
  if (clock_source == nullptr || root_inputs.empty()) {
    return lengths_um;
  }

  const auto dbu_per_um_value = wrapper.queryDbUnit();
  LOG_FATAL_IF(dbu_per_um_value <= 0) << "Topology: DBU-per-micron is unavailable for characterization length estimation.";
  const auto dbu_per_um = static_cast<double>(dbu_per_um_value);
  lengths_um.reserve(root_inputs.size());
  for (const auto* root_input : root_inputs) {
    if (root_input == nullptr) {
      continue;
    }
    const int distance_dbu = geometry::Manhattan(clock_source->get_location(), root_input->get_location());
    const double length_um = static_cast<double>(std::max(distance_dbu, 0)) / dbu_per_um;
    if (length_um > 0.0) {
      lengths_um.push_back(length_um);
    }
  }
  return lengths_um;
}

class ClockTopologySynthesis
{
 public:
  explicit ClockTopologySynthesis(const ClockTopologyInput& input)
      : _config(input.config),
        _design(input.design),
        _wrapper(input.wrapper),
        _fast_sta(input.fast_sta),
        _reporter(input.reporter),
        _clock(input.clock),
        _clock_index(input.clock_index),
        _clock_layout(input.clock_layout),
        _summary(input.summary),
        _status_table(input.status_table),
        _characterization_library(input.characterization_library),
        _valid_sinks(input.valid_sinks),
        _sink_domains(input.sink_domains)
  {
  }

  auto synthesize() -> bool
  {
    auto root_inputs = collectRootInputs(*_sink_domains);
    const auto source_trunk_lengths_um = collectSourceTrunkLengthsUm(*_wrapper, _clock->get_clock_source(), root_inputs);
    for (const auto& context : *_sink_domains) {
      if (!buildAndCommitSinkDomain(context, source_trunk_lengths_um)) {
        return false;
      }
    }
    return buildAndCommitSourceTrunk(root_inputs);
  }

 private:
  auto commitSinkDomainBuild(const ClockDistributionContext& context, Topology::Build& synthesis_build, std::string& failure_reason) -> bool
  {
    auto commit_stage = _reporter->beginStage("Topology", "Commit sink domain layout",
                                              {
                                                  {"sink_domain", ToString(context.sink_domain)},
                                                  {"inserted_insts", std::to_string(synthesis_build.output.inserted_insts.size())},
                                                  {"inserted_nets", std::to_string(synthesis_build.output.inserted_nets.size())},
                                              },
                                              StageReportOptions{.context_sink = ReportSink::kDetail, .emit_success_summary = false});
    auto pending_clock_layout = ClockLayoutBuilder::makeSinkDomainLayout(*_clock, _clock_index, context.makeLayoutTopology(),
                                                                         ClockLayoutAdapter::makeSinkDomainLayoutTopology(synthesis_build));
    if (!ClockTreeRealization::commitInsertedObjects(InsertedObjectCommitInput{
            .design = _design,
            .clock = _clock,
            .inserted_insts = &synthesis_build.output.inserted_insts,
            .inserted_pins = &synthesis_build.output.inserted_pins,
            .inserted_nets = &synthesis_build.output.inserted_nets,
        })) {
      ClockTreeRealization::reconnectNet(NetConnectionInput{
          .net = context.downstream_net,
          .driver = context.downstream_net->get_driver(),
          .loads = context.sinks,
      });
      failure_reason = "failed to commit inserted synthesis objects";
      Topology::resetClockTopology(*_design, *_clock);
      commit_stage.failed({{"reason", failure_reason}});
      return false;
    }

    ClockLayoutBuilder::merge(*_clock_layout, pending_clock_layout);
    recordSynthesisBuild(*_summary, synthesis_build);
    commit_stage.finished();
    return true;
  }

  auto buildAndCommitSinkDomain(const ClockDistributionContext& context, const std::vector<double>& source_trunk_lengths_um) -> bool
  {
    const auto* const sink_domain_label = ToString(context.sink_domain);
    if (context.sinks.size() < kMinSynthesisSinkCount) {
      ClockLayoutBuilder::appendDirectSinkDomain(*_clock_layout, *_clock, _clock_index, context.makeLayoutTopology());
      _status_table->append(*_clock, DomainStatus::kFinished, context.sink_domain, _valid_sinks, context.sinks.size(), "direct");
      return true;
    }

    Topology::Input synthesis_input{
        .config = _config,
        .design = _design,
        .wrapper = _wrapper,
        .fast_sta = _fast_sta,
        .reporter = _reporter,
        .root_net = context.downstream_net,
        .object_name_prefix = context.domain_prefix,
        .characterization_library = _characterization_library,
        .additional_characterization_lengths_um = source_trunk_lengths_um,
        .clock_period_ns = _clock->get_clock_period_ns(),
        .clock_period_source = _clock->get_clock_period_source(),
        .log_context = makeLogContext(*_clock, sink_domain_label, "downstream_htree", context.domain_prefix),
        .htree_loads_are_local_buffers = _clock->is_preclustered_sink_reuse(),
    };
    Topology::Config synthesis_config{
        .enable_sink_clustering = _clock->is_preclustered_sink_reuse() ? false : _config->is_enable_sink_clustering(),
    };

    std::string failure_reason;
    auto synthesis_build = Topology::build(synthesis_input, synthesis_config);
    if (!synthesis_build.summary.success) {
      failure_reason
          = synthesis_build.summary.failure_reason.empty() ? "sink-domain synthesis failed" : synthesis_build.summary.failure_reason;
    } else {
      (void) commitSinkDomainBuild(context, synthesis_build, failure_reason);
    }

    if (!failure_reason.empty()) {
      _status_table->append(*_clock, DomainStatus::kFailed, context.sink_domain, _valid_sinks, context.sinks.size(), failure_reason);
      LOG_ERROR << "Topology: clock \"" << _clock->get_clock_name() << "\" sink domain " << sink_domain_label
                << " failed: " << failure_reason;
      Topology::resetClockTopology(*_design, *_clock);
      return false;
    }

    _status_table->append(*_clock, DomainStatus::kFinished, context.sink_domain, _valid_sinks, context.sinks.size(), "synthesis");
    return true;
  }

  auto buildAndCommitSourceTrunk(const std::vector<Pin*>& root_inputs) -> bool
  {
    const auto source_trunk_domain = SinkDomainKind::kSourceToRoot;
    const auto* const source_trunk_label = ToString(source_trunk_domain);
    auto* clock_source = _clock->get_clock_source();
    auto* clock_source_net = _clock->get_clock_source_net();
    if (clock_source_net == nullptr && clock_source != nullptr) {
      clock_source_net = clock_source->get_net();
      _clock->set_clock_source_net(clock_source_net);
    }
    if (clock_source == nullptr || clock_source_net == nullptr) {
      _status_table->append(*_clock, DomainStatus::kFailed, source_trunk_domain, _valid_sinks, root_inputs.size(),
                            "missing clock source or source net");
      LOG_ERROR << "Topology: clock \"" << _clock->get_clock_name()
                << "\" source trunk formation failed because the source pin or net is missing.";
      Topology::resetClockTopology(*_design, *_clock);
      return false;
    }

    const auto source_trunk_prefix = ClockTreeRealization::makeSinkDomainPrefix(*_clock, _clock_index, source_trunk_domain);
    topology::SourceTrunkInput source_trunk_input{
        .config = _config,
        .design = _design,
        .wrapper = _wrapper,
        .fast_sta = _fast_sta,
        .reporter = _reporter,
        .source_net = clock_source_net,
        .clock_source = clock_source,
        .root_inputs = root_inputs,
        .object_name_prefix = source_trunk_prefix,
        .characterization_library = _characterization_library,
        .clock_period_ns = _clock->get_clock_period_ns(),
        .clock_period_source = _clock->get_clock_period_source(),
        .log_context = makeLogContext(*_clock, source_trunk_label, "source_to_root", source_trunk_prefix),
    };
    topology::SourceTrunkBuild source_trunk_build;
    {
      auto build_stage = _reporter->beginStage("Topology", "Build source trunk",
                                               {
                                                   {"root_inputs", std::to_string(root_inputs.size())},
                                                   {"object_name_prefix", source_trunk_prefix},
                                               },
                                               DetailStageReportOptions());
      source_trunk_build = topology::BuildSourceTrunkTree(source_trunk_input);
      if (source_trunk_build.summary.success) {
        build_stage.finished({
            {"stage", ToString(source_trunk_build.summary.stage)},
            {"inserted_insts", std::to_string(source_trunk_build.output.inserted_insts.size())},
            {"inserted_nets", std::to_string(source_trunk_build.output.inserted_nets.size())},
            {"used_boundary_relaxation", source_trunk_build.summary.used_boundary_relaxation ? "true" : "false"},
        });
      } else {
        build_stage.failed({{"reason", source_trunk_build.summary.failure_reason.empty() ? "source_trunk_formation_failed"
                                                                                         : source_trunk_build.summary.failure_reason}});
      }
    }
    const auto source_trunk_phase = sourceTrunkSynthesisPhase(source_trunk_build.summary.stage);
    if (!source_trunk_build.summary.success) {
      const auto failure_reason
          = source_trunk_build.summary.failure_reason.empty() ? "source trunk formation failed" : source_trunk_build.summary.failure_reason;
      _status_table->append(*_clock, DomainStatus::kFailed, source_trunk_domain, _valid_sinks, root_inputs.size(), failure_reason);
      LOG_ERROR << "Topology: clock \"" << _clock->get_clock_name() << "\" source trunk formation failed: " << failure_reason;
      Topology::resetClockTopology(*_design, *_clock);
      return false;
    }

    {
      auto commit_stage = _reporter->beginStage("Topology", "Commit source trunk layout",
                                                {
                                                    {"stage", ToString(source_trunk_build.summary.stage)},
                                                    {"inserted_insts", std::to_string(source_trunk_build.output.inserted_insts.size())},
                                                    {"inserted_nets", std::to_string(source_trunk_build.output.inserted_nets.size())},
                                                },
                                                StageReportOptions{.context_sink = ReportSink::kDetail, .emit_success_summary = false});
      auto pending_clock_layout = ClockLayoutBuilder::makeSourceToRootLayout(
          *_clock, _clock_index, *clock_source_net,
          ClockLayoutAdapter::makeSourceTrunkLayoutTopology(source_trunk_build, source_trunk_phase), source_trunk_phase);
      if (!ClockTreeRealization::commitInsertedObjects(InsertedObjectCommitInput{
              .design = _design,
              .clock = _clock,
              .inserted_insts = &source_trunk_build.output.inserted_insts,
              .inserted_pins = &source_trunk_build.output.inserted_pins,
              .inserted_nets = &source_trunk_build.output.inserted_nets,
          })) {
        _status_table->append(*_clock, DomainStatus::kFailed, source_trunk_domain, _valid_sinks, root_inputs.size(),
                              "failed to commit source trunk objects");
        LOG_ERROR << "Topology: clock \"" << _clock->get_clock_name()
                  << "\" source trunk formation failed while committing inserted objects.";
        Topology::resetClockTopology(*_design, *_clock);
        commit_stage.failed({{"reason", "failed_to_commit_source_trunk_objects"}});
        return false;
      }

      ClockLayoutBuilder::merge(*_clock_layout, pending_clock_layout);
      commit_stage.finished();
    }
    recordSourceTrunkBuild(*_summary, source_trunk_build);
    _status_table->append(*_clock, DomainStatus::kFinished, source_trunk_domain, _valid_sinks, root_inputs.size(),
                          ToString(source_trunk_build.summary.stage));
    return true;
  }

  const Config* _config = nullptr;
  Design* _design = nullptr;
  Wrapper* _wrapper = nullptr;
  FastSTA* _fast_sta = nullptr;
  SchemaWriter* _reporter = nullptr;
  Clock* _clock = nullptr;
  std::size_t _clock_index = 0U;
  ClockLayout* _clock_layout = nullptr;
  SynthesisTraceSummary* _summary = nullptr;
  DomainStatusTable* _status_table = nullptr;
  CharacterizationLibrary* _characterization_library = nullptr;
  std::size_t _valid_sinks = 0U;
  const std::vector<ClockDistributionContext>* _sink_domains = nullptr;
};

}  // namespace

auto Topology::build(const Input& input, const Config& config) -> Build
{
  return topology::BuildSinkTree(input, config);
}

auto Topology::resetClockTopology(Clock& clock) -> void
{
  ClockTreeRealization::restoreClockSourceNetToClockLoads(clock);
  clock.clearMembership();
}

auto Topology::resetClockTopology(Design& design, Clock& clock) -> void
{
  ClockTreeRealization::restoreClockSourceNetToClockLoads(clock);
  clearClockCtsMembership(design, clock);
}

auto Topology::formClock(const ClockTopologyInput& input) -> bool
{
  LOG_FATAL_IF(input.config == nullptr) << "Topology: clock topology config is null.";
  LOG_FATAL_IF(input.design == nullptr) << "Topology: clock topology design is null.";
  LOG_FATAL_IF(input.wrapper == nullptr) << "Topology: clock topology wrapper is null.";
  LOG_FATAL_IF(input.fast_sta == nullptr) << "Topology: clock topology FastSTA is null.";
  LOG_FATAL_IF(input.reporter == nullptr) << "Topology: clock topology reporter is null.";
  LOG_FATAL_IF(input.clock == nullptr) << "Topology: clock topology clock is null.";
  LOG_FATAL_IF(input.clock_layout == nullptr) << "Topology: clock topology layout is null.";
  LOG_FATAL_IF(input.summary == nullptr) << "Topology: clock topology summary is null.";
  LOG_FATAL_IF(input.status_table == nullptr) << "Topology: clock topology status table is null.";
  LOG_FATAL_IF(input.characterization_library == nullptr) << "Topology: clock topology characterization library is null.";
  LOG_FATAL_IF(input.sink_domains == nullptr) << "Topology: clock topology sink domains are null.";

  ClockTopologySynthesis synthesis(input);
  return synthesis.synthesize();
}

}  // namespace icts
