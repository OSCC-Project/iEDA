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
 * @file Synthesis.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-30
 * @brief CTS synthesis entry facade implementation.
 */

#include "synthesis/Synthesis.hh"

#include <glog/logging.h>

#include <cstddef>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Log.hh"
#include "config/Config.hh"
#include "design/Clock.hh"
#include "design/ClockLayout.hh"
#include "design/Design.hh"
#include "design/Pin.hh"
#include "io/Wrapper.hh"
#include "logger/Schema.hh"
#include "synthesis/distribution/ClockDistribution.hh"
#include "synthesis/topology/Topology.hh"
#include "synthesis/trace/domain_status/DomainStatus.hh"
#include "synthesis/trace/layout/ClockLayoutBuilder.hh"

namespace icts {

class Net;

namespace {

struct ClockSynthesisSummary
{
  bool success = false;
  bool skipped = false;
};

struct ClockSynthesisCounters
{
  std::size_t total_sink_domains = 0U;
  std::size_t hard_macro_sinks = 0U;
  std::size_t regular_sinks = 0U;
};

struct ClockSynthesisRunInput
{
  const Config* config = nullptr;
  Design* design = nullptr;
  Wrapper* wrapper = nullptr;
  FastSTA* fast_sta = nullptr;
  SchemaWriter* reporter = nullptr;
  Clock* clock = nullptr;
  std::size_t clock_index = 0U;
  ClockLayout* clock_layout = nullptr;
  SynthesisTraceSummary* summary = nullptr;
  TableRows* domain_status_rows = nullptr;
  ClockSynthesisCounters* counters = nullptr;
  CharacterizationLibrary* characterization_library = nullptr;
};

auto RequireDomainStatusRows(TableRows* rows) -> TableRows&
{
  LOG_FATAL_IF(rows == nullptr) << "Synthesis: domain status rows are null.";
  return *rows;
}

class ClockSynthesisRun
{
 public:
  explicit ClockSynthesisRun(const ClockSynthesisRunInput& input)
      : _config(input.config),
        _design(input.design),
        _wrapper(input.wrapper),
        _fast_sta(input.fast_sta),
        _reporter(input.reporter),
        _clock(input.clock),
        _clock_index(input.clock_index),
        _clock_layout(input.clock_layout),
        _summary(input.summary),
        _status_table(RequireDomainStatusRows(input.domain_status_rows)),
        _counters(input.counters),
        _characterization_library(input.characterization_library)
  {
    LOG_FATAL_IF(_config == nullptr) << "Synthesis: per-clock config is null.";
    LOG_FATAL_IF(_design == nullptr) << "Synthesis: per-clock design is null.";
    LOG_FATAL_IF(_wrapper == nullptr) << "Synthesis: per-clock wrapper is null.";
    LOG_FATAL_IF(_fast_sta == nullptr) << "Synthesis: per-clock FastSTA is null.";
    LOG_FATAL_IF(_reporter == nullptr) << "Synthesis: per-clock reporter is null.";
    LOG_FATAL_IF(_clock == nullptr) << "Synthesis: per-clock clock is null.";
    LOG_FATAL_IF(_clock_layout == nullptr) << "Synthesis: per-clock layout is null.";
    LOG_FATAL_IF(_summary == nullptr) << "Synthesis: per-clock summary is null.";
    LOG_FATAL_IF(_counters == nullptr) << "Synthesis: per-clock counters are null.";
    LOG_FATAL_IF(_characterization_library == nullptr) << "Synthesis: per-clock characterization library is null.";
  }

  auto run() -> ClockSynthesisSummary;

 private:
  auto ensureClockSource() -> std::pair<Pin*, Net*>;
  auto prepareSinkDomain(SinkDomainKind sink_domain, const std::vector<Pin*>& sinks, std::size_t valid_sinks) -> bool;
  auto formClockTopology(std::size_t valid_sinks) -> bool;

  const Config* _config = nullptr;
  Design* _design = nullptr;
  Wrapper* _wrapper = nullptr;
  FastSTA* _fast_sta = nullptr;
  SchemaWriter* _reporter = nullptr;
  Clock* _clock = nullptr;
  std::size_t _clock_index = 0U;
  ClockLayout* _clock_layout = nullptr;
  SynthesisTraceSummary* _summary = nullptr;
  DomainStatusTable _status_table;
  ClockSynthesisCounters* _counters = nullptr;
  CharacterizationLibrary* _characterization_library = nullptr;
  ClockLayout _per_clock_layout;
  std::vector<ClockDistributionContext> _sink_domain_contexts;
};

auto synthesisOutcomeName(SynthesisOutcome outcome) -> std::string
{
  switch (outcome) {
    case SynthesisOutcome::kFinished:
      return "finished";
    case SynthesisOutcome::kFailed:
      return "failed";
    case SynthesisOutcome::kNoOp:
      return "no_op";
  }
  return "failed";
}

auto emitSynthesisOverview(SchemaWriter& reporter, const SynthesisTraceSummary& summary, const TableRows& rows) -> void
{
  reporter.emitSection("### Flow Status");
  KeyValueFields fields = {
      {"status", synthesisOutcomeName(summary.outcome)},
  };
  if (!summary.no_op_reason.empty()) {
    fields.emplace_back("no_op_reason", summary.no_op_reason);
  }
  fields.insert(fields.end(), {
                                  {"total_clocks", std::to_string(summary.total_clocks)},
                                  {"finished_clocks", std::to_string(summary.successful_clocks)},
                                  {"skipped_clocks", std::to_string(summary.skipped_clocks)},
                                  {"failed_clocks", std::to_string(summary.failed_clocks)},
                                  {"total_sink_domains", std::to_string(summary.total_sink_domains)},
                                  {"hard_macro_sinks", std::to_string(summary.hard_macro_sinks)},
                                  {"regular_sinks", std::to_string(summary.regular_sinks)},
                              });
  EmitKeyValueTable(reporter, "CTS Clock Tree Synthesis Overview", fields);

  if (!rows.empty()) {
    EmitTable(reporter, "CTS Clock Tree Sink Domains", {"Clock", "Net", "Status", "Sink Domain", "Valid Sinks", "Domain Sinks", "Detail"},
              rows);
  }
}

auto ClockSynthesisRun::ensureClockSource() -> std::pair<Pin*, Net*>
{
  auto* clock_source = _clock->get_clock_source();
  auto* clock_source_net = _clock->get_clock_source_net();
  if (clock_source_net == nullptr && clock_source != nullptr) {
    clock_source_net = clock_source->get_net();
    _clock->set_clock_source_net(clock_source_net);
  }
  return {clock_source, clock_source_net};
}

auto ClockSynthesisRun::prepareSinkDomain(SinkDomainKind sink_domain, const std::vector<Pin*>& sinks, std::size_t valid_sinks) -> bool
{
  if (sinks.empty()) {
    return true;
  }
  ++_counters->total_sink_domains;
  auto context = ClockDistribution::prepare(ClockDistributionInput{
      .design = _design,
      .clock = _clock,
      .wrapper = _wrapper,
      .clock_index = _clock_index,
      .sink_domain = sink_domain,
      .sinks = sinks,
      .valid_sinks = valid_sinks,
      .root_buffer_types = _config->get_buffer_types(),
      .status_table = &_status_table,
  });
  if (!context.has_value()) {
    Topology::resetClockTopology(*_design, *_clock);
    return false;
  }
  _sink_domain_contexts.push_back(std::move(*context));
  return true;
}

auto ClockSynthesisRun::formClockTopology(std::size_t valid_sinks) -> bool
{
  if (!Topology::formClock(ClockTopologyInput{
          .config = _config,
          .design = _design,
          .wrapper = _wrapper,
          .fast_sta = _fast_sta,
          .reporter = _reporter,
          .clock = _clock,
          .clock_index = _clock_index,
          .clock_layout = &_per_clock_layout,
          .summary = _summary,
          .status_table = &_status_table,
          .characterization_library = _characterization_library,
          .valid_sinks = valid_sinks,
          .sink_domains = &_sink_domain_contexts,
      })) {
    return false;
  }
  ClockLayoutBuilder::merge(*_clock_layout, _per_clock_layout);
  return true;
}

auto ClockSynthesisRun::run() -> ClockSynthesisSummary
{
  Topology::resetClockTopology(*_design, *_clock);
  _per_clock_layout.ensureClock(_clock->get_clock_name(), _clock->get_clock_net_name(), _clock_index);

  const auto [clock_source, clock_source_net] = ensureClockSource();
  if (clock_source == nullptr) {
    _status_table.appendNoDomain(*_clock, DomainStatus::kSkipped, 0U, 0U, "clock source is null");
    LOG_WARNING << "Synthesis: skip clock \"" << _clock->get_clock_name() << "\" because clock source is null.";
    return ClockSynthesisSummary{.success = false, .skipped = true};
  }
  if (clock_source_net == nullptr) {
    _status_table.appendNoDomain(*_clock, DomainStatus::kFailed, 0U, 0U, "clock source net is null");
    LOG_ERROR << "Synthesis: clock \"" << _clock->get_clock_name() << "\" failed because the clock source net is null.";
    return ClockSynthesisSummary{.success = false, .skipped = false};
  }

  const auto sink_partition = ClockDistribution::partitionSinkDomains(*_clock);
  const auto valid_sinks = sink_partition.valid_sink_count;
  _counters->hard_macro_sinks += sink_partition.macro_sinks.size();
  _counters->regular_sinks += sink_partition.regular_sinks.size();
  if (valid_sinks == 0U) {
    _status_table.appendNoDomain(*_clock, DomainStatus::kSkipped, 0U, 0U, "no valid sinks");
    LOG_WARNING << "Synthesis: skip clock \"" << _clock->get_clock_name() << "\" because no valid sinks are available.";
    return ClockSynthesisSummary{.success = false, .skipped = true};
  }
  ClockLayoutBuilder::appendSinkInsts(_per_clock_layout, *_clock, _clock_index, sink_partition.macro_sinks, SinkDomainKind::kHardMacro);
  ClockLayoutBuilder::appendSinkInsts(_per_clock_layout, *_clock, _clock_index, sink_partition.regular_sinks, SinkDomainKind::kRegular);

  _sink_domain_contexts.reserve(2U);
  if (!prepareSinkDomain(SinkDomainKind::kHardMacro, sink_partition.macro_sinks, valid_sinks)) {
    return ClockSynthesisSummary{.success = false, .skipped = false};
  }
  if (!prepareSinkDomain(SinkDomainKind::kRegular, sink_partition.regular_sinks, valid_sinks)) {
    return ClockSynthesisSummary{.success = false, .skipped = false};
  }
  return formClockTopology(valid_sinks) ? ClockSynthesisSummary{.success = true, .skipped = false}
                                        : ClockSynthesisSummary{.success = false, .skipped = false};
}

auto recordDomainStatusRows(SynthesisTraceSummary& summary, const TableRows& rows) -> void
{
  summary.domain_status.clear();
  summary.domain_status.reserve(rows.size());
  for (const auto& row : rows) {
    if (row.size() < 7U) {
      continue;
    }
    auto parse_count = [](const std::string& value) -> std::size_t {
      std::size_t count = 0U;
      std::istringstream stream(value);
      stream >> count;
      return stream.fail() ? 0U : count;
    };
    summary.domain_status.push_back(SynthesisTraceStatusRecord{
        .clock_name = row.at(0),
        .clock_net_name = row.at(1),
        .status = row.at(2),
        .sink_domain = row.at(3),
        .valid_sink_count = parse_count(row.at(4)),
        .sink_domain_sink_count = parse_count(row.at(5)),
        .detail = row.at(6),
    });
  }
}

}  // namespace

auto Synthesis::run(const SynthesisInput& input) -> SynthesisTraceSummary
{
  LOG_FATAL_IF(input.config == nullptr) << "Synthesis: config is null.";
  LOG_FATAL_IF(input.design == nullptr) << "Synthesis: design is null.";
  LOG_FATAL_IF(input.wrapper == nullptr) << "Synthesis: wrapper is null.";
  LOG_FATAL_IF(input.fast_sta == nullptr) << "Synthesis: FastSTA is null.";
  LOG_FATAL_IF(input.reporter == nullptr) << "Synthesis: reporter is null.";
  LOG_FATAL_IF(input.clock_layout == nullptr) << "Synthesis: clock layout is null.";
  LOG_FATAL_IF(input.characterization_library == nullptr) << "Synthesis: characterization library is null.";

  const auto& config = *input.config;
  auto& design = *input.design;
  auto& wrapper = *input.wrapper;
  auto& fast_sta = *input.fast_sta;
  auto& reporter = *input.reporter;
  auto& clock_layout = *input.clock_layout;
  auto& char_library = *input.characterization_library;

  auto runtime = reporter.beginRuntimeMetric("synthesis");
  auto flow_stage = reporter.beginStage("CTSFlow", "Run CTS synthesis flow", {}, StageReportOptions{.emit_success_summary = false});
  reporter.emitSection("## Synthesis Overview");

  clock_layout.reset();
  SynthesisTraceSummary summary;
  auto clocks = design.get_clocks();
  const std::size_t total_clocks = clocks.size();
  std::size_t successful_clocks = 0U;
  std::size_t skipped_clocks = 0U;
  std::size_t failed_clocks = 0U;
  ClockSynthesisCounters synthesis_counters;
  TableRows rows;

  for (std::size_t clock_index = 0; clock_index < clocks.size(); ++clock_index) {
    auto* clock = clocks.at(clock_index);
    if (clock == nullptr) {
      ++skipped_clocks;
      rows.push_back({"", "", "skipped", "none", "0", "0", "clock pointer is null"});
      continue;
    }

    const auto sink_partition = ClockDistribution::partitionSinkDomains(*clock);
    if (sink_partition.valid_sink_count > 0U && wrapper.is_design_ready()) {
      const auto dbu_per_um = wrapper.queryDbUnit();
      LOG_FATAL_IF(dbu_per_um <= 0) << "Synthesis: DBU-per-micron is unavailable.";
      clock_layout.set_design_dbu_per_um(dbu_per_um);
    }

    ClockSynthesisRun clock_synthesis(ClockSynthesisRunInput{
        .config = &config,
        .design = &design,
        .wrapper = &wrapper,
        .fast_sta = &fast_sta,
        .reporter = &reporter,
        .clock = clock,
        .clock_index = clock_index,
        .clock_layout = &clock_layout,
        .summary = &summary,
        .domain_status_rows = &rows,
        .counters = &synthesis_counters,
        .characterization_library = &char_library,
    });
    const auto clock_summary = clock_synthesis.run();
    if (clock_summary.success) {
      ++successful_clocks;
    } else if (clock_summary.skipped) {
      ++skipped_clocks;
    } else {
      ++failed_clocks;
    }
  }

  summary.total_clocks = total_clocks;
  summary.successful_clocks = successful_clocks;
  summary.skipped_clocks = skipped_clocks;
  summary.failed_clocks = failed_clocks;
  summary.success = successful_clocks > 0U && failed_clocks == 0U;
  if (total_clocks == 0U) {
    summary.outcome = SynthesisOutcome::kNoOp;
    summary.no_op_reason = "no_clocks_discovered";
  } else if (successful_clocks == 0U && skipped_clocks > 0U && failed_clocks == 0U) {
    summary.outcome = SynthesisOutcome::kNoOp;
    summary.no_op_reason = "all_clocks_skipped";
  } else {
    summary.outcome = summary.success ? SynthesisOutcome::kFinished : SynthesisOutcome::kFailed;
  }
  summary.total_sink_domains = synthesis_counters.total_sink_domains;
  summary.hard_macro_sinks = synthesis_counters.hard_macro_sinks;
  summary.regular_sinks = synthesis_counters.regular_sinks;
  recordDomainStatusRows(summary, rows);
  clock_layout.markSynthesisComplete(summary.success);

  LOG_INFO << "CTS clock-tree synthesis finished with " << successful_clocks << " successful, " << skipped_clocks << " skipped, "
           << failed_clocks << " failed clocks.";
  emitSynthesisOverview(reporter, summary, rows);

  if (summary.outcome == SynthesisOutcome::kFinished) {
    (void) runtime.finished();
    flow_stage.finished();
  } else if (summary.outcome == SynthesisOutcome::kNoOp) {
    (void) runtime.finish("no_op");
    flow_stage.skip({{"reason", summary.no_op_reason}});
  } else {
    (void) runtime.failed();
    flow_stage.failed();
  }
  return summary;
}

}  // namespace icts
