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
 * @file ClockDataRead.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief CTS clock-data materialization from SDC declarations.
 */

#include "setup/clock_data/ClockDataRead.hh"

#include <glog/logging.h>

#include <map>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "Log.hh"
#include "adapter/sdc/SdcClockReader.hh"
#include "config/Config.hh"
#include "design/Clock.hh"
#include "design/Design.hh"
#include "io/Wrapper.hh"
#include "logger/Schema.hh"

namespace icts {

auto ClockDataRead::read(const ClockDataReadInput& input) -> bool
{
  LOG_FATAL_IF(input.config == nullptr) << "ClockDataRead: config is null.";
  LOG_FATAL_IF(input.design == nullptr) << "ClockDataRead: design is null.";
  LOG_FATAL_IF(input.wrapper == nullptr) << "ClockDataRead: wrapper is null.";
  LOG_FATAL_IF(input.reporter == nullptr) << "ClockDataRead: reporter is null.";

  const auto& config = *input.config;
  auto& design = *input.design;
  auto& wrapper = *input.wrapper;
  auto& reporter = *input.reporter;

  std::string clock_source = "sdc";
  std::vector<ClockTraceClockTarget> clock_targets;

  const auto sdc_clock_data = SdcClockReader().readClockData(reporter);
  std::set<std::string> sdc_clock_names;
  std::set<std::string> traceable_sdc_clock_names;
  std::map<std::string, double> sdc_period_by_clock;
  std::map<std::string, bool> sdc_resolved_by_clock;
  for (const auto& clock_decl : sdc_clock_data.clocks) {
    if (clock_decl.clock_name.empty()) {
      continue;
    }
    sdc_clock_names.insert(clock_decl.clock_name);
    if (!clock_decl.is_virtual) {
      traceable_sdc_clock_names.insert(clock_decl.clock_name);
    }
    sdc_period_by_clock[clock_decl.clock_name] = clock_decl.period_ns;
    sdc_resolved_by_clock[clock_decl.clock_name] = clock_decl.period_resolved;
  }

  if (sdc_clock_names.empty()) {
    EmitDiagnostic(reporter, DiagnosticLevel::kWarning, "ClockDataRead",
                   "no SDC clocks were declared; CTS clock read will be an explicit no-op.", {{"clock_source", "sdc"}});
  }

  bool preflight_failed = false;
  std::string failure_reason = "n/a";
  if (!sdc_clock_data.clocks.empty()) {
    const auto clock_trace_build = wrapper.traceSdcClocks(SdcClockTraceInput{
        .clock_data = &sdc_clock_data,
        .max_fanout = config.get_max_fanout(),
        .reporter = &reporter,
    });
    clock_targets = clock_trace_build.output.clock_targets;
    std::set<std::string> accepted_trace_clock_names;
    for (const auto& clock_target : clock_targets) {
      accepted_trace_clock_names.insert(clock_target.clock_name);
    }
    for (const auto& clock_name : traceable_sdc_clock_names) {
      if (accepted_trace_clock_names.contains(clock_name)) {
        continue;
      }
      preflight_failed = true;
      if (failure_reason == "n/a") {
        failure_reason = "clock_trace_no_targets";
      }
      EmitDiagnostic(reporter, DiagnosticLevel::kError, "ClockDataRead",
                     "SDC clock tracing found no CTS target net for a traceable SDC clock.",
                     {{"clock", clock_name}, {"clock_source", "sdc"}});
      LOG_ERROR << "ClockDataRead: SDC clock tracing found no CTS target net for \"" << clock_name << "\".";
    }
  }

  if (!preflight_failed && !clock_targets.empty() && !wrapper.readTraceClockTargets(design, reporter, clock_targets)) {
    preflight_failed = true;
    failure_reason = "clock_materialization_failed";
  }

  for (auto* clock : design.get_clocks()) {
    if (clock == nullptr) {
      continue;
    }
    const auto period_iter = sdc_period_by_clock.find(clock->get_clock_name());
    const auto resolved_iter = sdc_resolved_by_clock.find(clock->get_clock_name());
    const bool period_resolved = resolved_iter == sdc_resolved_by_clock.end() || resolved_iter->second;
    if (period_iter != sdc_period_by_clock.end() && period_iter->second > 0.0 && period_resolved) {
      clock->set_clock_period_ns(period_iter->second);
      clock->set_clock_period_source("sdc");
    }
  }

  const auto materialized_clock_count = design.get_clocks().size();
  if (!preflight_failed) {
    design.emitClockDistributionSummary(reporter);
  }
  KeyValueFields read_data_fields = {
      {"clock_source", clock_source},
      {"status", preflight_failed ? "failed" : "finished"},
  };
  if (preflight_failed && !failure_reason.empty()) {
    read_data_fields.emplace_back("failure_reason", failure_reason);
  }
  read_data_fields.insert(read_data_fields.end(), {
                                                      {"sdc_declared_clocks", std::to_string(sdc_clock_names.size())},
                                                      {"added_clock_nets", std::to_string(materialized_clock_count)},
                                                      {"total_clock_nets", std::to_string(materialized_clock_count)},
                                                  });
  EmitKeyValueTable(reporter, "ReadData Overview", read_data_fields);
  if (preflight_failed) {
    design.clearClocks();
    design.clearTopologyObjects();
  }
  return !preflight_failed;
}

}  // namespace icts
