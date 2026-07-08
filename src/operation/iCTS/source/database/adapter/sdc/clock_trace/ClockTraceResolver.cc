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
 * @file ClockTraceResolver.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-15
 * @brief SDC-rooted CTS clock trace resolver implementation.
 */

#include "ClockTraceResolver.hh"

#include <glog/logging.h>

#include <algorithm>
#include <map>
#include <optional>
#include <ostream>
#include <ranges>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "IdbDesign.h"
#include "IdbNet.h"
#include "Log.hh"
#include "SdcClockReader.hh"
#include "clock_trace/SdcClockTraceAlgorithm.hh"
#include "logger/Schema.hh"

namespace icts {
namespace {

auto findIdbNet(idb::IdbDesign* idb_design, const std::string& net_name) -> idb::IdbNet*
{
  auto* net_list = idb_design == nullptr ? nullptr : idb_design->get_net_list();
  return net_list == nullptr ? nullptr : net_list->find_net(net_name);
}

auto resolveSingleSeedNet(idb::IdbDesign* idb_design, const SdcClockDecl& clock) -> idb::IdbNet*
{
  std::vector<idb::IdbNet*> seed_nets;
  for (const auto& target : clock.targets) {
    auto resolved_nets = clock_trace::ResolveRefNets(idb_design, target);
    seed_nets.insert(seed_nets.end(), resolved_nets.begin(), resolved_nets.end());
  }
  std::ranges::sort(seed_nets);
  const auto unique_seed_nets = std::ranges::unique(seed_nets);
  seed_nets.erase(unique_seed_nets.begin(), unique_seed_nets.end());
  return seed_nets.size() == 1U ? seed_nets.front() : nullptr;
}

auto makeDirectClockTarget(const std::string& clock_name, const std::string& net_name) -> ClockTraceClockTarget
{
  return ClockTraceClockTarget{
      .clock_name = clock_name,
      .clock_net_name = net_name,
      .preclustered_sink_reuse = false,
      .preclustered_sink_anchors = {},
  };
}

auto tryBuildPreclusteredClockTarget(const SdcLibertyCellLookup& liberty_cell_lookup, idb::IdbDesign* idb_design, const SdcClockDecl& clock,
                                     const std::vector<ClockTraceRecord>& accepted_records) -> std::optional<ClockTraceClockTarget>
{
  if (clock.kind != SdcClockDecl::Kind::kPrimary || accepted_records.size() < 2U) {
    return std::nullopt;
  }
  auto* source_net = resolveSingleSeedNet(idb_design, clock);
  if (source_net == nullptr) {
    return std::nullopt;
  }

  std::vector<ClockTracePreclusteredSinkAnchor> anchors;
  anchors.reserve(accepted_records.size());
  std::set<std::string> anchor_input_pins;
  for (const auto& record : accepted_records) {
    auto* leaf_net = findIdbNet(idb_design, record.net_name);
    auto anchor = clock_trace::BuildPreclusteredSinkAnchor(liberty_cell_lookup, leaf_net);
    if (!anchor.has_value() || !anchor_input_pins.insert(anchor->driver_inst_name + "/" + anchor->input_pin_name).second) {
      return std::nullopt;
    }
    anchors.push_back(std::move(*anchor));
  }

  return ClockTraceClockTarget{
      .clock_name = clock.clock_name,
      .clock_net_name = source_net->get_net_name(),
      .preclustered_sink_reuse = true,
      .preclustered_sink_anchors = std::move(anchors),
  };
}

}  // namespace

auto ClockTraceResolver::resolve(const SdcClockData& clock_data, idb::IdbDesign* idb_design,
                                 const SdcLibertyCellLookup& liberty_cell_lookup, std::size_t max_fanout, SchemaWriter& reporter)
    -> ClockTraceBuild
{
  ClockTraceBuild build;
  if (idb_design == nullptr || idb_design->get_net_list() == nullptr) {
    EmitDiagnostic(reporter, DiagnosticLevel::kError, "ClockTraceResolver", "clock tracing skipped because iDB design is not ready.",
                   {{"clock_source", "sdc"}});
    LOG_ERROR << "ClockTraceResolver: iDB design or net list is null.";
    return build;
  }

  const auto case_constraints = clock_trace::BuildCaseConstraintSet(clock_data);
  const auto generated_boundary_owner_by_net = clock_trace::BuildGeneratedBoundaryOwners(idb_design, clock_data);
  const auto clock_view_by_name = clock_trace::BuildClockDeclViews(idb_design, clock_data);
  std::unordered_map<std::string, const SdcClockDecl*> clock_decl_by_name;
  clock_decl_by_name.reserve(clock_data.clocks.size());
  for (const auto& clock : clock_data.clocks) {
    if (!clock.clock_name.empty()) {
      clock_decl_by_name[clock.clock_name] = &clock;
    }
  }

  std::vector<ClockTraceRecord> candidate_records;
  for (const auto& clock : clock_data.clocks) {
    auto records = clock_trace::TraceClock(liberty_cell_lookup, idb_design, clock, case_constraints, generated_boundary_owner_by_net);
    candidate_records.insert(candidate_records.end(), records.begin(), records.end());
  }

  std::map<std::string, std::set<std::string>> accepted_clock_names_by_net;
  std::map<std::string, bool> has_strong_target_by_clock;
  const auto sink_threshold = clock_trace::StrongTargetSinkThreshold(max_fanout);
  for (const auto& record : candidate_records) {
    if (record.status == "accepted" && !record.net_name.empty()) {
      accepted_clock_names_by_net[record.net_name].insert(record.clock_name);
      has_strong_target_by_clock[record.clock_name]
          = has_strong_target_by_clock[record.clock_name] || clock_trace::IsStrongClockTarget(record, sink_threshold);
    }
  }

  std::vector<ClockTraceRecord> resolved_records;
  resolved_records.reserve(candidate_records.size());
  for (auto record : candidate_records) {
    if (record.status == "accepted" && has_strong_target_by_clock[record.clock_name]
        && !clock_trace::IsStrongClockTarget(record, sink_threshold)) {
      record.status = "trace_stop";
      record.reason = "source_side_clock_sinks_below_target_threshold";
    }
    if (record.status == "accepted" && accepted_clock_names_by_net[record.net_name].size() > 1U) {
      record.status = "ambiguous";
      record.reason = "target_net_reachable_from_multiple_sdc_clocks";
    }
    clock_trace::AnnotateRecordOwnership(record, clock_view_by_name);
    resolved_records.push_back(std::move(record));
  }

  std::map<std::string, std::vector<ClockTraceRecord>> accepted_records_by_clock;
  for (const auto& record : resolved_records) {
    if (record.status == "accepted" && !record.clock_name.empty() && !record.net_name.empty()) {
      accepted_records_by_clock[record.clock_name].push_back(record);
    }
  }

  std::set<std::pair<std::string, std::string>> emitted_pairs;
  for (const auto& [clock_name, accepted_records] : accepted_records_by_clock) {
    const auto decl_iter = clock_decl_by_name.find(clock_name);
    if (decl_iter != clock_decl_by_name.end()) {
      auto preclustered_target = tryBuildPreclusteredClockTarget(liberty_cell_lookup, idb_design, *decl_iter->second, accepted_records);
      if (preclustered_target.has_value()) {
        if (emitted_pairs.insert({preclustered_target->clock_name, preclustered_target->clock_net_name}).second) {
          build.output.clock_targets.push_back(std::move(*preclustered_target));
        }
        continue;
      }
    }

    for (const auto& record : accepted_records) {
      if (emitted_pairs.insert({record.clock_name, record.net_name}).second) {
        build.output.clock_targets.push_back(makeDirectClockTarget(record.clock_name, record.net_name));
      }
    }
  }

  build.summary.records = std::move(resolved_records);
  build.summary.unowned_clock_like_records
      = clock_trace::CollectUnownedClockLikeRecords(liberty_cell_lookup, idb_design, build.summary.records);
  clock_trace::EmitClockTraceReport(reporter, build.summary.records);
  clock_trace::EmitSdcClockOwnershipReport(clock_data, clock_view_by_name, reporter, build.summary.records);
  clock_trace::EmitUnownedClockLikeNetReport(reporter, build.summary.unowned_clock_like_records);
  LOG_INFO << "ClockTraceResolver: accepted " << build.output.clock_targets.size() << " CTS clock target net(s) from "
           << clock_data.clocks.size() << " SDC clock declaration(s); reported " << build.summary.unowned_clock_like_records.size()
           << " unowned clock-like net(s).";
  return build;
}

}  // namespace icts
