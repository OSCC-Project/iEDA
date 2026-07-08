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
 * @file ClockTraceReport.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Report assembly helpers for SDC clock tracing.
 */

#include <algorithm>
#include <compare>
#include <cstddef>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "IdbDesign.h"
#include "IdbNet.h"
#include "SdcClockReader.hh"
#include "SdcClockTraceAlgorithm.hh"
#include "logger/Schema.hh"

namespace icts::clock_trace {

auto JoinNames(const std::set<std::string>& names, std::size_t display_limit) -> std::string
{
  if (names.empty()) {
    return "n/a";
  }
  std::ostringstream stream;
  std::size_t emitted_count = 0U;
  for (const auto& name : names) {
    if (emitted_count > 0U) {
      stream << ", ";
    }
    if (emitted_count >= display_limit) {
      stream << "...";
      break;
    }
    stream << name;
    ++emitted_count;
  }
  if (names.size() > display_limit) {
    stream << " (" << names.size() << " total)";
  }
  return stream.str();
}

auto BuildClockDeclViews(idb::IdbDesign* idb_design, const SdcClockData& clock_data) -> std::map<std::string, ClockDeclView>
{
  std::map<std::string, ClockDeclView> clock_view_by_name;
  for (const auto& clock : clock_data.clocks) {
    auto& view = clock_view_by_name[clock.clock_name];
    view.clock_kind = ClockKindName(clock);
    view.master_clock_name = MasterClockName(clock);
    for (const auto& target : clock.targets) {
      for (auto* net : ResolveRefNets(idb_design, target)) {
        if (net != nullptr) {
          view.sdc_target_net_names.insert(net->get_net_name());
        }
      }
    }
  }
  return clock_view_by_name;
}

auto AnnotateRecordOwnership(ClockTraceRecord& record, const std::map<std::string, ClockDeclView>& clock_view_by_name) -> void
{
  const auto view_iter = clock_view_by_name.find(record.clock_name);
  if (view_iter == clock_view_by_name.end()) {
    return;
  }
  record.clock_kind = view_iter->second.clock_kind;
  record.master_clock_name = view_iter->second.master_clock_name;
  record.dominance = DominanceForRecord(record, record.clock_kind);
}

auto CollectTracedNetNames(const std::vector<ClockTraceRecord>& records) -> std::set<std::string>
{
  std::set<std::string> traced_net_names;
  for (const auto& record : records) {
    if (record.net_name.empty() || record.status == "rejected" || record.status == "skipped") {
      continue;
    }
    traced_net_names.insert(record.net_name);
  }
  return traced_net_names;
}

auto CollectUnownedClockLikeRecords(const SdcLibertyCellLookup& liberty_cell_lookup, idb::IdbDesign* idb_design,
                                    const std::vector<ClockTraceRecord>& records) -> std::vector<ClockTraceRecord>
{
  std::vector<ClockTraceRecord> unowned_records;
  auto* net_list = idb_design == nullptr ? nullptr : idb_design->get_net_list();
  if (net_list == nullptr) {
    return unowned_records;
  }

  const auto traced_net_names = CollectTracedNetNames(records);
  for (auto* net : net_list->get_net_list()) {
    if (net == nullptr || traced_net_names.contains(net->get_net_name())) {
      continue;
    }
    const auto stats = CountDirectClockSinksForReport(liberty_cell_lookup, net);
    if (!IsClockTarget(stats)) {
      continue;
    }
    ClockTraceRecord record;
    record.clock_name = "unowned";
    record.net_name = net->get_net_name();
    record.status = "warning";
    record.target_kind = TargetKind(stats);
    record.sequential_clock_sinks = stats.sequential_clock_sinks;
    record.macro_clock_sinks = stats.macro_clock_sinks;
    record.reason = "no_sdc_clock_ownership";
    record.clock_kind = "unowned";
    record.master_clock_name = "n/a";
    record.dominance = "unowned_clock_like_net";
    unowned_records.push_back(std::move(record));
  }

  std::ranges::sort(unowned_records, [](const auto& lhs, const auto& rhs) -> bool { return lhs.net_name < rhs.net_name; });
  return unowned_records;
}

auto NumberToString(std::size_t value) -> std::string
{
  std::ostringstream stream;
  stream << value;
  return stream.str();
}

auto EmitClockTraceReport(SchemaWriter& reporter, const std::vector<ClockTraceRecord>& records) -> void
{
  TableRows rows;
  rows.reserve(records.size());
  for (const auto& record : records) {
    rows.push_back({
        record.clock_name,
        record.clock_kind,
        record.master_clock_name,
        record.net_name,
        record.status,
        record.dominance,
        record.target_kind,
        NumberToString(record.sequential_clock_sinks),
        NumberToString(record.macro_clock_sinks),
        record.trace_path,
        record.reason,
    });
  }
  if (rows.empty()) {
    rows.push_back({"n/a", "n/a", "n/a", "n/a", "skipped", "undetermined", "n/a", "0", "0", "", "no_sdc_clock_trace_records"});
  }
  EmitTable(reporter, "Clock Trace Overview",
            {"Clock", "Kind", "Master Clock", "Net", "Status", "Dominance", "Target Kind", "Seq CK Sinks", "Macro CK Sinks", "Trace Path",
             "Reason"},
            rows);
}

auto EmitSdcClockOwnershipReport(const SdcClockData& clock_data, const std::map<std::string, ClockDeclView>& clock_view_by_name,
                                 SchemaWriter& reporter, const std::vector<ClockTraceRecord>& records) -> void
{
  std::map<std::string, std::set<std::string>> owned_net_names_by_clock;
  std::map<std::string, std::set<std::string>> cts_target_net_names_by_clock;
  std::map<std::string, ClockSinkStats> target_stats_by_clock;
  for (const auto& record : records) {
    if (record.net_name.empty()) {
      continue;
    }
    if (record.dominance == "owned_by_primary_clock" || record.dominance == "owned_by_generated_clock") {
      owned_net_names_by_clock[record.clock_name].insert(record.net_name);
    }
    if (record.status == "accepted") {
      cts_target_net_names_by_clock[record.clock_name].insert(record.net_name);
      target_stats_by_clock[record.clock_name].sequential_clock_sinks += record.sequential_clock_sinks;
      target_stats_by_clock[record.clock_name].macro_clock_sinks += record.macro_clock_sinks;
    }
  }

  TableRows rows;
  rows.reserve(clock_data.clocks.size());
  for (const auto& clock : clock_data.clocks) {
    const auto view_iter = clock_view_by_name.find(clock.clock_name);
    const auto clock_kind = view_iter == clock_view_by_name.end() ? ClockKindName(clock) : view_iter->second.clock_kind;
    const auto master_clock = view_iter == clock_view_by_name.end() ? MasterClockName(clock) : view_iter->second.master_clock_name;
    const auto sdc_target_nets = view_iter == clock_view_by_name.end() ? std::set<std::string>{} : view_iter->second.sdc_target_net_names;
    const auto stats_iter = target_stats_by_clock.find(clock.clock_name);
    const auto sequential_clock_sinks = stats_iter == target_stats_by_clock.end() ? 0U : stats_iter->second.sequential_clock_sinks;
    const auto macro_clock_sinks = stats_iter == target_stats_by_clock.end() ? 0U : stats_iter->second.macro_clock_sinks;
    rows.push_back({clock.clock_name, clock_kind, master_clock, JoinNames(sdc_target_nets),
                    JoinNames(owned_net_names_by_clock[clock.clock_name]), JoinNames(cts_target_net_names_by_clock[clock.clock_name]),
                    NumberToString(sequential_clock_sinks), NumberToString(macro_clock_sinks)});
  }
  if (rows.empty()) {
    rows.push_back({"n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "0", "0"});
  }
  EmitTable(reporter, "SDC Clock Ownership Overview",
            {"Clock", "Kind", "Master Clock", "SDC Target Nets", "Owned Nets", "CTS Target Nets", "Seq CK Sinks", "Macro CK Sinks"}, rows);
}

auto EmitUnownedClockLikeNetReport(SchemaWriter& reporter, const std::vector<ClockTraceRecord>& records) -> void
{
  TableRows rows;
  rows.reserve(records.size());
  for (const auto& record : records) {
    rows.push_back({record.net_name, record.dominance, record.target_kind, NumberToString(record.sequential_clock_sinks),
                    NumberToString(record.macro_clock_sinks), record.reason});
  }
  if (rows.empty()) {
    rows.push_back({"n/a", "none", "n/a", "0", "0", "no_unowned_clock_like_nets"});
  }
  EmitTable(reporter, "Unowned Clock-like Nets", {"Net", "Dominance", "Target Kind", "Seq CK Sinks", "Macro CK Sinks", "Reason"}, rows);
}

}  // namespace icts::clock_trace
