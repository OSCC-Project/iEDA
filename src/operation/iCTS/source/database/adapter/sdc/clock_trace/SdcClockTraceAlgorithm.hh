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
 * @file SdcClockTraceAlgorithm.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief SDC clock trace traversal, ownership, and report helper declarations.
 */

#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "ClockTraceResolver.hh"
#include "SdcClockReader.hh"

struct RustLibertyExpr;

namespace idb {
class IdbDesign;
class IdbInstance;
class IdbNet;
class IdbPin;
}  // namespace idb

namespace idb {
class LibCell;
class LibPort;
}  // namespace idb

namespace icts {
class SchemaWriter;
}  // namespace icts

namespace icts::clock_trace {

inline constexpr std::size_t kTraceNetVisitLimit = 200000U;
inline constexpr std::size_t kTraceDepthLimit = 512U;

struct IdbNetPins
{
  idb::IdbPin* driver = nullptr;
  std::vector<idb::IdbPin*> loads;
  std::vector<idb::IdbPin*> all_pins;
};

struct ClockSinkStats
{
  std::size_t sequential_clock_sinks = 0U;
  std::size_t macro_clock_sinks = 0U;
};

struct TraceNode
{
  idb::IdbNet* net = nullptr;
  std::string path;
  std::size_t depth = 0U;
};

struct TraceTransition
{
  idb::IdbNet* net = nullptr;
  std::string path_step;
  std::string reason;
};

struct CaseConstraintSet
{
  std::set<std::string> pin_names;
  std::set<std::string> net_names;
};

struct ClockDeclView
{
  std::string clock_kind;
  std::string master_clock_name;
  std::set<std::string> sdc_target_net_names;
};

auto NetName(idb::IdbNet* net) -> std::string;
auto TermName(idb::IdbPin* pin) -> std::string;
auto PinFullName(idb::IdbPin* pin) -> std::string;
auto PinDisplayName(idb::IdbPin* pin) -> std::string;
auto CollectNetPins(idb::IdbNet* net) -> IdbNetPins;
auto IsInputLike(idb::IdbPin* pin) -> bool;
auto IsOutputLike(idb::IdbPin* pin) -> bool;
auto FindLibCell(const SdcLibertyCellLookup& liberty_cell_lookup, idb::IdbInstance* inst) -> idb::LibCell*;
auto FindLibPort(idb::LibCell* lib_cell, idb::IdbPin* pin) -> idb::LibPort*;
auto IsSequentialCell(idb::IdbInstance* inst, idb::LibCell* lib_cell) -> bool;
auto IsClockSinkPin(idb::IdbPin* pin, idb::LibCell* lib_cell) -> bool;
auto IsMacroClockSinkPin(idb::IdbPin* pin, idb::LibCell* lib_cell) -> bool;
auto CountDirectClockSinks(const SdcLibertyCellLookup& liberty_cell_lookup, idb::IdbNet* net) -> ClockSinkStats;
auto CountDirectClockSinksForReport(const SdcLibertyCellLookup& liberty_cell_lookup, idb::IdbNet* net) -> ClockSinkStats;
auto IsClockTarget(const ClockSinkStats& stats) -> bool;
auto TargetKind(const ClockSinkStats& stats) -> std::string;
auto ClockKindName(const SdcClockDecl& clock) -> std::string;
auto MasterClockName(const SdcClockDecl& clock) -> std::string;
auto DominanceForRecord(const ClockTraceRecord& record, const std::string& clock_kind) -> std::string;
auto StrongTargetSinkThreshold(std::size_t max_fanout) -> std::size_t;
auto IsStrongClockTarget(const ClockTraceRecord& record, std::size_t sink_threshold) -> bool;
auto ResolveInstPinByLibPort(idb::IdbInstance* inst, idb::LibPort* lib_port) -> idb::IdbPin*;
auto BuildPreclusteredSinkAnchor(const SdcLibertyCellLookup& liberty_cell_lookup, idb::IdbNet* leaf_net)
    -> std::optional<ClockTracePreclusteredSinkAnchor>;

auto ObjectKindName(SdcObjectKind kind) -> std::string;
auto ResolvePortNet(idb::IdbDesign* idb_design, const std::string& port_name) -> idb::IdbNet*;
auto ResolvePinNet(idb::IdbDesign* idb_design, const std::string& pin_name) -> idb::IdbNet*;
auto ResolveRefNets(idb::IdbDesign* idb_design, const SdcObjectRef& ref) -> std::vector<idb::IdbNet*>;
auto BuildCaseConstraintSet(const SdcClockData& clock_data) -> CaseConstraintSet;
auto BuildGeneratedBoundaryOwners(idb::IdbDesign* idb_design, const SdcClockData& clock_data)
    -> std::unordered_map<idb::IdbNet*, std::string>;
auto TraceClock(const SdcLibertyCellLookup& liberty_cell_lookup, idb::IdbDesign* idb_design, const SdcClockDecl& clock,
                const CaseConstraintSet& case_constraints,
                const std::unordered_map<idb::IdbNet*, std::string>& generated_boundary_owner_by_net) -> std::vector<ClockTraceRecord>;

auto JoinNames(const std::set<std::string>& names, std::size_t display_limit = 8U) -> std::string;
auto BuildClockDeclViews(idb::IdbDesign* idb_design, const SdcClockData& clock_data) -> std::map<std::string, ClockDeclView>;
auto AnnotateRecordOwnership(ClockTraceRecord& record, const std::map<std::string, ClockDeclView>& clock_view_by_name) -> void;
auto CollectTracedNetNames(const std::vector<ClockTraceRecord>& records) -> std::set<std::string>;
auto CollectUnownedClockLikeRecords(const SdcLibertyCellLookup& liberty_cell_lookup, idb::IdbDesign* idb_design,
                                    const std::vector<ClockTraceRecord>& records) -> std::vector<ClockTraceRecord>;
auto NumberToString(std::size_t value) -> std::string;
auto EmitClockTraceReport(SchemaWriter& reporter, const std::vector<ClockTraceRecord>& records) -> void;
auto EmitSdcClockOwnershipReport(const SdcClockData& clock_data, const std::map<std::string, ClockDeclView>& clock_view_by_name,
                                 SchemaWriter& reporter, const std::vector<ClockTraceRecord>& records) -> void;
auto EmitUnownedClockLikeNetReport(SchemaWriter& reporter, const std::vector<ClockTraceRecord>& records) -> void;

}  // namespace icts::clock_trace
