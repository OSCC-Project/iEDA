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
 * @file SdcClockReader.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-15
 * @brief Side-effect-free SDC clock subset reader for iCTS.
 */

#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

namespace idb {
class IdbDesign;
}  // namespace idb

namespace idb {
class LibCell;
}  // namespace idb

namespace icts {

class SchemaWriter;

using SdcLibertyCellLookup = std::function<idb::LibCell*(const std::string&)>;

enum class SdcObjectKind
{
  kPort,
  kPin,
  kNet,
  kClock,
  kUnknown,
};

struct SdcObjectRef
{
  SdcObjectKind kind = SdcObjectKind::kUnknown;
  std::string pattern;
  bool from_collection_cmd = false;
};

struct SdcClockDecl
{
  enum class Kind
  {
    kPrimary,
    kGenerated,
  };

  Kind kind = Kind::kPrimary;
  std::string clock_name;
  std::vector<SdcObjectRef> targets;
  std::vector<SdcObjectRef> generated_sources;
  std::string master_clock_name;
  double period_ns = 0.0;
  bool period_resolved = false;
  int divide_by = 1;
  int multiply_by = 1;
  bool invert = false;
  bool is_virtual = false;
};

struct SdcCaseAnalysis
{
  int value = 0;
  std::vector<SdcObjectRef> objects;
};

struct SdcClockData
{
  std::vector<SdcClockDecl> clocks;
  std::vector<SdcCaseAnalysis> case_analyses;
  std::vector<std::string> diagnostics;
};

struct ClockTraceRecord
{
  std::string clock_name;
  std::string net_name;
  std::string status;
  std::string target_kind;
  std::size_t sequential_clock_sinks = 0U;
  std::size_t macro_clock_sinks = 0U;
  std::string trace_path;
  std::string reason;
  std::string clock_kind = "unknown";
  std::string master_clock_name = "n/a";
  std::string dominance = "undetermined";
};

struct ClockTracePreclusteredSinkAnchor
{
  std::string leaf_net_name;
  std::string driver_inst_name;
  std::string input_pin_name;
  std::string output_pin_name;
  std::string input_net_name;
};

struct ClockTraceClockTarget
{
  std::string clock_name;
  std::string clock_net_name;
  bool preclustered_sink_reuse = false;
  std::vector<ClockTracePreclusteredSinkAnchor> preclustered_sink_anchors;
};

struct ClockTraceOutput
{
  std::vector<ClockTraceClockTarget> clock_targets;
};

struct ClockTraceSummary
{
  std::vector<ClockTraceRecord> records;
  std::vector<ClockTraceRecord> unowned_clock_like_records;
};

struct ClockTraceBuild
{
  ClockTraceOutput output;
  ClockTraceSummary summary;
};

struct SdcClockTraceInput
{
  const SdcClockData* clock_data = nullptr;
  std::size_t max_fanout = 0U;
  SchemaWriter* reporter = nullptr;
};

class SdcClockReader
{
 public:
  SdcClockReader();
  explicit SdcClockReader(std::string sdc_path);

  auto readClockData(SchemaWriter& reporter) const -> SdcClockData;
  auto readDeclarationsOnly(SchemaWriter& reporter) const -> std::vector<std::tuple<std::string, std::string, double, bool>>;
  static auto traceClockTargets(const SdcClockData& clock_data, idb::IdbDesign* idb_design, const SdcLibertyCellLookup& liberty_cell_lookup,
                                std::size_t max_fanout, SchemaWriter& reporter) -> ClockTraceBuild;

 private:
  std::string _sdc_path;
};

}  // namespace icts
