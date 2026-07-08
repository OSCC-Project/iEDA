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
 * @file IdbConversion.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-30
 * @brief CTS iDB conversion entry implementation.
 */

#include "instantiation/idb_conversion/IdbConversion.hh"

#include <glog/logging.h>

#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "Log.hh"
#include "design/Design.hh"
#include "io/Wrapper.hh"
#include "logger/Schema.hh"

namespace icts {

auto IdbConversion::run(const IdbConversionInput& input) -> IdbConversionSummary
{
  LOG_FATAL_IF(input.design == nullptr) << "iDB conversion requires a design.";
  LOG_FATAL_IF(input.wrapper == nullptr) << "iDB conversion requires a wrapper.";
  LOG_FATAL_IF(input.reporter == nullptr) << "iDB conversion requires a reporter.";
  auto& design = *input.design;
  auto& wrapper = *input.wrapper;
  auto& reporter = *input.reporter;
  (void) wrapper;

  auto runtime = reporter.beginRuntimeMetric("instantiation");
  auto instantiation_stage = reporter.beginStage("Instantiation", "Instantiate synthesized CTS topology into iDB", {},
                                                 StageReportOptions{.emit_success_summary = false});
  reporter.emitSection("### iDB Conversion");

  auto clocks = design.get_clocks();
  IdbConversionSummary summary;
  summary.attempted = true;
  summary.design_ready = wrapper.is_design_ready();
  summary.clock_count = clocks.size();

  WrapperWriteSummary write_summary;
  if (summary.design_ready) {
    write_summary = wrapper.writeClocksDetailed(design, reporter, clocks);
    summary.success = write_summary.success;
  } else {
    summary.failure_reason = "design_not_ready";
  }
  if (!summary.success) {
    summary.failed_clock = write_summary.failed_clock;
    summary.failed_net = write_summary.failed_net;
    summary.idb_clock_tree_restored = write_summary.idb_clock_tree_restored;
    if (!write_summary.reason.empty()) {
      summary.failure_reason = write_summary.reason;
    }
  }

  const std::string status = summary.success ? "finished" : "failed";
  KeyValueFields overview_fields = {
      {"semantic_owner", "instantiation"},
      {"status", status},
      {"design_ready", summary.design_ready ? "true" : "false"},
      {"clock_count", std::to_string(summary.clock_count)},
  };
  if (!summary.success) {
    overview_fields.emplace_back("failed_clock", summary.failed_clock.empty() ? "n/a" : summary.failed_clock);
    overview_fields.emplace_back("failed_net", summary.failed_net.empty() ? "n/a" : summary.failed_net);
    overview_fields.emplace_back("idb_clock_tree_restored", summary.idb_clock_tree_restored ? "true" : "false");
    overview_fields.emplace_back("failure_reason", summary.failure_reason);
  }
  EmitKeyValueTable(reporter, "CTS Instantiation Overview", overview_fields);

  if (summary.success) {
    (void) runtime.finished();
    instantiation_stage.finished({{"clock_count", std::to_string(summary.clock_count)}});
  } else {
    (void) runtime.failed();
    instantiation_stage.failed({{"clock_count", std::to_string(summary.clock_count)}});
  }
  return summary;
}

}  // namespace icts
