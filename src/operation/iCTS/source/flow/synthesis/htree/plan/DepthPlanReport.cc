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
 * @file DepthPlanReport.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief H-tree depth candidate report helpers.
 */

#include <string>
#include <vector>

#include "LogFormat.hh"
#include "logger/Schema.hh"
#include "synthesis/htree/plan/DepthPlan.hh"

namespace icts::htree {
namespace {

auto FormatDepthCandidateStatus(const DepthSummary& summary) -> std::string
{
  if (summary.selected) {
    return "selected";
  }
  if (summary.success) {
    return summary.used_boundary_relaxation ? "relaxed_boundary" : "feasible";
  }
  return "failed";
}

}  // namespace

auto EmitDepthCandidateSummary(SchemaWriter& reporter, const std::vector<DepthSummary>& depth_summaries) -> void
{
  if (depth_summaries.empty()) {
    return;
  }

  TableRows rows;
  rows.reserve(depth_summaries.size());
  for (const auto& summary : depth_summaries) {
    rows.push_back({
        std::to_string(summary.depth),
        std::to_string(summary.leaf_count),
        FormatDepthCandidateStatus(summary),
        std::to_string(summary.final_frontier_count),
        std::to_string(summary.feasible_frontier_entry_count),
        std::to_string(summary.candidate_frontier_entry_count),
        summary.used_boundary_relaxation ? "true" : "false",
        summary.selected_delay_ns > 0.0 ? logformat::FormatWithUnit(summary.selected_delay_ns, "ns") : "n/a",
        summary.selected_power_w > 0.0 ? logformat::FormatPowerW(summary.selected_power_w) : "n/a",
        summary.failure_reason.empty() ? "none" : summary.failure_reason,
    });
  }

  reporter.emitTableTo("HTree Depth Candidate Summary",
                       {"Depth", "Leaves", "Status", "Final Frontier", "Feasible Entries", "Candidate Entries", "Boundary Relaxation",
                        "Best Delay", "Best Power", "Failure"},
                       rows, ReportSink::kBoth);
}

}  // namespace icts::htree
