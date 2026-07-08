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
 * @file AnalyticalValidation.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-14
 * @brief Discrete legality validation bridge for analytical H-tree candidates.
 */

#include "synthesis/htree/analytical_solver/selection/AnalyticalValidation.hh"

#include <optional>
#include <string>
#include <utility>

#include "HTreeTopologyChar.hh"
#include "synthesis/htree/analytical_solver/candidate/AnalyticalCandidate.hh"

namespace icts::htree::analytical_solver {
namespace {

auto MakeValidationFailure(std::string reason) -> AnalyticalValidationSummary
{
  AnalyticalValidationSummary result;
  result.legal = false;
  result.failure_reason = std::move(reason);
  return result;
}

auto ValidateCandidateLegalityCheck(const AnalyticalCandidateLegalityCheck& legality_check) -> std::string
{
  if (legality_check.topology == nullptr) {
    return "missing_topology";
  }
  if (legality_check.segment_pattern_library == nullptr) {
    return "missing_segment_pattern_library";
  }
  if (legality_check.validate_sink_load_region && legality_check.sink_load_region_legality_context == nullptr) {
    return "missing_sink_load_region_legality_context";
  }
  if (legality_check.validate_root_driver_compensation && legality_check.root_driver_compensation_pass == nullptr) {
    return "missing_root_driver_compensation_pass";
  }
  return {};
}

}  // namespace

auto ValidateAnalyticalCandidate(AnalyticalCandidate& candidate, const AnalyticalCandidateLegalityCheck& legality_check)
    -> AnalyticalValidationSummary
{
  const auto legality_check_failure = ValidateCandidateLegalityCheck(legality_check);
  if (!legality_check_failure.empty()) {
    return MakeValidationFailure(legality_check_failure);
  }
  if (!candidate.isValid() || !candidate.materialized_char.has_value()) {
    return MakeValidationFailure(candidate.rejection_reason.empty() ? "invalid_analytical_candidate" : candidate.rejection_reason);
  }

  AnalyticalValidationSummary result;
  if (legality_check.validate_sink_load_region) {
    result.sink_load_region = ResolveSinkLoadRegionLegality(*legality_check.topology, candidate.materialized_char->get_pattern_id(),
                                                            candidate.topology_pattern_library, *legality_check.segment_pattern_library,
                                                            *legality_check.sink_load_region_legality_context);
    if (!result.sink_load_region.legal) {
      result.failure_reason
          = result.sink_load_region.failure_reason.empty() ? "sink_load_region_illegal" : result.sink_load_region.failure_reason;
      return result;
    }
  }

  if (legality_check.validate_root_driver_compensation) {
    result.root_driver_compensation = legality_check.root_driver_compensation_pass->evaluate(
        candidate.materialized_char->get_pattern_id(), candidate.topology_pattern_library, *legality_check.segment_pattern_library,
        *legality_check.topology);
    if (!result.root_driver_compensation.valid) {
      result.failure_reason = "root_driver_compensation_invalid";
      return result;
    }
  }

  result.legal = true;
  return result;
}

}  // namespace icts::htree::analytical_solver
