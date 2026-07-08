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
 * @file AnalyticalValidation.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-14
 * @brief Discrete legality validation bridge for analytical H-tree candidates.
 */

#pragma once

#include <string>

#include "synthesis/htree/compensation/RootDriverCompensation.hh"
#include "synthesis/htree/region/SinkLoadRegion.hh"

namespace icts {
class Tree;
}  // namespace icts

namespace icts::htree {
struct BufferPatternLibrary;
}  // namespace icts::htree

namespace icts::htree::analytical_solver {

struct AnalyticalCandidate;

struct AnalyticalCandidateLegalityCheck
{
  const Tree* topology = nullptr;
  const BufferPatternLibrary* segment_pattern_library = nullptr;
  SinkLoadRegionLegalityContext* sink_load_region_legality_context = nullptr;
  RootDriverCompensationPass* root_driver_compensation_pass = nullptr;
  bool validate_sink_load_region = true;
  bool validate_root_driver_compensation = true;
};

struct AnalyticalValidationSummary
{
  bool legal = false;
  std::string failure_reason;
  SinkLoadRegionLegalitySummary sink_load_region;
  RootDriverCompensationDetail root_driver_compensation;
};

auto ValidateAnalyticalCandidate(AnalyticalCandidate& candidate, const AnalyticalCandidateLegalityCheck& legality_check)
    -> AnalyticalValidationSummary;

}  // namespace icts::htree::analytical_solver
