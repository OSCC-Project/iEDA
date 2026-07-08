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
 * @file SolutionFinalizer.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-26
 * @brief Shared H-tree selected-solution finalization contract.
 */

#pragma once

#include <optional>
#include <string>

#include "logger/Schema.hh"
#include "synthesis/htree/compensation/RootDriverCompensation.hh"
#include "synthesis/htree/plan/DepthPlan.hh"
#include "synthesis/htree/topology_pruning/TopologyPruning.hh"

namespace icts::htree {

struct HTreeSynthesisState;

enum class HTreeSelectionEngine
{
  kDiscrete,
  kAnalytical,
};

auto ToStageValue(HTreeSelectionEngine engine) -> std::string;

struct HTreeSelectedSolution
{
  HTreeSelectionEngine engine = HTreeSelectionEngine::kDiscrete;
  CandidateBuildEvaluation evaluation;
  DepthSummary summary;
  RootDriverCompensationStats compensation_stats;
  RootDriverCompensationDetail compensation_detail;
  std::string root_driver_clock_period_source;
  bool used_boundary_relaxation = false;
  std::string boundary_relaxation_reason;
  std::optional<double> boundary_relaxation_score = std::nullopt;
};

struct HTreeSelectionBuild
{
  bool selected = false;
  std::string failure_reason;
  HTreeSelectionEngine engine = HTreeSelectionEngine::kDiscrete;
  HTreeSelectedSolution selected_solution;
};

auto FinalizeSelectedHTreeSolution(HTreeSynthesisState& state, SchemaWriter::StageScope& build_stage,
                                   const HTreeSelectedSolution& selected_solution) -> bool;

}  // namespace icts::htree
