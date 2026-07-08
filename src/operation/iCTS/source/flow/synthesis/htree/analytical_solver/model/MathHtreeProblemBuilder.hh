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
 * @file MathHtreeProblemBuilder.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-25
 * @brief Adapter from analytical H-tree solve problems to mathematical slot-choice models.
 */

#pragma once

#include <string>
#include <vector>

#include "PatternId.hh"
#include "synthesis/htree/analytical_solver/model/MathHtreeModel.hh"

namespace icts::htree::analytical_solver {
struct AnalyticalHTreeSolveProblem;
}  // namespace icts::htree::analytical_solver

namespace icts::htree::analytical_solver {

struct MathHtreeSlotChoiceRef
{
  PatternId unit_pattern_id = PatternId::segment(0U);
};

struct MathHtreeProblemBuildResult
{
  bool success = false;
  std::string failure_reason;
  MathHtreeProblem problem;
  std::vector<std::vector<MathHtreeSlotChoiceRef>> choice_refs_by_slot;
  std::vector<unsigned> level_slot_counts;
};

auto BuildMathHtreeProblem(const analytical_solver::AnalyticalHTreeSolveProblem& solve_problem) -> MathHtreeProblemBuildResult;

}  // namespace icts::htree::analytical_solver
