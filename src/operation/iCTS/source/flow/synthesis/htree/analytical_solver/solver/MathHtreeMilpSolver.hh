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
 * @file MathHtreeMilpSolver.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-26
 * @brief MILP backend facade for the mathematical H-tree model.
 */

#pragma once

#include "synthesis/htree/analytical_solver/model/MathHtreeModel.hh"

namespace icts::htree::analytical_solver {

struct MathHtreeMilpSolverOptions
{
  double anchor_time_limit_ms = 20000.0;
  double normalized_time_limit_ms = 20000.0;
};

class MathHtreeMilpSolver
{
 public:
  explicit MathHtreeMilpSolver(MathHtreeMilpSolverOptions options = {});

  static auto solve(const MathHtreeProblem& problem, MathHtreeObjective objective) -> MathHtreeSolution;
  auto solveNormalizedTradeoff(const MathHtreeProblem& problem) const -> MathHtreeSolution;

 private:
  MathHtreeMilpSolverOptions _options;
};

}  // namespace icts::htree::analytical_solver
