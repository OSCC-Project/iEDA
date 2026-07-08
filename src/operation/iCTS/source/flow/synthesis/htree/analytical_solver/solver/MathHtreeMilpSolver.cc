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
 * @file MathHtreeMilpSolver.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-26
 * @brief MILP backend facade implementation for the mathematical H-tree model.
 */

#include "synthesis/htree/analytical_solver/solver/MathHtreeMilpSolver.hh"

#include <utility>

#include "synthesis/htree/analytical_solver/solver/HighsMathHtreeSolver.hh"

namespace icts::htree::analytical_solver {
namespace {

auto ToHighsOptions(const MathHtreeMilpSolverOptions& options) -> HighsMathHtreeSolverOptions
{
  return HighsMathHtreeSolverOptions{
      .anchor_time_limit_ms = options.anchor_time_limit_ms,
      .normalized_time_limit_ms = options.normalized_time_limit_ms,
  };
}

}  // namespace

MathHtreeMilpSolver::MathHtreeMilpSolver(MathHtreeMilpSolverOptions options) : _options(std::move(options))
{
}

auto MathHtreeMilpSolver::solve(const MathHtreeProblem& problem, MathHtreeObjective objective) -> MathHtreeSolution
{
  return HighsMathHtreeSolver::solve(problem, objective);
}

auto MathHtreeMilpSolver::solveNormalizedTradeoff(const MathHtreeProblem& problem) const -> MathHtreeSolution
{
  const HighsMathHtreeSolver solver(ToHighsOptions(_options));
  return solver.solveNormalizedTradeoff(problem);
}

}  // namespace icts::htree::analytical_solver
