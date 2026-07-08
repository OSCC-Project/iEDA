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
 * @file DiscreteSolution.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-26
 * @brief Discrete H-tree selected-solution search contract.
 */

#pragma once

#include "synthesis/htree/solution/finalization/SolutionFinalizer.hh"

namespace icts {

namespace htree {
struct HTreeSynthesisState;
}  // namespace htree

namespace htree::discrete_solution {

auto SelectDiscreteHTreeSolution(HTreeSynthesisState& state) -> HTreeSelectionBuild;

}  // namespace htree::discrete_solution
}  // namespace icts
