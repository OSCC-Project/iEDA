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
 * @file PdnOptimizer.cpp
 * @author Xinhao li
 * @brief
 * @version 0.1
 * @date 2024-07-15
 */

#include "PdnOptimizer.hh"

#include "SimulatedAnnealing.hh"

namespace ipnp {

/**
 * @brief The entire evaluation process.
 * @attention including IR and Congestion, can add EM or be replaced by ML models in the future
 */
void PdnOptimizer::evaluate()
{
  /**
   * @todo
   */
}

/**
 * @brief The entire optimization process.
 * @attention Including calling evaluator and modify PDN by calling algorithm module.
 * Should include all optimization cycles of simulated annealing, not just one cycle.
 */
void PdnOptimizer::optimize(GridManager initial_pdn)
{
  _input_pdn_grid = initial_pdn;

  /**
   * @todo _input_pdn_grid --> Optimize --> _output_pdn_grid
   */

  _output_pdn_grid = _input_pdn_grid;
}

}  // namespace ipnp
