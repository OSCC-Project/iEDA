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

  PdnOptimizer::PdnOptimizer()
    : _opt_score(0.0),
    _initial_temp(100.0),
    _cooling_rate(0.95),
    _min_temp(0.1),
    _iterations_per_temp(10),
    _ir_drop_weight(0.6),
    _overflow_weight(0.4)
  {
  }

  PdnOptimizer::~PdnOptimizer()
  {
  }


  void PdnOptimizer::optimize(GridManager initial_pdn, idb::IdbBuilder* idb_builder)
  {
    // _input_pdn_grid = initial_pdn;

    // std::cout << "Starting two-phase optimization..." << std::endl;

    // // 阶段1：全局探索 - 使用较高初始温度和较慢冷却率
    // std::cout << "Phase 1: Global exploration" << std::endl;
    // SimulatedAnnealing global_sa(_initial_temp, _cooling_rate, 1.0, _iterations_per_temp);
    // global_sa.setWeights(_ir_drop_weight, _overflow_weight);
    // OptimizationResult global_result = global_sa.optimize(_input_pdn_grid, idb_builder);

    // // 阶段2：区域精调 - 使用较低初始温度和较快冷却率
    // std::cout << "Phase 2: Regional fine-tuning" << std::endl;
    // SimulatedAnnealing regional_sa(_initial_temp * 0.2, _cooling_rate, _min_temp, _iterations_per_temp / 2);
    // regional_sa.setWeights(_ir_drop_weight, _overflow_weight);
    // OptimizationResult final_result = regional_sa.optimizeByRegion(global_result.best_grid, idb_builder);

    // _output_pdn_grid = final_result.best_grid;
    // _opt_score = final_result.best_cost;
    // _region_data = final_result.region_data;

    // std::cout << "Optimization complete. Final score: " << _opt_score << std::endl;
  }

  void PdnOptimizer::optimizeGlobal(GridManager initial_pdn, idb::IdbBuilder* idb_builder)
  {
    _input_pdn_grid = initial_pdn;

    std::cout << "Starting global optimization..." << std::endl;

    // 创建模拟退火对象
    SimulatedAnnealing sa(_initial_temp, _cooling_rate, _min_temp, _iterations_per_temp);
    sa.setWeights(_ir_drop_weight, _overflow_weight);

    // 运行全局优化
    OptimizationResult result = sa.optimize(_input_pdn_grid, idb_builder);

    _output_pdn_grid = result.best_grid;
    _opt_score = result.best_cost;
    _region_data = result.region_data;

    std::cout << "Global optimization complete. Score: " << _opt_score << std::endl;
  }

  void PdnOptimizer::optimizeByRegion(GridManager initial_pdn, idb::IdbBuilder* idb_builder)
  {
    // _input_pdn_grid = initial_pdn;

    // std::cout << "Starting region-based optimization..." << std::endl;

    // // 创建模拟退火对象
    // SimulatedAnnealing sa(_initial_temp, _cooling_rate, _min_temp, _iterations_per_temp);
    // sa.setWeights(_ir_drop_weight, _overflow_weight);

    // // 运行区域化优化
    // OptimizationResult result = sa.optimizeByRegion(_input_pdn_grid, idb_builder);

    // _output_pdn_grid = result.best_grid;
    // _opt_score = result.best_cost;
    // _region_data = result.region_data;

    // std::cout << "Region-based optimization complete. Score: " << _opt_score << std::endl;
  }

  void PdnOptimizer::setWeights(double ir_drop_weight, double overflow_weight)
  {
    _ir_drop_weight = ir_drop_weight;
    _overflow_weight = overflow_weight;
  }

  void PdnOptimizer::setAnnealingParams(double initial_temp, double cooling_rate, double min_temp, int iterations_per_temp)
  {
    _initial_temp = initial_temp;
    _cooling_rate = cooling_rate;
    _min_temp = min_temp;
    _iterations_per_temp = iterations_per_temp;
  }

}  // namespace ipnp
