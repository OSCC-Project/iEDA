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
 * @file SimulatedAnnealing.hh
 * @author Xinhao li
 * @brief
 * @version 0.1
 * @date 2024-07-15
 */

 // SimulatedAnnealing.hh
#pragma once

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <random>
#include <cmath>

#include "GridManager.hh"
#include "IREval.hh"
#include "CongestionEval.hh"

namespace idb {
  class IdbBuilder;
}

namespace ipnp {
  

  struct RegionData {
    double ir_drop;
    int32_t overflow;
  };

  struct OptimizationResult {
    GridManager best_grid;
    double best_cost;
    std::vector<std::vector<std::vector<RegionData>>> region_data; // [layer][row][col]
  };

  struct CostResult {
    double cost;
    double ir_drop;
    int32_t overflow;
    double normalized_ir_drop;
    double normalized_overflow;
  };

  class SimulatedAnnealing
  {
  public:
    SimulatedAnnealing(double initial_temp,
      double cooling_rate,
      double min_temp,
      int iterations_per_temp,
      double ir_drop_weight,
      double overflow_weight);
    ~SimulatedAnnealing() = default;

    
    OptimizationResult optimize(const GridManager& initial_grid, idb::IdbBuilder* idb_builder);
    CostResult evaluateCost(const GridManager& new_solution, const GridManager& current_solution, idb::IdbBuilder* idb_builder);

  private:
    CongestionEval _cong_eval;
    IREval _ir_eval;

    double _initial_temp;
    double _cooling_rate;
    double _min_temp;
    int _iterations_per_temp;
    int _max_no_improvement;

    std::mt19937 _rng;

    double _ir_drop_weight;
    double _overflow_weight;

    double _max_ir_drop;
    double _min_ir_drop;
    int32_t _max_overflow;
    int32_t _min_overflow;

    GridManager generateNeighbor(const GridManager& current);
    bool acceptSolution(double current_cost, double new_cost, double temp);
    bool shouldTerminate(int iterations, double temp, int no_improvement_count);
    double normalizeIRDrop(double max_ir_drop, double min_ir_drop, double avg_ir_drop);
    double normalizeOverflow(int32_t overflow);
    bool isSameTemplate(const SingleTemplate& t1, const SingleTemplate& t2);
  };

}  // namespace ipnp
