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
 * @author Jianrong Su
 * @brief
 * @version 1.0
 * @date 2025-06-23
 */

#include "PdnOptimizer.hh"

#include "SimulatedAnnealing.hh"
#include "PNPConfig.hh"
#include "iPNPApi.hh"
#include "iPNP.hh"
#include "log/Log.hh"

namespace ipnp {

  PdnOptimizer::PdnOptimizer()
    : _opt_score(0.0)
  {
    PNPConfig* temp_config = new PNPConfig();
    if (temp_config->get_sa_initial_temp() && 
        temp_config->get_sa_cooling_rate() && 
        temp_config->get_sa_min_temp() && 
        temp_config->get_sa_iterations_per_temp() && 
        temp_config->get_sa_ir_drop_weight() && 
      temp_config->get_sa_overflow_weight())
    {
      _initial_temp = temp_config->get_sa_initial_temp();
      _cooling_rate = temp_config->get_sa_cooling_rate();
      _min_temp = temp_config->get_sa_min_temp();
      _iterations_per_temp = temp_config->get_sa_iterations_per_temp();
      _ir_drop_weight = temp_config->get_sa_ir_drop_weight();
      _overflow_weight = temp_config->get_sa_overflow_weight();
    } else {
      _initial_temp = 100.0;
      _cooling_rate = 0.95;
      _min_temp = 0.1;
      _iterations_per_temp = 10;
      _ir_drop_weight = 0.6;
      _overflow_weight = 0.4;
      LOG_WARNING << "iPNP instance not found, using default simulated annealing parameters.";
    }
    delete temp_config;
  }

  PdnOptimizer::~PdnOptimizer()
  {
  }


  void PdnOptimizer::optimize(GridManager initial_pdn, idb::IdbBuilder* idb_builder)
  {
    // TODO: Implement the two-stage PDN optimization process
  }

  void PdnOptimizer::optimizeGlobal(GridManager initial_pdn, idb::IdbBuilder* idb_builder)
  {
    _input_pdn_grid = initial_pdn;
    LOG_INFO << "Starting global optimization..." << std::endl;
    LOG_INFO << "Simulated Annealing parameters: ";
    LOG_INFO << "  Initial temperature: " << _initial_temp;
    LOG_INFO << "  Cooling rate: " << _cooling_rate;
    LOG_INFO << "  Minimum temperature: " << _min_temp;
    LOG_INFO << "  Iterations per temperature: " << _iterations_per_temp;
    LOG_INFO << "  IR drop weight: " << _ir_drop_weight;
    LOG_INFO << "  Overflow weight: " << _overflow_weight;

    SimulatedAnnealing sa(_initial_temp, _cooling_rate, _min_temp, _iterations_per_temp, _ir_drop_weight, _overflow_weight);

    OptimizationResult result = sa.optimize(_input_pdn_grid, idb_builder);

    _output_pdn_grid = result.best_grid;
    _opt_score = result.best_cost;
    _region_data = result.region_data;

    LOG_INFO << "Global optimization complete. Score: " << _opt_score;
  }

  void PdnOptimizer::optimizeByRegion(GridManager initial_pdn, idb::IdbBuilder* idb_builder)
  {
    // TODO: Implement region-based PDN optimization process
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
