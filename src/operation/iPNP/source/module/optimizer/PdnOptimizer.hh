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
 * @file PdnOptimizer.hh
 * @author Jianrong Su
 * @brief
 * @version 1.0
 * @date 2025-06-23
 */

#pragma once

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "CongestionEval.hh"
#include "FastPlacer.hh"
#include "IREval.hh"
#include "PNPGridManager.hh"

namespace ipnp {

struct RegionData;
struct OptimizationResult;

class PdnOptMethod
{
 public:
  virtual ~PdnOptMethod() = default;
};

class SimulatedAnnealing;

class PdnOptimizer
{
 public:
  PdnOptimizer();
  ~PdnOptimizer();

  // getter
  PNPGridManager get_out_put_grid() { return _output_pdn_grid; }
  double get_opt_score() const { return _opt_score; }

  void optimize(PNPGridManager initial_pdn);

  void optimizeGlobal(PNPGridManager initial_pdn);

  void optimizeByRegion(PNPGridManager initial_pdn);

  void setWeights(double ir_drop_weight, double overflow_weight);

  void setAnnealingParams(double initial_temp, double cooling_rate, double min_temp, int iterations_per_temp);

 private:
  PNPGridManager _input_pdn_grid;
  PNPGridManager _output_pdn_grid;
  double _opt_score;
  std::vector<std::vector<std::vector<RegionData>>> _region_data;

  double _initial_temp;
  double _cooling_rate;
  double _min_temp;
  int _iterations_per_temp;

  double _ir_drop_weight;
  double _overflow_weight;
};

}  // namespace ipnp
