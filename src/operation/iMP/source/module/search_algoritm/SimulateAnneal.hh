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

#pragma once
#include <random>

namespace imp {

struct SimulateAnneal
{
  template <typename T, typename EvalFunction, typename ActionFunction, typename Log>
  T operator()(T solution, EvalFunction evaluate, ActionFunction action, Log log)
  {
    double temperature = inital_temperature;
    double cur_cost = evaluate(solution);
    double last_cost{0.f}, temp_cost{0.}, delta_cost{0.}, random{0.};

    std::mt19937 e1(seed == -1 ? std::random_device()() : seed);
    std::uniform_real_distribution<double> real_rand(0., 1.);
    // fast sa
    T solution_t = solution;
    for (size_t iter = 0; iter < max_iters || temperature > 1e-10; ++iter) {
      last_cost = cur_cost;
      for (size_t times = 0; times < num_perturb; ++times) {
        action(solution_t);
        temp_cost = evaluate(solution_t);
        delta_cost = temp_cost - cur_cost;
        random = real_rand(e1);
        if (exp(-delta_cost * inital_temperature / temperature) > random) {
          solution = solution_t;
          cur_cost = temp_cost;
        } else {
          solution_t = solution;
        }
      }
      std::string report = "iter: " + std::to_string(iter) + " temperature: " + std::to_string(temperature)
                           + " cost: " + std::to_string(cur_cost) + " dis: " + std::to_string(cur_cost - last_cost);
      log(report);
      temperature *= cool_rate;
    }
    return solution;
  }
  int seed{-1};
  size_t max_iters = 300;
  size_t num_perturb = 60;
  double cool_rate = 0.92;
  double inital_temperature = 1000;
};

}  // namespace imp
