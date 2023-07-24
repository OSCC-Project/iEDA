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
#include "SimulateAnneal.hh"

namespace ipl {

SimulateAnneal::SimulateAnneal(SAParam* param, Evaluation* evaluation)
{
  _param = param;
  _evaluation = evaluation;
  _solution = _evaluation->get_solution();
}

void SimulateAnneal::runAnneal()
{
  // option
  uint32_t max_num_step = _param->get_max_num_step();
  uint32_t perturb_per_step = _param->get_perturb_per_step();
  float cool_rate = _param->get_cool_rate();
  float temperature = _param->get_init_temperature();

  float curr_cost = _evaluation->evaluate();
  float temp_cost, delta_cost, random;

  // fast sa
  uint32_t step = 1;
  while (step < max_num_step) {
    for (uint32_t i = 0; i < perturb_per_step; ++i) {
      _solution->perturb();
      temp_cost = _evaluation->evaluate();
      delta_cost = temp_cost - curr_cost;
      random = rand() % 10000;
      random /= 10000;
      if (delta_cost < 0 || exp(-delta_cost / temperature) > random) {
        _solution->update();
        curr_cost = temp_cost;
      } else {
        _solution->rollback();
      }
    }
    step++;
    temperature *= cool_rate;
    _evaluation->showMassage();
  }
  _solution->pack();
  _evaluation->showMassage();
}

}  // namespace ipl