#include "SimulateAnneal.hh"

#include <math.h>

namespace ipl {

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