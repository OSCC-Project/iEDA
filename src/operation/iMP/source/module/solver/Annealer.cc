#include "Annealer.hh"

#include "Logger.hpp"
namespace imp {
bool SimulateAnneal::solve(SASolution& solution, const SAOption& opt)
{
  double cool_rate = opt.cool_rate;
  double temperature = opt.start_temperature;

  double curr_cost = solution.evaluate();
  double temp_cost{0.}, delta_cost{0.}, random{0.};
  std::random_device r;
  std::default_random_engine e1(r());
  std::uniform_real_distribution<double> real_rand(0., 1.);
  // fast sa
  for (int iter = 0; iter < opt.max_iters; ++iter) {
    for (int times = 0; times < opt.num_operates; ++times) {
      solution.operate();
      temp_cost = solution.evaluate();
      delta_cost = temp_cost - curr_cost;
      random = real_rand(e1);
      if (delta_cost < 0 || exp(-delta_cost / temperature) > random) {
        solution.update();
        curr_cost = temp_cost;
      } else {
        solution.rollback();
      }
    }
    temperature *= cool_rate;
  }
  return true;
}
}  // namespace imp
