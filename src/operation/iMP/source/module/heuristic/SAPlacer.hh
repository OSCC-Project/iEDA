/**
 * @file SAPlacer.hh
 * @author Yuezuo Liu (yuezuoliu@163.com)
 * @brief
 * @version 0.1
 * @date 2023-12-1
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef IMP_SAPlacer_H
#define IMP_SAPlacer_H
#include <limits>

#include "../../solver/Annealer.hh"
#include "SAAction.hh"
#include "SAEvaluateWl.hh"
#include "SAPlacement.hh"
namespace imp {

template <typename CoordType, typename RepresentType>
std::vector<SAPlacement<CoordType, RepresentType>> SAPlace(
    std::vector<CoordType>& solution_lx, std::vector<CoordType>& solution_ly, int max_iters, int num_actions, double cool_rate,
    const std::vector<CoordType>* pin_x_off, const std::vector<CoordType>* pin_y_off, const std::vector<CoordType>* initial_lx,
    const std::vector<CoordType>* initial_ly, const std::vector<CoordType>* dx, const std::vector<CoordType>* dy,
    const std::vector<CoordType>* halo_x, const std::vector<CoordType>* halo_y, const std::vector<size_t>* pin2vertex,
    const std::vector<size_t>* net_span, const std::vector<bool>* is_shape_discrete,
    const std::vector<std::vector<std::pair<CoordType, CoordType>>>* possible_shape_width,
    const std::vector<std::vector<std::pair<CoordType, CoordType>>>* possible_shape_height, CoordType region_lx, CoordType region_ly,
    CoordType region_dx, CoordType region_dy, size_t num_moveable, bool pack_left = true, bool pack_bottom = true)
{
  typedef SAPlacement<CoordType, RepresentType> SolutionType;
  RepresentType represent(num_moveable);
  SolutionType placement(represent, pin_x_off, pin_y_off, initial_lx, initial_ly, dx, dy, halo_x, halo_y, pin2vertex, net_span,
                         is_shape_discrete, possible_shape_width, possible_shape_height, region_lx, region_ly, region_dx, region_dy,
                         num_moveable, pack_left, pack_bottom);
  auto eval = SAEvaluateWl<CoordType, RepresentType>(placement);
  auto act = SAAction<CoordType, RepresentType>();

  double init_prob = 0.99;
  double init_temp = (-1.0) * eval.get_cost_delta_avg() / std::log(init_prob);

  std::function<double(SolutionType&)> evaluate = eval;
  std::function<void(SolutionType&)> action = act;
  auto sa_task = SATask<SolutionType>(placement, eval, act, max_iters, num_actions, cool_rate, init_temp);
  auto history = SASolve(sa_task);

  double min_cost = std::numeric_limits<double>::min(), cost;
  size_t best_idx = -1;
  for (size_t i = 0; i < history.size(); ++i) {
    cost = history[i].get_cost();
    if (cost < min_cost) {
      min_cost = cost;
      best_idx = i;
    }
  }
  solution_lx = *initial_lx;
  solution_ly = *initial_ly;
  history[best_idx].packing(solution_lx, solution_ly);
  return history;
}

}  // namespace imp

#endif