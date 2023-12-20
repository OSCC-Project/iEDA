/**
 * @file MPAPI.cc
 * @author Yuezuo Liu (yuezuoliu@163.com)
 * @brief
 * @version 0.1
 * @date 2023-12-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "MPAPI.hh"

namespace imp {

std::vector<std::pair<int64_t, int64_t>> SAPlaceSeqPairInt64(int max_iters, int num_actions, double cool_rate,
                                                             const std::vector<int64_t>* pin_x_off, const std::vector<int64_t>* pin_y_off,
                                                             const std::vector<int64_t>* initial_lx, const std::vector<int64_t>* initial_ly,
                                                             const std::vector<int64_t>* dx, const std::vector<int64_t>* dy,
                                                             const std::vector<int64_t>* halo_x, const std::vector<int64_t>* halo_y,
                                                             const std::vector<size_t>* pin2vertex, const std::vector<size_t>* net_span,
                                                             int64_t region_lx, int64_t region_ly, int64_t region_dx, int64_t region_dy,
                                                             size_t num_moveable, bool pack_left, bool pack_bottom)
{
  //   typedef SeqPair<int64_t> RepresentType;
  //   typedef SAPlacement<int64_t, RepresentType> SolutionType;
  typedef int64_t CoordType;
  typedef SeqPair<CoordType> RepresentType;
  const std::vector<bool>* p1 = nullptr;
  const std::vector<std::vector<std::pair<int64_t, int64_t>>>* p2 = nullptr;
  const std::vector<std::vector<std::pair<int64_t, int64_t>>>* p3 = nullptr;
  std::vector<int64_t> solution_lx;
  std::vector<int64_t> solution_ly;
  SAPlace<CoordType, RepresentType>(solution_lx, solution_ly, max_iters, num_actions, cool_rate, pin_x_off, pin_y_off, initial_lx,
                                    initial_ly, dx, dy, halo_x, halo_y, pin2vertex, net_span, p1, p2, p3, region_lx, region_ly, region_dx,
                                    region_dy, num_moveable, pack_left, pack_bottom);
  std::vector<std::pair<int64_t, int64_t>> result;
  for (size_t i = 0; i < solution_lx.size(); ++i) {
    result.emplace_back(solution_lx[i], solution_ly[i]);
  }
  return result;
}

}  // namespace imp