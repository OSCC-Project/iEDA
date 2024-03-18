/**
 * @file MPAPI.hh
 * @author Yuezuo Liu (yuezuoliu@163.com)
 * @brief
 * @version 0.1
 * @date 2023-12-15
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef IMP_MPAPI_H
#define IMP_MPAPI_H
// #include "../source/module/heuristic/SAPlacer.hh"
// #include "../source/module/representation/SeqPair.hh"
#include <vector>
namespace imp {

std::vector<std::pair<int64_t, int64_t>> SAPlaceSeqPairInt64(int max_iters, int num_actions, double cool_rate,
                                                             const std::vector<int64_t>* pin_x_off, const std::vector<int64_t>* pin_y_off,
                                                             const std::vector<int64_t>* initial_lx, const std::vector<int64_t>* initial_ly,
                                                             const std::vector<int64_t>* dx, const std::vector<int64_t>* dy,
                                                             const std::vector<int64_t>* halo_x, const std::vector<int64_t>* halo_y,
                                                             const std::vector<size_t>* pin2vertex, const std::vector<size_t>* net_span,
                                                             int64_t region_lx, int64_t region_ly, int64_t region_dx, int64_t region_dy,
                                                             size_t num_moveable, bool pack_left = true, bool pack_bottom = true);
}  // namespace imp

#endif