/**
 * @file Evaluator.hh
 * @author Fuxing Huang (fxxhuang@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-07-13
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef IMP_EVALUATOR_H
#define IMP_EVALUATOR_H

#include <vector>

namespace imp {

template <typename T>
T hpwl(int num_nets, T* x, T* y, int* nets, int* pins, T* x_off, T* y_off, int num_threads = 1);
template <typename T>
T hpwl(int num_nets, T* x, T* y, int* nets, int* pins, T* x_off, T* y_off, int num_threads, T* x_min, T* x_max, T* y_min, T* y_max);

template <typename T>
T hpwl(const std::vector<T>& pin_x, const std::vector<T>& pin_y, const std::vector<size_t>& netspan, int num_threads = 1)
{
  T sum{0};
  int chunk_size = std::max(int(netspan.size() / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size) reduction(+ : sum)
  for (size_t i = 0; i < netspan.size() - 1; i++) {
    T x_min = std::numeric_limits<T>::max();
    T x_max = std::numeric_limits<T>::lowest();
    T y_min = std::numeric_limits<T>::max();
    T y_max = std::numeric_limits<T>::lowest();
    for (size_t j = netspan[i]; j < netspan[i + 1]; j++) {
      x_max = std::max(x_max, pin_x[j]);
      x_min = std::min(x_min, pin_x[j]);
      y_max = std::max(y_max, pin_y[j]);
      y_min = std::min(y_min, pin_y[j]);
    }
    sum += x_max - x_min + y_max - y_min;
  }
  return sum;
}

}  // namespace imp

#endif