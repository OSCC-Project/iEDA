#include "Evaluator.hh"

#include <omp.h>

#include <cmath>
#include <numeric>
namespace imp {

template <typename T>
T hpwl(int num_nets, T* x, T* y, int* nets, int* pins, T* x_off, T* y_off, int num_threads)
{
  T sum{0};
  int chunk_size = std::max(int(num_nets / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size) reduction(+ : sum)
  for (int i = 0; i < num_nets; i++) {
    T x_min = std::numeric_limits<T>::max();
    T x_max = std::numeric_limits<T>::lowest();
    T y_min = std::numeric_limits<T>::max();
    T y_max = std::numeric_limits<T>::lowest();
    for (int j = nets[i]; j < nets[i + 1]; j++) {
      T xx = x[pins[j]] + x_off[j];
      x_max = std::max(x_max, xx);
      x_min = std::min(x_min, xx);
      T yy = y[pins[j]] + y_off[j];
      y_max = std::max(y_max, yy);
      y_min = std::min(y_min, yy);
    }
    sum += x_max - x_min + y_max - y_min;
  }
  return sum;
}
template <typename T>
T hpwl(int num_nets, T* x, T* y, int* nets, int* pins, T* x_off, T* y_off, int num_threads, T* x_min, T* x_max, T* y_min, T* y_max)
{
  int chunk_size = std::max(int(num_nets / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
  for (int i = 0; i < num_nets; i++) {
    x_min[i] = std::numeric_limits<T>::max();
    x_max[i] = std::numeric_limits<T>::lowest();
    y_min[i] = std::numeric_limits<T>::max();
    y_max[i] = std::numeric_limits<T>::lowest();
    for (int j = nets[i]; j < nets[i + 1]; j++) {
      T xx = x[pins[j]] + x_off[j];
      x_min[i] = std::min(x_min, xx);
      x_max[i] = std::max(x_max, xx);
      T yy = y[pins[j]] + y_off[j];
      y_min[i] = std::min(y_min, yy);
      y_max[i] = std::max(y_max, yy);
    }
  }

  T sum{0};
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size) reduction(+ : sum)
  for (int i = 0; i < num_nets; i++) {
    sum += x_max[i] - x_min[i] + y_max[i] - y_min[i];
  }

  return sum;
}
// template <typename T>
// T hpwl(const std::vector<T>& pin_x, const std::vector<T>& pin_y, const std::vector<size_t>& netspan, int num_threads)
// {
//   T sum{0};
//   int chunk_size = std::max(int(netspan.size() / num_threads / 16), 1);
// #pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size) reduction(+ : sum)
//   for (int i = 0; i < netspan.size(); i++) {
//     T x_min = std::numeric_limits<T>::max();
//     T x_max = std::numeric_limits<T>::lowest();
//     T y_min = std::numeric_limits<T>::max();
//     T y_max = std::numeric_limits<T>::lowest();
//     for (int j = netspan[i]; j < netspan[i + 1]; j++) {
//       x_max = std::max(x_max, pin_x[j]);
//       x_min = std::min(x_min, pin_x[j]);
//       y_max = std::max(y_max, pin_y[j]);
//       y_min = std::min(y_min, pin_y[j]);
//     }
//     sum += x_max - x_min + y_max - y_min;
//   }
//   return sum;
// }
}  // namespace imp
