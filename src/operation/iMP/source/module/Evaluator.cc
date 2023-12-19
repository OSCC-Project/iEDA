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
    T x_max = std::numeric_limits<T>::lowest();
    T x_min = std::numeric_limits<T>::max();
    T y_max = std::numeric_limits<T>::lowest();
    T y_min = std::numeric_limits<T>::max();
    for (int j = nets[i]; j < nets[i + 1]; j++) {
      int xx = x[pins[j]] + x_off[j];
      x_max = std::max(x_max, xx);
      x_min = std::min(x_min, xx);
      int yy = y[pins[j]] + y_off[j];
      y_max = std::max(y_max, yy);
      y_min = std::min(y_min, yy);
    }
    sum += x_max - x_min + y_max - y_min;
  }
  return sum;
}
}  // namespace imp
