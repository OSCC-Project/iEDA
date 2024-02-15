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

#include <algorithm>
#include <numeric>
#include <vector>

namespace imp {
template <typename T>
struct Hpwl
{
  float operator()(const T* lx, const T* ly) const
  {
    float sum{0};
    int chunk_size = std::max(int(_eptr.size() / _num_threads / 16), 1);
#pragma omp parallel for num_threads(_num_threads) schedule(dynamic, chunk_size) reduction(+ : sum)
    for (size_t i = 0; i < _eptr.size() - 1; i++) {
      size_t k = _sequence[i];
      T x_min = std::numeric_limits<T>::max();
      T x_max = std::numeric_limits<T>::lowest();
      T y_min = std::numeric_limits<T>::max();
      T y_max = std::numeric_limits<T>::lowest();
      for (size_t j = _eptr[k]; j < _eptr[k + 1]; j++) {
        T xx = lx[_eind[j]] + _lx_off[j];
        x_min = std::min(x_min, xx);
        x_max = std::max(x_max, xx);
        T yy = ly[_eind[j]] + _ly_off[j];
        y_min = std::min(y_min, yy);
        y_max = std::max(y_max, yy);
      }
      sum += _weight[k] * (x_max - x_min + y_max - y_min);
    }
    return sum;
  }

  float operator()(const std::vector<T>& lx, const std::vector<T>& ly) const { return this->operator()(lx.data(), ly.data()); }

  Hpwl(const std::vector<size_t>& eptr, const std::vector<size_t>& eind, const std::vector<T>& x_off = {}, const std::vector<T>& y_off = {},
       const std::vector<T>& weight = {}, int num_threads = 1)
      : _eptr(eptr), _eind(eind), _lx_off(x_off), _ly_off(y_off), _weight(weight), _num_threads(num_threads)
  {
    if (_lx_off.empty())
      _lx_off.resize(eind.size(), 0);
    if (_ly_off.empty())
      _ly_off.resize(eind.size(), 0);
    if (_weight.empty())
      _weight.resize(eptr.size() - 1, 1);
    std::vector<size_t> degree(eptr.size() - 1);
    for (size_t i = 0; i < degree.size(); i++) {
      degree[i] = _eptr[i + 1] - _eptr[i];
    }
    _sequence.resize(_eptr.size() - 1);
    std::iota(_sequence.begin(), _sequence.end(), 0);
    std::sort(_sequence.begin(), _sequence.end(), [&](size_t& a, size_t& b) { return degree[a] < degree[b]; });
  }

  std::vector<size_t> _eptr;
  std::vector<size_t> _eind;
  std::vector<T> _lx_off;
  std::vector<T> _ly_off;
  std::vector<T> _weight;
  int _num_threads{1};
  std::vector<size_t> _sequence;
};

}  // namespace imp

#endif