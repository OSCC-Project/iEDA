#include <Eigen/Sparse>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
// #define EIGEN_USE_BLAS
#include <omp.h>

#include <cstdlib>
#include <ctime>

#include "analytical_method/LSEWirelength.hh"

int main()
{
  int v_num = 5;
  int e_num = 5;
  Triplets<pair<double, double>> mv_pin_off;
  Triplets<pair<double, double>> fix_pin_loc;
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < e_num; j++) {
      mv_pin_off.emplace_back(j + i, j, std::make_pair(10.0, -10.0));
    }
  }
  for (size_t i = 0; i < e_num; i++) {
    fix_pin_loc.emplace_back(0, i, std::make_pair(10.0, 10.0));
    fix_pin_loc.emplace_back(1, i, std::make_pair(10.0, 10.0));
  }
  ipl::LSEWirelength lse;
  lse.setConstant(std::move(mv_pin_off), fix_pin_loc);
}