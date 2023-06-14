#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <set>

#include "analytical_method/LSEWirelength.hh"
#include "analytical_method/MProblem.hh"
#include "analytical_method/Solver.hh"
int main()
{
  int num_v = 5;
  int num_e = 5;
  int netdegree = 5;
  int core_w = 10000;
  int core_h = 10000;
  ipl::MProblem mp;
  mp.setRandom(num_v, num_e, netdegree, core_w, core_h);
  Vec x = (Vec::Random(num_v) + Vec::Ones(num_v)) * (core_w / 2);
  Vec y = (Vec::Random(num_v) + Vec::Ones(num_v)) * (core_h / 2);
  Vec r = (Vec::Random(num_v) + Vec::Ones(num_v)) * M_PI;
  // x.setConstant(core_w / 2);
  // y.setConstant(core_h / 2);
  Mat var(num_v, 3);
  var.col(0) = x;
  var.col(1) = y;
  var.col(2) = r;
  ipl::Solver::doNesterovSolve(mp, var);
  // std::cout << var;
}