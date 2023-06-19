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
#include "utility/Image.hh"
int main()
{
  int num_v = 10;
  int num_e = 10;
  int netdegree = 5;
  int core_w = 10000;
  int core_h = 10000;
  // ipl::Image img(core_w, core_h, num_v);
  // img.drawText("34213412341241")
  // img.save("/home/huangfuxing/Prog_cpp/iEDA/bin/test.jpg");
  // return 0;
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
  ipl::Option opt;
  opt.max_iter = 100;
  ipl::Solver::solve(mp, var, opt);
  // std::cout << var;
}