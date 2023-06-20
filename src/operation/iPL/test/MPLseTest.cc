#include <omp.h>

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
#include "utility/Geometry.hh"
#include "utility/Image.hh"
int main()
{
  ipl::Line<double> line({0, 0}, {1, 1});

  ipl::Ploygon<double> subject_polygon{{50, 150}, {200, 50}, {350, 150}, {350, 300}};
  ipl::Ploygon<double> clip_polygon{{100, 100}, {300, 100}, {300, 300}, {100, 300}};

  auto c = subject_polygon & clip_polygon;
  std::cout << c.area();
  // std::cout << line.isLeftPoint({0.4, 0.5});
  // std::cout << line.isOnlinePoint({0.5, 0.5});

  // int _num_var = 100;
  // double ar = std::round(2.5);
  // int _orig = std::pow(2, static_cast<int>(std::ceil(std::log2(std::sqrt(9 * _num_var)))));
  // int _num_bins_x = std::pow(2, static_cast<int>(std::ceil(std::log2(std::sqrt(9 * _num_var / ar)))));
  // int _num_bins_y = std::pow(2, static_cast<int>(std::ceil(std::log2(std::sqrt(9 * _num_var * ar)))));
  // std::cout << _orig << " " << _num_bins_x << " " << _num_bins_y;
  // // return 0;
  // int num_v = 10;
  // int num_e = 10;
  // int netdegree = 5;
  // int core_w = 100;
  // int core_h = 100;
  // // ipl::Image img(core_w, core_h, num_v);
  // // img.drawText("34213412341241")
  // // img.save("/home/huangfuxing/Prog_cpp/iEDA/bin/test.jpg");
  // // return 0;
  // ipl::MProblem mp;
  // mp.setRandom(num_v, num_e, netdegree, core_w, core_h);
  // Vec x = (Vec::Random(num_v) + Vec::Ones(num_v)) * (core_w / 2);
  // Vec y = (Vec::Random(num_v) + Vec::Ones(num_v)) * (core_h / 2);
  // Vec r = (Vec::Random(num_v) + Vec::Ones(num_v)) * M_PI;
  // // Vec r = (Vec::Random(num_v) + Vec::Ones(num_v)) * 0;
  // x.setConstant(core_w / 2);
  // y.setConstant(core_h / 2);
  // for (size_t i = 0; i < r.rows(); i++) {
  //   r(i) = (i % 4) * M_PI_2;
  // }

  // Mat var(num_v, 3);
  // var.col(0) = x;
  // var.col(1) = y;
  // var.col(2) = r;
  // ipl::Option opt;
  // opt.max_iter = 100;
  // ipl::Solver::solve(mp, var, opt);
  // // std::cout << var;
}