#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <set>

#include "analytical_method/LSEWirelength.hh"
int main()
{
  size_t v_num = 5;
  size_t e_num = 5;
  vector<Triplet<double>> moveable_x_offset;
  vector<Triplet<double>> moveable_y_offset;
  vector<Triplet<double>> fixed_x_location;
  vector<Triplet<double>> fixed_y_location;
  for (size_t j = 0; j < e_num; j++) {
    for (size_t i = 0; i < 2; i++) {
      moveable_x_offset.emplace_back((j + i) % v_num, j, 0.0);
      moveable_y_offset.emplace_back((j + i) % v_num, j, 0.0);
    }
  }
  for (size_t i = 0; i < e_num; i++) {
    fixed_x_location.emplace_back(0, i, 0);
    fixed_y_location.emplace_back(0, i, 0);
    fixed_x_location.emplace_back(1, i, 10);
    fixed_y_location.emplace_back(1, i, 0);
  }
  Vec x = (Vec::Random(v_num) + Vec::Ones(v_num)) * 10000;
  Vec y = (Vec::Random(v_num) + Vec::Ones(v_num)) * 0;
  Vec r = (Vec::Random(v_num) + Vec::Ones(v_num)) * 0;
  Mat var(v_num, 3);
  Mat grad(v_num, 3);
  var.col(0) = x;
  var.col(1) = y;
  var.col(2) = r;
  ipl::LSEWirelength lse(v_num, e_num);
  lse.setConstant(moveable_x_offset, moveable_y_offset, fixed_x_location, fixed_y_location);

  std::cout << "Start...\n";
  int times = 1;
  double hpwl = 0;
  Eigen::VectorXi time(times);
  for (int i = 0; i < times; i++) {
    auto begin = std::chrono::steady_clock::now();
    lse.evaluate(var, grad, hpwl, 100);
    auto end = std::chrono::steady_clock::now();
    std::cout << "hpwl = " << hpwl << "\n";
    std::cout << "Gradient 2-norm: " << grad.norm() << std::endl;
    auto time_i = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    time(i) = time_i;
  }
  std::cout << grad << std::endl;
  std::cout << "min/avg/max = " << time.minCoeff() << "/" << time.sum() / times << "/" << time.maxCoeff() << " ms" << std::endl;

  // std::cout << grad << std::endl;
}