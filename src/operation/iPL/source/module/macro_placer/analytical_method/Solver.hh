/**
 * @file solver.hh
 * @author Fuxing Huang(fxxhuang@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-05-16
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef IPL_MP_SOLVER_H
#define IPL_MP_SOLVER_H

#include <eigen3/Eigen/Dense>
#include <memory>
using Eigen::MatrixXf;
using Eigen::VectorXf;

namespace ipl {
class Problem;

class Solver
{
 public:
  explicit Solver(){};
  ~Solver(){};
  void doNesterovSolve(VectorXf& solution);
  void set_steplength_bound(float l, float u)
  {
    if (u > l)
      return;
    _steplength_l = l;
    _steplength_u = u;
  }

 private:
  MatrixXf _jacobi;
  VectorXf _variable;
  VectorXf _patarmeter;
  VectorXf _cost;

  float _steplength_l;
  float _steplength_u;

  std::shared_ptr<Problem> _problem;
};

}  // namespace ipl

#endif  // IPL_MP_SOLVER_H