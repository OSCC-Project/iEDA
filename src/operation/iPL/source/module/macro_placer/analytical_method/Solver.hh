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
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace ipl {
class Problem;

class Solver
{
 public:
  explicit Solver(std::shared_ptr<Problem> problem) : _problem(problem){};
  ~Solver(){};
  void doNesterovSolve(MatrixXd& solution);
  void set_steplength_bound(float l, float u)
  {
    if (u > l)
      return;
    _steplength_l = l;
    _steplength_u = u;
  }

 private:
  MatrixXd _gradiant;

  float _steplength_l;
  float _steplength_u;

  std::shared_ptr<Problem> _problem;
};

}  // namespace ipl

#endif  // IPL_MP_SOLVER_H