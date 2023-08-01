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

#include <Eigen/Dense>
using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

namespace ipl {
class Problem;

enum GradientDescentMethod
{
  kNesterov = 0,
  kConjugate
};

struct Option
{
  GradientDescentMethod grad_decent_method = kNesterov;
  bool hasBoundConstraint = false;
  size_t max_iter = 1000;
  size_t threads = 1;
};

class Solver
{
 public:
  explicit Solver(){};
  ~Solver(){};
  static void solve(Problem& problem, Mat& solution, const Option& opt = Option());
};

}  // namespace ipl

#endif  // IPL_MP_SOLVER_H