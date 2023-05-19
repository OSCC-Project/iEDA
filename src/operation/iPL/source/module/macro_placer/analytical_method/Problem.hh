/**
 * @file Problem.hh
 * @author Fuxing Huang (fxxhuang@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-05-16
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef IPL_MP_PROBLEM_H
#define IPL_MP_PROBLEM_H
#include "eigen3/Eigen/Dense"
namespace ipl {

class Problem
{
 public:
  Problem(/* args */){};
  virtual ~Problem(){};
  virtual void evaluate(const Eigen::VectorXf& variable, Eigen::MatrixXf& jacobi, Eigen::VectorXf& cost) = 0;
  virtual int numVariable() = 0;

 protected:
  Eigen::MatrixXd _constant;
};

}  // namespace ipl

#endif  // IPL_MP_PROBLEM_H