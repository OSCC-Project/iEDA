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
#include <Eigen/Dense>

// #include "eigen3/Eigen/Dense"
using Eigen::MatrixXd;
using Eigen::VectorXd;
namespace ipl {

class Problem
{
 public:
  Problem(/* args */){};
  virtual ~Problem(){};
  virtual void evaluate(const MatrixXd& variable, MatrixXd& gradient, double& cost, int iter) const = 0;
  // virtual void updateParameter(VectorXf& parameter, int iter) = 0;
  // virtual MatrixXd hessianMatrix() = 0;
  virtual double getLowerBound(int row, int col) const { return std::numeric_limits<double>::lowest(); }
  virtual double getUpperBound(int row, int col) const { return std::numeric_limits<double>::max(); };
  virtual int variableMatrixRows() const = 0;
  virtual int variableMatrixcols() const = 0;

 protected:
  // MatrixXd _constant;
};

}  // namespace ipl

#endif  // IPL_MP_PROBLEM_H