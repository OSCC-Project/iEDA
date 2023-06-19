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

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
// #include "eigen3/Eigen/Dense"
namespace ipl {

class Problem
{
 public:
  Problem(/* args */){};
  virtual ~Problem(){};
  virtual void evaluate(const Mat& variable, Mat& gradient, double& cost, int iter) const = 0;
  virtual double getSolutionDistance(const Vec& a, const Vec& b, int col) const = 0;
  virtual double getLowerBound(int row, int col) const { return std::numeric_limits<double>::lowest(); }
  virtual double getUpperBound(int row, int col) const { return std::numeric_limits<double>::max(); };
  virtual int variableMatrixRows() const = 0;
  virtual int variableMatrixcols() const = 0;

 protected:
  // Mat _constant;
};

}  // namespace ipl

#endif  // IPL_MP_PROBLEM_H