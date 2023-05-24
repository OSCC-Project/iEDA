#include <iostream>

#include "analytical_method/Problem.hh"
#include "analytical_method/Solver.hh"
// f(x,y) = (1-x)^2 + 100(y - x^2)^2;
class Quadratic : public ipl::Problem
{
 public:
  void evaluate(const MatrixXd& variable, MatrixXd& gradient, float& cost, int iter) const override
  {
    const double x = variable(0, 0);
    const double y = variable(1, 0);

    cost = (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);

    gradient(0, 0) = -2.0 * (1.0 - x) - 200.0 * (y - x * x) * 2.0 * x;
    gradient(1, 0) = 200.0 * (y - x * x);
    // gradient /= std::abs(gradient.maxCoeff());
  }
  int variableMatrixRows() const override { return 2; }
  int variableMatrixcols() const override { return 1; }
};
int main()
{
  ipl::Solver slover = ipl::Solver(std::shared_ptr<Quadratic>(new Quadratic()));
  Eigen::MatrixXd var = MatrixXd::Random(2, 1);
  // var(0, 0) = 8.59414;
  // var(1, 0) = 73.8475;
  var *= 1000;
  slover.doNesterovSolve(var);
  std::cout << std::endl;
  // slover.doNesterovSolve(var);
}