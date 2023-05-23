#include "analytical_method/Problem.hh"
#include "analytical_method/Solver.hh"
// f(x,y) = (1-x)^2 + 100(y - x^2)^2;
class Rosenbrock final : public ipl::Problem
{
 public:
  void evaluate(const MatrixXf& variable, MatrixXf& gradient, float& cost, int iter) const override
  {
    const double x = variable(0, 0);
    const double y = variable(1, 0);

    cost = (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);

    gradient(0, 0) = -2.0 * (1.0 - x) - 200.0 * (y - x * x) * 2.0 * x;
    gradient(1, 0) = 200.0 * (y - x * x);
  }
  int variableMatrixRows() const override { return 2; }
  int variableMatrixcols() const override { return 1; }
};
int main()
{
  ipl::Solver slover = ipl::Solver(std::shared_ptr<Rosenbrock>(new Rosenbrock()));
  Eigen::MatrixXf var(2, 1);
  var(0, 0) = -1.2;
  var(1, 0) = 1.0;
  slover.doNesterovSolve(var);
}