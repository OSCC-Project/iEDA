#include <Eigen/Sparse>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
// #define EIGEN_USE_BLAS
#include "analytical_method/MProblem.hh"
#include "analytical_method/Problem.hh"
#include "analytical_method/Solver.hh"
// f(x,y) = (1-x)^2 + 100(y - x^2)^2;
class Quadratic : public ipl::Problem
{
 public:
  virtual void evaluate(const MatrixXd& variable, MatrixXd& gradient, double& cost, int iter) const override
  {
    const double x = variable(0, 0);
    const double y = variable(1, 0);

    cost = (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);

    gradient(0, 0) = -2.0 * (1.0 - x) - 200.0 * (y - x * x) * 2.0 * x;
    gradient(1, 0) = 200.0 * (y - x * x);
    // gradient /= std::max(std::abs(gradient.maxCoeff()), std::abs(gradient.minCoeff()));
  }
  virtual int variableMatrixRows() const override { return 2; }
  virtual int variableMatrixcols() const override { return 1; }
};
int main()
{
  Eigen::initParallel();
  Eigen::setNbThreads(48);
  int size = 50000;
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  Eigen::SparseMatrix<double> m(size, size);
  // Eigen::SparseMatrix<double> n(size, size);
  // m.reserve(5 * size);
  std::vector<Eigen::Triplet<double>> triplets;

  for (int i = 0; i < size; i += size / 5) {
    for (int j = 0; j < size; j += 1) {
      triplets.emplace_back(i, j, 100);
    }
  }
  std::sort(triplets.begin(), triplets.end(), [](const Eigen::Triplet<double>& a, const Eigen::Triplet<double>& b) {
    return (a.row() < b.row()) || ((a.row() == b.row()) && (a.col() < b.col()));
  });
  m.setFromTriplets(triplets.begin(), triplets.end());

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Creat time= " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

  // MatrixXd m = MatrixXd::Random(10000, 10000);
  // MatrixXd n = MatrixXd::Random(10000, 10000);
  VectorXd v = VectorXd::Ones(size);
  begin = std::chrono::steady_clock::now();
  // m.unaryViewExpr([](double a) { return std::exp(a); });
  // m = m.unaryViewExpr([](double a) { return std::exp(a); });
  Eigen::SparseMatrix<double> n = m;
  Eigen::RowVectorXd x;
  for (size_t i = 0; i < 100; i++) {
    x = v.transpose() * n;
  }
  m.coeffRef(0, 0) = 200;
  n.coeffRef(0, 0) = 200;

  std::cout << "x(0,0) = " << n.coeff(0, 0) << "\n";
  end = std::chrono::steady_clock::now();
  std::cout << "Compete time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

  // m << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  // std::cout << m << std::endl;
  // VectorXd v = m.colwise().sum();
  // std::cout << v << std::endl;
  // Eigen::DiagonalMatrix<double, 3> dm = v.asDiagonal();
  // std::cout << dm.toDenseMatrix() << std::endl;

  // Eigen::SparseMatrix<double> m(5, 5);
  // ipl::Solver solver = ipl::Solver(std::shared_ptr<Quadratic>(new Quadratic()));
  // Eigen::MatrixXd var = MatrixXd::Random(2, 1);
  // var *= 100;
  // solver.doNesterovSolve(var);
  // std::cout << std::endl;
  return 0;
}