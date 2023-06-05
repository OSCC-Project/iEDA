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
  virtual void evaluate(const Mat& variable, Mat& gradient, double& cost, int iter) const override
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
  Eigen::setNbThreads(1);
  int size = 5;
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  Eigen::SparseMatrix<double> m(size, size);
  // Eigen::SparseMatrix<double> n(size, size);
  // m.reserve(5 * size);
  std::vector<Eigen::Triplet<double>> triplets;

  for (int i = 1; i < size; i += size / 5) {
    for (int j = 0; j < size; j += 1) {
      triplets.emplace_back(i, j, 2);
    }
  }
  std::sort(triplets.begin(), triplets.end(), [](const Eigen::Triplet<double>& a, const Eigen::Triplet<double>& b) {
    return (a.row() < b.row()) || ((a.row() == b.row()) && (a.col() < b.col()));
  });
  m.setFromTriplets(triplets.begin(), triplets.end());

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Creat time= " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

  // Mat m = Mat::Random(10000, 10000);
  // Mat n = Mat::Random(10000, 10000);
  Eigen::RowVectorXd v = Eigen::RowVectorXd::Ones(size);
  Mat x = Mat::Ones(2, 2);
  Mat y = Mat::Ones(2, 2);
  x(0, 0) = 0;
  Mat a = Mat::Ones(1000, 1000);
  Mat b = Mat::Ones(1000, 1000);
  // Mat c;
  begin = std::chrono::steady_clock::now();

  // for (size_t i = 0; i < 1; i++) {
  // auto a = x.unaryExpr([](double val) { return val * val; });
  // auto b = x.unaryExpr([](double val) { return val * val; });
  // b += a;
  // Mat c;
  auto c = a * b;
  // z.eval();
  // std::cout << m.unaryExpr([](double val) { return std::exp(val); }).toDense() << "\n";
  // }

  // m.unaryViewExpr([](double a) { return std::exp(a); });
  // Eigen::SparseMatrix<double> n;
  // v = Eigen::RowVectorXd::Ones(size) * (m / -1).unaryExpr([](double a) { return a * a; });
  // Eigen::VectorXd x;
  // for (size_t i = 0; i < 1000; i++) {
  //   v = v * m;
  // }

  // std::cout << "sum = " << v.sum();

  // x = VectorXd::Ones(size).transpose() * x;
  // m.coeffRef(0, 0) = 0;
  // m.unaryExpr([](double x) { return (x == 0) ? 0 : 1 / x; });
  // std::cout << "n = \n" << m.toDense() << "\n";
  // std::cout << "x(0,0) = " << n.coeff(0, 0) << "\n";
  c.eval();
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
  // Eigen::Mat var = Mat::Random(2, 1);
  // var *= 100;
  // solver.doNesterovSolve(var);
  // std::cout << std::endl;
  return 0;
}