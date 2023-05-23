#include "Solver.hh"

#include <algorithm>
#include <iostream>

#include "Problem.hh"
namespace ipl {
void Solver::doNesterovSolve(MatrixXf& solution)
{
  int rows = _problem->variableMatrixRows();
  int cols = _problem->variableMatrixcols();
  if (solution.rows() != rows || solution.cols() != cols) {
    std::cerr << "Invalid start solution!\n";
    return;
  }

  // Major solution u_k
  MatrixXf major = std::move(solution);
  MatrixXf new_major = MatrixXf::Zero(rows, cols);

  // Reference solution v_k
  MatrixXf reference = major;
  MatrixXf prev_reference = MatrixXf::Random(rows, cols);

  // Gradiant of Reference solution v_k
  MatrixXf grad = MatrixXf::Zero(rows, cols);
  MatrixXf prev_grad = MatrixXf::Zero(rows, cols);

  for (Eigen::Index i = 0; i < prev_reference.rows(); i++) {
    for (Eigen::Index j = 0; j < prev_reference.cols(); j++) {
      float coeff = prev_reference(i, j);
      float l = _problem->getLowerBound(i, j);
      float u = _problem->getUpperBound(i, j);
      prev_reference(i, j) = l + (coeff + 1.0) / 2 * (u - l);
    }
  }
  float cost = 0;

  _problem->evaluate(prev_reference, prev_grad, cost, -1);

  float a_k = 1.0F;
  float a_k_1 = 1.0F;

  // Alpha
  VectorXf steplength = VectorXf::Zero(rows);

  for (int iter = 0; iter < 1000; iter++) {
    _problem->evaluate(reference, grad, cost, iter);

    std::cout << cost << std::endl;

    for (Eigen::Index i = 0; i < cols; i++) {
      VectorXf dv = reference.col(i) - prev_reference.col(i);
      VectorXf df = grad.col(i) - prev_grad.col(i);
      steplength(i) = dv.lpNorm<1>() / df.lpNorm<1>();
    }

    new_major = reference - grad * steplength.asDiagonal();

    a_k_1 = (1 + sqrt(4 * a_k * a_k + 1)) * 0.5;

    float coeff = (a_k - 1) / a_k_1;

    prev_reference = std::move(reference);

    reference = new_major + coeff * (new_major - major);

    a_k = std::move(a_k_1);
    prev_grad = std::move(grad);
    major = std::move(new_major);
  }
}
}  // namespace ipl
