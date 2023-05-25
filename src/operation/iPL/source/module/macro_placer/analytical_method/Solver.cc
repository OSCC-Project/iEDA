#include "Solver.hh"

#include <algorithm>
#include <iomanip>
#include <iostream>

#include "Problem.hh"
namespace ipl {
void Solver::doNesterovSolve(MatrixXd& solution)
{
  int rows = _problem->variableMatrixRows();
  int cols = _problem->variableMatrixcols();
  if (solution.rows() != rows || solution.cols() != cols) {
    std::cerr << "Invalid start solution!\n";
    return;
  }

  // Major solution u_k
  MatrixXd major = std::move(solution);
  MatrixXd new_major = MatrixXd::Zero(rows, cols);

  // Reference solution v_k
  MatrixXd reference = major;
  MatrixXd prev_reference = MatrixXd::Zero(rows, cols);

  // Gradiant of Reference solution v_k
  MatrixXd grad = MatrixXd::Zero(rows, cols);
  MatrixXd prev_grad = MatrixXd::Zero(rows, cols);

  // for (Eigen::Index i = 0; i < prev_reference.rows(); i++) {
  //   for (Eigen::Index j = 0; j < prev_reference.cols(); j++) {
  //     float coeff = prev_reference(i, j);
  //     float l = _problem->getLowerBound(i, j);
  //     float u = _problem->getUpperBound(i, j);
  //     prev_reference(i, j) = l + (coeff + 1.0) / 2 * (u - l);
  //   }
  // }
  float cost = 0;

  prev_reference = reference - reference * 0.1;
  _problem->evaluate(prev_reference, prev_grad, cost, -1);

  float a_k = 1.0F;
  float a_k_1 = 1.0F;

  // Alpha
  VectorXd steplength = VectorXd::Ones(cols);

  for (int iter = 0; iter < 1000; iter++) {
    _problem->evaluate(reference, grad, cost, iter);

    // if (iter % 10 == 0) {
    //   std::cout << std::left << std::setw(6) << "Iter:" << std::setw(5) << iter;
    //   std::cout << std::left << std::setw(6) << "Cost:" << std::setw(15) << cost;
    //   std::cout << std::left << std::setw(7) << "Major:" << std::setw(0) << Eigen::Map<Eigen::RowVectorXd>(major.data(), major.size())
    //             << "     ";
    //   std::cout << std::left << std::setw(6) << "Grad:" << std::setw(0)
    //             << Eigen::Map<Eigen::RowVectorXd>(prev_grad.data(), prev_grad.size()) << "     ";
    //   std::cout << std::left << std::setw(6) << "Step:" << std::setw(0)
    //             << Eigen::Map<Eigen::RowVectorXd>(steplength.data(), steplength.size());
    //   std::cout << std::endl;
    // }
    for (Eigen::Index i = 0; i < cols; i++) {
      steplength(i) = (reference.col(i) - prev_reference.col(i)).norm() / (grad.col(i) - prev_grad.col(i)).norm();
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
  solution = std::move(major);
}
}  // namespace ipl
