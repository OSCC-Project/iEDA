#include "Solver.hh"

#include <algorithm>
#include <iomanip>
#include <iostream>

#include "Log.hh"
#include "Problem.hh"
namespace ipl {
template <typename T>
void log(const T& val)
{
  std::cout << std::left << " " << val << "\n";
}

template <typename T, typename... Args>
void log(const T& val, const Args&... args)
{
  std::cout << std::left << " " << val;
  log(args...);
}

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

  double cur_cost = 0.0;
  double prev_cost = 0.0;

  prev_reference = reference - reference * 0.1;
  _problem->evaluate(prev_reference, prev_grad, prev_cost, -1);

  float a_k = 1.0F;
  float a_k_1 = 1.0F;

  // Alpha
  VectorXd steplength = VectorXd::Ones(cols);

  for (int iter = 0; iter < 1000; iter++) {
    _problem->evaluate(reference, grad, cur_cost, iter);

    for (Eigen::Index i = 0; i < cols; i++) {
      steplength(i) = (reference.col(i) - prev_reference.col(i)).norm() / (grad.col(i) - prev_grad.col(i)).norm();
    }
    if (iter % 10 == 0) {
      log("[NAG]", "iter:", iter, "cost:", cur_cost, "dis:", prev_cost - cur_cost,
          "step:", Eigen::Map<Eigen::RowVectorXd>(steplength.data(), steplength.size()));
      prev_cost = cur_cost;
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
