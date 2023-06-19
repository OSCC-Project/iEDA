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

void Solver::solve(const Problem& problem, Mat& solution, const Option& opt)
{
  int rows = problem.variableMatrixRows();
  int cols = problem.variableMatrixcols();
  if (solution.rows() != rows || solution.cols() != cols) {
    std::cerr << "Invalid start solution!\n";
    return;
  }

  // Major solution u_k
  Mat major = std::move(solution);
  // std::cout << major;
  Mat new_major = Mat::Zero(rows, cols);

  // Reference solution v_k
  Mat reference = major;
  Mat prev_reference = Mat::Zero(rows, cols);

  // Gradiant of Reference solution v_k
  Mat grad = Mat::Zero(rows, cols);
  Mat prev_grad = Mat::Zero(rows, cols);

  double cur_cost = 0.0;
  double prev_cost = 0.0;

  prev_reference = reference - reference.binaryExpr(Mat::Random(rows, cols), [](double a, double b) { return 0.1 * a * b; });
  problem.evaluate(prev_reference, prev_grad, prev_cost, -1);

  float a_k = 1.0F;
  float a_k_1 = 1.0F;

  // Alpha
  Vec steplength = Vec::Ones(cols);

  for (size_t iter = 0; iter < opt.max_iter; iter++) {
    problem.evaluate(reference, grad, cur_cost, iter);

    for (Eigen::Index i = 0; i < cols; i++) {
      double grad_dis = (grad.col(i) - prev_grad.col(i)).norm();
      double ref_dis_ = (reference.col(i) - prev_reference.col(i)).norm();
      double ref_dis = problem.getSolutionDistance(reference.col(i), prev_reference.col(i), i);
      steplength(i) = ref_dis / grad_dis;
    }
    // double step = (reference - prev_reference).norm() / (grad - prev_grad).norm();
    if (iter % 10 == 0) {
      log("[NAG]", "iter:", iter, "cost:", cur_cost, "dis:", cur_cost - prev_cost,
          "step:", Eigen::Map<Eigen::RowVectorXd>(steplength.data(), steplength.size()));
      // log("[NAG]", "iter:", iter, "cost:", cur_cost, "dis:", cur_cost - prev_cost, "step:", step);
      prev_cost = cur_cost;
      // std::cout << major << std::endl;
    }
    new_major = reference - grad * steplength.asDiagonal();
    // new_major = reference - grad * step;

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
