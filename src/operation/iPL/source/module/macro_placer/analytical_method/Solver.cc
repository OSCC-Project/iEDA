#include "Solver.hh"

#include <algorithm>
#include <iostream>

#include "Problem.hh"
namespace ipl {
void Solver::doNesterovSolve(VectorXf& solution)
{
  int num_variable = _problem->numVariable();
  if (solution.size() != num_variable) {
    std::cerr << "invalid initial solution!\n";
    return;
  }

  float a_k = 1.0F;
  float a_k_1 = 1.0F;

  // Alpha
  float steplength = 0.0F;

  // Major solution u_k
  VectorXf major = solution;
  VectorXf new_major(num_variable);

  // Reference solution v_k
  VectorXf reference = major;
  VectorXf new_reference(num_variable);

  // Gradiant of Reference solution v_k
  VectorXf grad(num_variable);
  VectorXf prev_grad(num_variable);

  prev_grad.setRandom();

  _problem->evaluate(prev_grad, _jacobi, _cost);
  prev_grad = _patarmeter.transpose() * _jacobi;

  for (int iter = 0; iter < 1000; iter++) {
    _problem->evaluate(reference, _jacobi, _cost);

    grad = _jacobi.transpose() * _patarmeter;

    steplength = (grad - prev_grad).norm() / (reference - prev_grad).norm();

    steplength = std::clamp(steplength, _steplength_l, _steplength_u);

    new_major = reference - steplength * grad.asDiagonal() * reference;

    a_k_1 = (1 + sqrt(4 * a_k * a_k + 1)) * 0.5;

    new_reference = new_major + (a_k - 1) * (new_major - major) / a_k_1;

    prev_grad = std::move(grad);
    major = std::move(new_major);
    reference = std::move(new_reference);
  }
}
}  // namespace ipl
