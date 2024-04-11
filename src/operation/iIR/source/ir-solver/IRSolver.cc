/**
 * @file IRSolver.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief
 * @version 0.1
 * @date 2023-08-18
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "IRSolver.hh"

#include <fstream>
#include <iomanip>
#include <iostream>

namespace iir {

/**
 * @brief solver the ir drop.
 * 
 * @param G_matrix 
 * @param J_vector 
 * @return std::vector<double> 
 */
std::vector<double> IRSolver::operator()(
    Eigen::Map<Eigen::SparseMatrix<double>>& G_matrix,
    Eigen::VectorXd& J_vector) {
  unsigned node_num = J_vector.size();

  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(G_matrix);
  Eigen::VectorXd v_vector = solver.solve(J_vector);

  double voltage_max = v_vector.maxCoeff();
  std::vector<double> ir_drops;
  ir_drops.reserve(node_num);
  for (unsigned i = 0; i < node_num; i++) {
    double val = v_vector(i);
    ir_drops.push_back(voltage_max - val);
  }

  return ir_drops;
}

}  // namespace iir