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

#include "log/Log.hh"

namespace iir {

/**
 * @brief print matrix data for debug.
 *
 * @param G_matrix
 * @param base_index
 */
void PrintMatrix(Eigen::Map<Eigen::SparseMatrix<double>>& G_matrix,
                 Eigen::Index base_index) {
  for (Eigen::Index i = base_index; i < base_index + 100; ++i) {
    for (Eigen::Index j = base_index; j < base_index + 100; ++j) {
      LOG_INFO << "Element at (" << i << ", " << j
               << "): " << G_matrix.coeff(i, j);
    }
  }
}

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
  solver.analyzePattern(G_matrix);
  solver.factorize(G_matrix);

  if (solver.info() != Eigen::Success) {
    PrintMatrix(G_matrix, 0);
    LOG_FATAL << "LU solver error";
  }

  Eigen::VectorXd v_vector = solver.solve(J_vector);

  double voltage_max = v_vector.maxCoeff();
  std::vector<double> ir_drops;
  ir_drops.reserve(node_num);
  for (unsigned i = 0; i < node_num; ++i) {
    double val = v_vector(i);
    ir_drops.push_back(voltage_max - val);
  }

  return ir_drops;
}

}  // namespace iir