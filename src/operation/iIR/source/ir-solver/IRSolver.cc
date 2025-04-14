// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
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
  std::ofstream out("/home/taosimin/iEDA24/iEDA/bin/matrix.txt", std::ios::trunc);
  for (Eigen::Index i = base_index; i < base_index + G_matrix.rows(); ++i) {
    for (Eigen::Index j = base_index; j < base_index + G_matrix.cols(); ++j) {
      // LOG_INFO << "matrix element at (" << i << ", " << j
      //          << "): " << G_matrix.coeff(i, j);
      out << std::fixed << std::setprecision(6) << G_matrix.coeff(i, j) << " ";
    }
    out << "\n";
  }

  out.close();
}

/**
 * @brief print vector data for debug.
 * 
 * @param v_vector 
 */
void PrintVector(Eigen::VectorXd& v_vector) {
  std::ofstream out("/home/taosimin/iEDA24/iEDA/bin/vector.txt",
                    std::ios::trunc);
  for (Eigen::Index i = 0; i < v_vector.size(); ++i) {
    // LOG_INFO << "vector element at (" << i << "): " << v_vector(i);
    out << std::scientific << v_vector(i) << "\n";
  }
  out.close();
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

  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
  // solver.analyzePattern(G_matrix);
  // solver.factorize(G_matrix);
  solver.compute(G_matrix);

  LOG_INFO << "G matrix size: " << G_matrix.rows() << " * " << G_matrix.cols();

  auto ret_value = solver.info();
  if (ret_value != Eigen::Success) {
    PrintMatrix(G_matrix, 0);
    LOG_FATAL << "LU solver error " << ret_value;
  }

    // for debug
  // PrintVector(J_vector);
  // PrintMatrix(G_matrix, 0);

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