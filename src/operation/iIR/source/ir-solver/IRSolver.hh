/**
 * @file IRSolver.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief
 * @version 0.1
 * @date 2023-08-18
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <map>

namespace iir {

void PrintMatrix(Eigen::Map<Eigen::SparseMatrix<double>>& G_matrix, Eigen::Index base_index); 
class IRSolver {
 public:
  std::vector<double> operator()(
      Eigen::Map<Eigen::SparseMatrix<double>>& G_matrix,
      Eigen::VectorXd& J_vector);
};
}  // namespace iir