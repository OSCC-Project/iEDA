/**
 * @file IRMatrix.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief build matrix of IR data.
 * @version 0.1
 * @date 2024-04-12
 */
#pragma once

#include <memory>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include "iir-rust/IRRustC.hh"

namespace iir {

/**
 * @brief The wrapper of build matrix.
 *
 */
class IRMatrix {
 public:
  Eigen::Map<Eigen::SparseMatrix<double>> buildConductanceMatrix(
      RustNetConductanceData& one_net_matrix_data);
  Eigen::VectorXd buildCurrentVector(void* instance_current_map, std::size_t node_num, std::string net_name);

private:
  std::unique_ptr<Eigen::SparseMatrix<double>> _mat;


};

}  // namespace iir