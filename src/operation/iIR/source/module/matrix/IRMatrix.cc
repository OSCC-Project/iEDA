/**
 * @file IRMatrix.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief build matrix of IR data.
 * @version 0.1
 * @date 2024-04-12
 */


#include "IRMatrix.hh"

namespace iir {

/**
 * @brief build conductance matrix of one net.
 * 
 * @param one_net_matrix_data 
 * @param node_num 
 * @return Eigen::Map<Eigen::SparseMatrix<double>> 
 */
Eigen::Map<Eigen::SparseMatrix<double>> IRMatrix::buildConductanceMatrix(
    RustNetConductanceData& one_net_matrix_data) {
  auto node_num = one_net_matrix_data.node_num;
  Eigen::SparseMatrix<double> mat(node_num, node_num);

  std::vector<Eigen::Triplet<double>> triplets;
  RustMatrix* one_data;
  FOREACH_VEC_ELEM(&one_net_matrix_data.g_matrix_vec, RustMatrix, one_data) {
    triplets.emplace_back(
        Eigen::Triplet<double>(one_data->row, one_data->col, one_data->data));
  }
  mat.setFromTriplets(triplets.begin(), triplets.end());

  Eigen::Map<Eigen::SparseMatrix<double>> G_matrix(
      mat.rows(), mat.cols(), mat.nonZeros(), mat.outerIndexPtr(),
      mat.innerIndexPtr(), mat.valuePtr());

  return G_matrix;
}

}  // namespace iir