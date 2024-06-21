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
 * @file Reduce.cc
 * @author LH (liuh0326@163.com)
 * @brief The arnoldi reduce method implemention use prime algorithm.
 * @version 0.1
 * @date 2023-03-08
 */
#include "Reduce.hh"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

#include "log/Log.hh"
#include "utility/EigenMatrixUtility.hh"

using namespace Eigen;
namespace ista {
/**
 * @brief use Arnoldi process based Krylov subspace to calculate a set of
 * orthonormal vectors.
 * @param A n*n matrix
 * @param u n*1 initial vector
 * @param k number of krylov space steps,the number of orthonormal vectors is
 * k+1.
 */
std::optional<MatrixXd> ArnoldiROM::genArnoldiBasis(const MatrixXd& A,
                                                    const VectorXd& u, int k) {
  constexpr double eplison = 1e-19;
  auto size = A.rows();
  MatrixXd v(size, k + 1);
  MatrixXd h(k + 1, k);
  VectorXd w(size);
  v.col(0) = u / (u.norm());
  for (int j = 0; j < k; ++j) {
    w = A * v.col(j);
    for (int i = 0; i <= j; ++i) {
      h(i, j) = v.col(i).transpose() * w;
      w = w - h(i, j) * v.col(i);
    }
    h(j + 1, j) = w.norm();
    if (IsDoubleEqual(h(j + 1, j), 0.0, eplison)) {
      LOG_ERROR << "Denominator " << h(j + 1, j) << " is zero";
      return std::nullopt;
    }
    v.col(j + 1) = w / h(j + 1, j);
  }
  return v;
}
/**
 * @brief use Block Arnoldi Algorithm based block Krylov subspace to calculate a
 * set of orthonormal vectors.
 *
 * @param A n*n matrix
 * @param R
 * @param q the column number of orthonormal vectors
 * @param N the number of terminals(ports)
 * @return MatrixXd
 */
MatrixXd ArnoldiROM::blockArnoldi(const MatrixXd& A, const MatrixXd& R, int q,
                                  int N) {
  HouseholderQR<MatrixXd> qr0;
  HouseholderQR<MatrixXd> qrj;
  qr0.compute(R);
  MatrixXd v0 = qr0.householderQ();  // Q
  MatrixXd vj;
  std::vector<MatrixXd*> V_basis;
  std::vector<std::vector<MatrixXd*>> H;
  *V_basis[0] = v0;
  MatrixXd w;
  MatrixXd v_trans;
  MatrixXd V(q, q);
  int k = q / N;
  for (int j = 1; j <= k; ++j) {
    w = A * (*V_basis[j - 1]);
    for (int i = 1; i <= j; ++j) {
      v_trans = V_basis[j - i]->transpose();
      *(H[j - i][j - 1]) = v_trans * w;
      w = w - *V_basis[j - i] * (*(H[j - i][j - 1]));
    }
    qrj.compute(w);
    vj = qrj.householderQ();
    *V_basis[j] = vj;
  }
  int basis_block_num = V_basis.size();
  int block_size = 0;
  int col = 0;
  for (int m = 0; m < basis_block_num; ++m) {
    block_size = V_basis[m]->cols();
    for (int n = 0; n < block_size; ++n) {
      V.col(col) = V_basis[m]->col(n);
      col++;
    }
  }
  return V;
}

/**
 * @brief transform the circuit parameters using  orthogonal basis calculated by
 * arnoldi process.
 *
 * @param cpi initial circuit parameters.
 * @param cpt transformed curcuit parameters.
 * @param k number of krylov space steps,the number of orthonormal vectors is
 * k+1.
 */
void ArnoldiROM::arnoldiTransfer(const CircuitParamInit& cpi,
                                 CircuitParamTrans& cpt, int k) {
  auto size = cpi.C.rows();
  MatrixXd A;
  VectorXd u(size);
  u.setOnes();
  double sum = 0.0;
  for (int i = 0; i < size; ++i) {
    sum += cpi.C(i, i);
  }
  u = u / sum;
  A = cpi.G.inverse() * cpi.C;
  auto v = genArnoldiBasis(A, u, k);
  if (v) {
    auto arnoldi_basis = v.value();
    cpt.G_rom = arnoldi_basis.transpose() * cpi.G * arnoldi_basis;
    cpt.C_rom = arnoldi_basis.transpose() * cpi.C * arnoldi_basis;
    cpt.B_rom = arnoldi_basis.transpose() * cpi.B;
    cpt.L_rom = arnoldi_basis.transpose() * cpi.L;
  }
}
/**
 * @brief use arnoldi process to calculate the orthogonal basis of Krylov
 * subspace.
 *
 * @param G the conductance matrix.
 * @param C the capacitance matrix.
 * @param k k number of krylov space steps,the number of orthonormal vectors is
 * k+1.
 * @return Eigen::MatrixXd
 */
std::optional<MatrixXd> ArnoldiROM::orthogonalBasis(const MatrixXd& G,
                                                    const MatrixXd& C,
                                                    const VectorXd& B, int k) {
  // diag decompose. C.inv * G = W * diag * W_inv
  MatrixXd cap_inv_dot_conductance_matrix = C.inverse() * G;
  EigenSolver<MatrixXd> es(cap_inv_dot_conductance_matrix);
  MatrixXd diag = es.pseudoEigenvalueMatrix();
  const auto& W = es.pseudoEigenvectors();
  auto W_inv = W.inverse();

  // A is -G_inv * C
  MatrixXd A = -W * diag.inverse() * W_inv;
  DVERBOSE_VLOG(1) << "A matrix\n" << A;

  DVERBOSE_VLOG(1) << "B matrix\n" << B;
  // b is G_inv * B
  MatrixXd b = -A * C.inverse() * B;

  DVERBOSE_VLOG(1) << "b matrix\n" << b;

  auto v = genArnoldiBasis(A, b, k);

  DVERBOSE_VLOG_IF(1, v) << "v matrix" << *v;
  return v;
}

}  // namespace ista