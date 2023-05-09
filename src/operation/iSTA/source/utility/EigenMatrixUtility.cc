// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file EigenMatrixUtility.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The eigen matrix utility class.
 * @version 0.1
 * @date 2022-04-02
 */
#include "EigenMatrixUtility.hh"

#include "log/Log.hh"

namespace ista {

/**
 * @brief use trapezoidal method to calculate voltage
 *
 * @param G conductances
 * @param C capacitances
 * @param step
 * @param I
 * @return Eigen::VectorXd
 */
Eigen::VectorXd EigenMatrixUtility::calVolTrapezoidal(Eigen::MatrixXd& G,
                                                      Eigen::MatrixXd& C,
                                                      Eigen::MatrixXd& V_init,
                                                      double step,
                                                      Eigen::VectorXd& I) {
  Eigen::MatrixXd G0 = (G / 2 + C / step) * 1e5;
  Eigen::MatrixXd G1 = C / step - (G) / 2;
  Eigen::MatrixXd G0_inverse = G0.inverse() / 1e5;
  Eigen::MatrixXd v1 = G0_inverse * G1 * V_init + G0_inverse * (I / 2);
  return v1;
}

Eigen::VectorXd EigenMatrixUtility::calVolModifiedEuler(Eigen::MatrixXd& G,
                                                        Eigen::MatrixXd& C,
                                                        Eigen::MatrixXd v_init,
                                                        double step,
                                                        Eigen::VectorXd& I) {
  // Euler method
  Eigen::MatrixXd C_inver = (C * 1e5).inverse() / 1e5;
  Eigen::VectorXd v_euler =
      v_init + (C_inver * I - C_inver * G * v_init) * step;
  // Modified Euler method
  Eigen::VectorXd temp1 = (C_inver * I * step) / 2;
  Eigen::VectorXd temp2 = ((C_inver * G * step) * (v_init + v_euler)) / 2;
  Eigen::MatrixXd v_modi_euler = v_init + temp1 - temp2;
  return v_modi_euler;
}

Eigen::MatrixXd EigenMatrixUtility::pseInverseBDCSvd(Eigen::MatrixXd& origin,
                                                     float er) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd_holder(
      origin, Eigen::ComputeThinU | Eigen::ComputeThinV);

  Eigen::MatrixXd U = svd_holder.matrixU();
  Eigen::MatrixXd V = svd_holder.matrixV();
  Eigen::MatrixXd D = svd_holder.singularValues();
  Eigen::MatrixXd D_d = D.asDiagonal();
  Eigen::MatrixXd INV = V * D_d.inverse() * U.transpose();

  Eigen::MatrixXd S(V.cols(), U.cols());
  S.setZero();

  for (unsigned int i = 0; i < D.size(); ++i) {
    if (D(i, 0) > er) {
      S(i, i) = 1 / D(i, 0);
    } else {
      S(i, i) = 0;
    }
  }

  // pinv_matrix = V * S * U^T
  return V * S * U.transpose();
}
/**
 * @brief using SVD methon to solve pseinverse
 *
 * @param origin
 * @return Eigen::MatrixXd
 */
Eigen::MatrixXd EigenMatrixUtility::pseInverseSvd(Eigen::MatrixXd& origin) {
  // svd
  Eigen::JacobiSVD<Eigen::MatrixXd> svd_holder(
      origin, Eigen::ComputeFullU | Eigen::ComputeFullV);
  // SVD result
  Eigen::MatrixXd U = svd_holder.matrixU();
  Eigen::MatrixXd V = svd_holder.matrixV();
  Eigen::MatrixXd D = svd_holder.singularValues();
  Eigen::MatrixXd D_d = D.asDiagonal();
  Eigen::MatrixXd INV = V * D_d.inverse() * U.transpose();

  return INV;
}
/**
 * @brief SVD method
 *
 * @param A
 * @return Eigen::MatrixXd
 */
Eigen::MatrixXd EigenMatrixUtility::pinv(Eigen::MatrixXd& A) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      A, Eigen::ComputeFullU | Eigen::ComputeFullV);  // M=USV*
  double pinvtoler = 1.e-7;                           // tolerance
  int row = A.rows();
  int col = A.cols();
  int k = std::min(row, col);
  Eigen::MatrixXd X = Eigen::MatrixXd::Zero(col, row);
  Eigen::MatrixXd singularValues_inv = svd.singularValues();
  Eigen::MatrixXd singularValues_inv_mat = Eigen::MatrixXd::Zero(col, row);
  for (long i = 0; i < k; ++i) {
    if (singularValues_inv(i) > pinvtoler)
      singularValues_inv(i) = 1.0 / singularValues_inv(i);
    else
      singularValues_inv(i) = 0;
  }
  for (long i = 0; i < k; ++i) {
    singularValues_inv_mat(i, i) = singularValues_inv(i);
  }
  X = (svd.matrixV()) * (singularValues_inv_mat) *
      (svd.matrixU().transpose());  // X=VS+U*

  return X;
}
/**
 * @brief back euler
 *
 * @param G
 * @param C
 * @param v_init
 * @param step
 * @param I
 * @return Eigen::MatrixXd
 */
Eigen::MatrixXd EigenMatrixUtility::backEuler(Eigen::MatrixXd& G,
                                              Eigen::MatrixXd& C,
                                              Eigen::MatrixXd v_init,
                                              double step, Eigen::VectorXd& I) {
  int size = C.rows();
  Eigen::MatrixXd C_inver(size, size);
  C_inver.setZero();

  for (int i = 0; i < size; i++) {
    if (C(i, i) != 0) {
      C_inver(i, i) = step / C(i, i);
    } else {
      C_inver(i, i) = 0;
    }
  }

  Eigen::MatrixXd result = -C_inver * G * v_init + v_init + C_inver * I;

  return result;
}

/**
 * @brief gauss seidel
 *
 * @param A
 * @param b
 * @param eps
 * @return Eigen::VectorXd
 */

Eigen::VectorXd EigenMatrixUtility::gaussSeidel(Eigen::MatrixXd& A,
                                                Eigen::VectorXd& b,
                                                int iter_num,
                                                double tolerence) {
  int row_size = A.rows();
  int col_size = A.cols();
  int row = b.rows();
  if (row_size != col_size) {
    LOG_ERROR << "Matrix A must be a square matrix" << std::endl;
  } else if (row_size != row) {
    LOG_ERROR << "rows of A must be the same as rows of b" << std::endl;
  }

  for (int i = 0; i < row_size; i++) {
    int row_sum = 0;
    for (int j = 0; j < col_size; j++) {
      if (i != j) {
        row_sum += fabs(A(i, j));
      }
    }
    if (row_sum < A(i, i)) {
      LOG_INFO << "The conditions of Gauss-Seidel have not met" << std::endl;
    }
  }

  Eigen::VectorXd x0(row_size);
  x0.setZero();
  Eigen::VectorXd x(row_size);
  x.setZero();
  int iter_counter = 1;
  while (iter_counter < iter_num) {
    for (int i = 0; i < row_size; i++) {
      double sum = 0;
      for (int j = 0; j < row_size; j++) {
        if (j != i) {
          sum += A(i, j) * x(j);
        }
      }
      x(i) = (b(i) - sum) / A(i, i);
      double esp = (x - x0).norm() / x.norm();
      if (esp < tolerence) {
        break;
      }
      x0 = x;
    }
    iter_counter++;
  }
  if (iter_counter >= iter_num) {
    LOG_INFO << "Max number of iterations has exceeded" << std::endl;
  } else {
    LOG_INFO << "The number of iterations is " << iter_counter << std::endl;
  }
  return x;
}

double EigenMatrixUtility::interplot(double t, Eigen::MatrixXd& T,
                                     Eigen::MatrixXd& CU) {
  int n = T.cols();
  if (t < T(0, 0) || t > T(0, n - 1)) {
    return 0;
  } else {
    int m = 0;
    for (; m < (n - 1); m++) {
      if (t < T(0, m)) {
        break;
      }
    }
    double k = (CU(0, m) - CU(0, m - 1)) / (T(0, m) - T(0, m - 1));
    return k * (t - T(0, m - 1)) + CU(0, m - 1);
  }
}

}  // namespace ista
