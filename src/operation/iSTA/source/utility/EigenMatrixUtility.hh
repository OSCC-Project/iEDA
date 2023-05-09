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
 * @file EigenMatrixUtility.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The eigen matrix utility class.
 * @version 0.1
 * @date 2022-04-02
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/SVD>

namespace ista {

class EigenMatrixUtility {
 public:
  /**
   * @brief LU factor full
   *
   * @param A
   * @param b
   * @return Eigen::VectorXd
   */
  static Eigen::VectorXd calVolLUFull(Eigen::MatrixXd& A, Eigen::VectorXd& b) {
    Eigen::VectorXd x = A.fullPivLu().solve(b);
    return x;
  }
  /**
   * @brief LU factor partial
   *
   * @param A
   * @param b
   * @return Eigen::VectorXd
   */
  static Eigen::VectorXd calVolLUPartial(Eigen::MatrixXd& A,
                                         Eigen::VectorXd& b) {
    Eigen::VectorXd x = A.partialPivLu().solve(b);
    return x;
  }
  /**
   * @brief QR factor
   *
   * @param A
   * @param b
   * @return Eigen::VectorXd
   */
  static Eigen::VectorXd calVolQRFactor(Eigen::MatrixXd& A,
                                        Eigen::VectorXd& b) {
    Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
    return x;
  }
  /**
   * @brief jacobi svd
   *
   * @param A
   * @param b
   * @return Eigen::VectorXd
   */
  static Eigen::VectorXd calVolJacobiSVD(Eigen::MatrixXd& A,
                                         Eigen::VectorXd& b) {
    // Eigen::VectorXd x;
    Eigen::VectorXd x = A.jacobiSvd().solve(b);
    return x;
  }

  /**
   * @brief BDC svd
   *
   * @param A
   * @param b
   * @return Eigen::VectorXd
   */
  static Eigen::VectorXd calVolBDCSVD(Eigen::MatrixXd A, Eigen::VectorXd b) {
    Eigen::VectorXd x;
    // Build Failed, TODO
    // Eigen::VectorXd x = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
    return x;
  }

  /**
   * @brief BDC svd
   *
   * @param A
   * @param b
   * @return Eigen::VectorXd
   */
  static auto calVolBDC(Eigen::MatrixXd& A, Eigen::VectorXd& b) {
    Eigen::VectorXd x;
    //  =
    //     A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b)
    return x;
  }

  static Eigen::VectorXd calVolTrapezoidal(Eigen::MatrixXd& G,
                                           Eigen::MatrixXd& C,
                                           Eigen::MatrixXd& V_init, double step,
                                           Eigen::VectorXd& I);

  static Eigen::VectorXd calVolModifiedEuler(Eigen::MatrixXd& G,
                                             Eigen::MatrixXd& C,
                                             Eigen::MatrixXd v_init,
                                             double step, Eigen::VectorXd& I);

  static Eigen::MatrixXd pseInverseBDCSvd(Eigen::MatrixXd& origin,
                                          float er = 0);

  static Eigen::MatrixXd pseInverseSvd(Eigen::MatrixXd& origin);

  static Eigen::MatrixXd pinv(Eigen::MatrixXd& A);

  static Eigen::MatrixXd backEuler(Eigen::MatrixXd& G, Eigen::MatrixXd& C,
                                   Eigen::MatrixXd v_init, double step,
                                   Eigen::VectorXd& I);

  static Eigen::VectorXd gaussSeidel(Eigen::MatrixXd& A, Eigen::VectorXd& b,
                                     int iter_num, double tolerence);

  static double interplot(double t, Eigen::MatrixXd& T, Eigen::MatrixXd& CU);
};

}  // namespace ista
