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
#include <Eigen/IterativeLinearSolvers>

#include "log/Log.hh"
#if CUDA_IR_SOLVER
  #include "ir-solver-cuda/ir_solver.cuh"
#endif

namespace iir {

/**
 * @brief print matrix data for debug.
 *
 * @param G_matrix
 * @param base_index
 */
void PrintMatrix(Eigen::Map<Eigen::SparseMatrix<double>>& G_matrix,
                 Eigen::Index base_index) {
  std::ofstream out("matrix.txt",
                    std::ios::trunc);
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
 * @brief print sparse matrix in CSR.
 * 
 * @param G_matrix 
 */
void PrintCSCMatrix(Eigen::Map<Eigen::SparseMatrix<double>>& G_matrix) {
  std::ofstream out("matrix_sparse.txt", std::ios::trunc);

  // Convert Eigen::Map to Eigen::SparseMatrix
  Eigen::SparseMatrix<double> sparse_matrix = G_matrix;

  // Ensure the matrix is in compressed format
  sparse_matrix.makeCompressed();

  // Iterate over the columns of the sparse matrix (CSC format)
  for (int col = 0; col < sparse_matrix.cols(); ++col) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(sparse_matrix, col); it; ++it) {
      Eigen::Index row = it.row();  // Row index
      double value = it.value();    // Non-zero value

      out << "Col: " << col << ", Row: " << row
          << ", Value: " << std::fixed << std::setprecision(6) << value << "\n";
    }
  }

  out.close();
}

/**
 * @brief print vector data for debug.
 *
 * @param v_vector
 */
void PrintVector(const Eigen::VectorXd& v_vector, const std::string& filename) {
  std::ofstream out(filename, std::ios::trunc);
  for (Eigen::Index i = 0; i < v_vector.size(); ++i) {
    // LOG_INFO << "vector element at (" << i << "): " << v_vector(i);
    out << std::scientific << v_vector(i) << "\n";
  }
  out.close();
}

/**
 * @brief print vector to csv for debug.
 *
 * @param data
 * @param filename
 */
void writeVectorToCsv(Eigen::VectorXd& data, const std::string& filename) {
  std::ofstream ofs(filename, std::ios::trunc);
  if (!ofs.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }
  ofs << "index,value\n";  // CSV header
  for (Eigen::Index i = 0; i < data.size(); ++i) {
    ofs << i << "," << data(i) << "\n";
  }
  ofs.close();
}

/**
 * @brief get the ir drop from voltage vector.
 *
 * @param v_vector
 * @return std::vector<double>
 */
std::vector<double> IRSolver::getIRDrop(Eigen::VectorXd& v_vector) {
  double voltage_max = v_vector.maxCoeff();
  auto node_num = v_vector.size();
  std::vector<double> ir_drops;
  ir_drops.reserve(node_num);
  for (unsigned i = 0; i < node_num; ++i) {
    double val = v_vector(i);
    // LOG_INFO << "node " << i << " voltage: " << val;
    ir_drops.push_back(voltage_max - val);
  }

  return ir_drops;
}

/**
 * @brief solver the ir drop use LU decomposition.
 *
 * @param G_matrix
 * @param J_vector
 * @return std::vector<double>
 */
std::vector<double> IRLUSolver::operator()(
    Eigen::Map<Eigen::SparseMatrix<double>>& G_matrix,
    Eigen::VectorXd& J_vector) {
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
  // use comute directorly, mask below code.
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
  // PrintVector(J_vector, "/home/taosimin/iEDA24/iEDA/bin/current.txt");
  // PrintMatrix(G_matrix, 0);

  Eigen::VectorXd v_vector = solver.solve(J_vector);

  // PrintVector(v_vector, "/home/taosimin/iEDA24/iEDA/bin/voltage.txt");

  // writeVectorToCsv(J_vector, "/home/taosimin/iEDA24/iEDA/bin/current.csv");
  // writeVectorToCsv(v_vector, "/home/taosimin/iEDA24/iEDA/bin/voltage.csv");

  auto ir_drops = getIRDrop(v_vector);

  return ir_drops;
}

/**
 * @brief CPU solver the ir drop use CG gradient.
 * 
 * @param A 
 * @param b 
 * @param x0 
 * @param tol 
 * @param max_iter 
 * @return Eigen::VectorXd 
 */
Eigen::VectorXd conjugateGradient(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, const Eigen::VectorXd& x0, double tol, int max_iter) {
  Eigen::VectorXd x = x0;
  Eigen::VectorXd Ax = A * x;
  // PrintVector(Ax, "Ax.txt");
  // PrintVector(b, "b.txt");
  Eigen::VectorXd r = b - Ax;
  // PrintVector(r, "residual.txt");
  Eigen::VectorXd p = r;
  double rsold = r.dot(r);
  
  int i = 0;
  for (; i < max_iter; ++i) {
      LOG_INFO_EVERY_N(200) << "CPU CG iteration num: " << i << " total " << max_iter;
      LOG_INFO_EVERY_N(200) << "x:\n"<< x.transpose(); 
      Eigen::VectorXd Ap = A * p;
      // PrintVector(Ax, "AP.txt");
      double alpha = rsold / p.dot(Ap);
      x += alpha * p;
      LOG_INFO_EVERY_N(1) << "x:\n"<< x.transpose(); 
      r -= alpha * Ap;
      LOG_INFO_EVERY_N(1) << "r:\n"<< r.transpose(); 
      double rsnew = r.dot(r);
      if (sqrt(rsnew) < tol) {
          break;
      }
      p = r + (rsnew / rsold) * p;
      rsold = rsnew;
  }

  LOG_INFO << "CPU CG iteration num: " << i - 1;
  LOG_INFO << "Final Residual Norm: " << sqrt(rsold);

  return x;
}

/**
 * @brief solver the ir drop use CG gradient.
 *
 * @param G_matrix
 * @param J_vector
 * @return std::vector<double>
 */
std::vector<double> IRCGSolver::operator()(
    Eigen::Map<Eigen::SparseMatrix<double>>& G_matrix,
    Eigen::VectorXd& J_vector) {
  // for debug
  // PrintVector(J_vector, "/home/taosimin/iEDA24/iEDA/bin/current.txt");
  // PrintMatrix(G_matrix, 0);
  // PrintCSCMatrix(G_matrix);
  
#if !CUDA_IR_SOLVER
  Eigen::SparseMatrix<double> A = G_matrix;
  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
                           Eigen::Lower | Eigen::Upper>
      cg;
  cg.compute(A);
  cg.setTolerance(_tolerance);
  cg.setMaxIterations(_max_iteration);
  
  Eigen::VectorXd X0 = Eigen::VectorXd::Constant(J_vector.size(), _nominal_voltage * 0.9);
  Eigen::VectorXd v_vector = conjugateGradient(A, J_vector, X0, _tolerance, _max_iteration);

  LOG_INFO << "CPU solver X[0] result:\n" << v_vector(0) << std::endl;

# else
  Eigen::SparseMatrix<double> A = G_matrix;
  Eigen::VectorXd X0 = Eigen::VectorXd::Constant(J_vector.size(), _nominal_voltage * 0.9);
  auto X = ir_cg_solver(A, J_vector, X0, _tolerance, _max_iteration);
  Eigen::VectorXd v_vector(X.size());
  for(decltype(X.size()) i = 0; i < X.size(); ++i) {
    v_vector(i) = X[i];
  }

  LOG_INFO << "GPU solver X[0] result:\n" << v_vector(0) << std::endl;
#endif

  Eigen::VectorXd residual = G_matrix * v_vector - J_vector;
  LOG_INFO << "residual norm: " << residual.norm() << std::endl;

  auto ir_drops = getIRDrop(v_vector);

  return ir_drops;
}

}  // namespace iir