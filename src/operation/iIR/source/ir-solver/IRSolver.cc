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

#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>

#include <Eigen/IterativeLinearSolvers>
#include <fstream>
#include <iomanip>
#include <iostream>

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
                 Eigen::Index base_index, Eigen::Index num_nodes) {
  LOG_INFO << "start write matrix, num nodes: " << num_nodes
           << ", base index: " << base_index;
  std::ofstream out("/home/taosimin/iEDA24/iEDA/bin/matrix_m9_m7.txt",
                    std::ios::trunc);
  for (Eigen::Index i = base_index; i < base_index + num_nodes; ++i) {
    for (Eigen::Index j = base_index; j < base_index + num_nodes; ++j) {
      // LOG_INFO << "matrix element at (" << i << ", " << j
      //          << "): " << G_matrix.coeff(i, j);
      out << std::fixed << std::setprecision(6) << G_matrix.coeff(i, j) << " ";
    }
    out << "\n";
  }

  out.close();

  LOG_INFO << "end write matrix";
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
    for (Eigen::SparseMatrix<double>::InnerIterator it(sparse_matrix, col); it;
         ++it) {
      Eigen::Index row = it.row();  // Row index
      double value = it.value();    // Non-zero value

      out << "Col: " << col << ", Row: " << row << ", Value: " << std::fixed
          << std::setprecision(6) << value << "\n";
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
  Eigen::SparseMatrix<double> A =
      G_matrix;  // Copy G_matrix to A for preconditioning

  // PrintMatrix(G_matrix, 0, G_matrix.rows());

  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
  // use comute directorly, mask below code.
  // solver.analyzePattern(G_matrix);
  // solver.factorize(G_matrix);
  solver.compute(A);

  LOG_INFO << "G matrix size: " << G_matrix.rows() << " * " << G_matrix.cols();

  auto ret_value = solver.info();
  if (ret_value != Eigen::Success) {
    // PrintMatrix(G_matrix, 0, G_matrix.rows());
    LOG_FATAL << "LU solver error " << ret_value;
  }

  Eigen::VectorXd v_vector = solver.solve(J_vector);

  // for debug
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
 * @param lambda
 * @return Eigen::VectorXd
 */
Eigen::VectorXd conjugateGradient(const Eigen::SparseMatrix<double>& A,
                                  const Eigen::VectorXd& b,
                                  const Eigen::SparseMatrix<double>& M_inv,
                                  const Eigen::VectorXd& x0, double tol,
                                  int max_iter, double lambda) {
  Eigen::VectorXd x = x0 * 0.95;
  Eigen::VectorXd Ax = A * x;
  // PrintVector(Ax, "/home/taosimin/iEDA24/iEDA/bin/Ax.txt");
  // PrintVector(b, "/home/taosimin/iEDA24/iEDA/bin/b.txt");

  // L2 regularization for residual
  Eigen::VectorXd r = b - Ax - lambda * x;
  // PrintVector(r, "/home/taosimin/iEDA24/iEDA/bin/residual.txt");

  Eigen::VectorXd z = M_inv * r;  // Preconditioned residual

  // PrintVector(z, "/home/taosimin/iEDA24/iEDA/bin/z.txt");

  Eigen::VectorXd p = z;
  double rsold = r.dot(z);

  // Initialize the minimum residual and its corresponding x value
  double min_rsnew = std::numeric_limits<double>::max();
  Eigen::VectorXd min_x = x;

  std::ofstream rsold_file("rsold.csv", std::ios::trunc);
  rsold_file << "iteration,rsold\n";

  int i = 0;
  int min_residual_iter = 0;
  for (; i < max_iter; ++i) {
    LOG_INFO_EVERY_N(1000) << "CG iteration num: " << i + 1
                           << " residual: " << sqrt(rsold);
    // LOG_INFO_EVERY_N(200) << "x:\n"<< x.transpose();

    // L2 regularization for gradient
    Eigen::VectorXd Ap = A * p + lambda * p;
    // PrintVector(Ap, "/home/taosimin/iEDA24/iEDA/bin/Ap.txt");

    double pAp = p.dot(Ap);
    double alpha = rsold / pAp;

    // PrintVector(p, "/home/taosimin/iEDA24/iEDA/bin/p.txt");

    // PrintVector(x, "/home/taosimin/iEDA24/iEDA/bin/voltage.txt");

    x += alpha * p;

    // PrintVector(x, "/home/taosimin/iEDA24/iEDA/bin/voltage.txt");

    // x = x.cwiseMax(x0 * 0.1);  // Ensure min x

    // LOG_INFO_EVERY_N(1) << "x:\n"<< x.transpose();
    r -= alpha * Ap;

    // PrintVector(r, "/home/taosimin/iEDA24/iEDA/bin/residual.txt");

    z = M_inv * r;  // Preconditioned residual

    // PrintVector(z, "/home/taosimin/iEDA24/iEDA/bin/z.txt");

    // LOG_INFO_EVERY_N(1) << "r:\n"<< r.transpose();
    double rsnew = r.dot(z);

    // Update the minimum residual and its corresponding x value
    if (rsnew < min_rsnew) {
      min_rsnew = rsnew;
      min_x = x;
      min_residual_iter = i;
    }

    // LOG_INFO << "current residual: " << sqrt(rsnew);

    if (sqrt(rsnew) < tol) {
      rsold = rsnew;
      ++i;
      break;
    }

    double beta = rsnew / rsold;
    p = z + beta * p;
    rsold = rsnew;

    rsold_file << i + 1 << "," << rsold << "\n";
  }

  LOG_INFO << "Last 20 elements of x:";
  int size = x.size();
  int start_index = std::max(0, size - 20);
  for (int index = start_index; index < size; ++index) {
    LOG_INFO << "x[" << index << "] = " << min_x[index];
  }

  LOG_INFO << "CPU CG toal iteration num: " << i
           << ", minum residual iter: " << min_residual_iter + 1;
  LOG_INFO << "Final residual Norm: " << sqrt(rsold)
           << ", minum residual Norm: " << sqrt(min_rsnew);

  // Close the file after the loop
  rsold_file.close();

  return min_x;
}

/**
 * @brief Guess-Seidel solver the ir drop.
 *
 * @param A
 * @param b
 * @param x0
 * @param tol
 * @param max_iter
 * @return Eigen::VectorXd
 */
Eigen::VectorXd gaussSeidel(const Eigen::SparseMatrix<double>& A,
                            const Eigen::VectorXd& b, const Eigen::VectorXd& x0,
                            double tol, int max_iter) {
  int n = A.rows();
  Eigen::VectorXd x = x0;
  Eigen::VectorXd x_prev(n);
  double residual;

  std::ofstream residual_file("gauss_seidel_residual.csv", std::ios::trunc);
  residual_file << "iteration,residual\n";

  for (int iter = 0; iter < max_iter; ++iter) {
    x_prev = x;

    // Gauss-Seidel iteration
    for (int i = 0; i < n; i++) {
      double sum = 0.0;
      for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
        if (it.row() != i) {
          sum += it.value() * x(it.row());
        }
      }
      x(i) = (b(i) - sum) / A.coeff(i, i);

      // Apply constraints
      x(i) = std::max(x(i), x0(i) * 0.5);
    }

    // Calculate residual
    residual = (x - x_prev).norm() / x.norm();
    residual_file << iter + 1 << "," << residual << "\n";

    LOG_INFO_EVERY_N(100) << "Gauss-Seidel iteration " << iter + 1
                          << " residual: " << residual;

    if (residual < tol) {
      LOG_INFO << "Gauss-Seidel converged after " << iter + 1 << " iterations";
      break;
    }
  }

  residual_file.close();
  return x;
}

/**
 * @brief Calculate the condition number of matrix A.
 *
 * @param A
 * @return double
 */
double calculateConditionNumber(const Eigen::SparseMatrix<double>& A) {
  Eigen::MatrixXd denseA = A;
  using namespace Spectra;

  Eigen::MatrixXd M = denseA + denseA.transpose();

  // Construct matrix operation object using the wrapper class DenseSymMatProd
  DenseSymMatProd<double> op(M);

  // Construct eigen solver object, requesting the largest three eigenvalues
  SymEigsSolver<DenseSymMatProd<double>> eigs(op, 3, 6);

  // Initialize and compute
  eigs.init();
  // int nconv = eigs.compute(SortRule::LargestAlge);

  // Retrieve results
  Eigen::VectorXd evalues;
  if (eigs.info() == CompInfo::Successful) evalues = eigs.eigenvalues();

  LOG_INFO << "Largest eigenvalue: " << evalues.maxCoeff();
  LOG_INFO << "Smallest eigenvalue: " << evalues.minCoeff();

  return 0.0;
}

/**
 * @brief for debug, print top 10 elements of a vector.
 *
 * @param vec
 */
void PrintTopTenVectorElements(Eigen::VectorXd& vec) {
  // for debug
  // Get the top 10 elements of J_vector
  std::vector<std::pair<double, int>> indexed_values;
  for (int i = 0; i < vec.size(); ++i) {
    indexed_values.emplace_back(vec(i), i);
  }

  // Sort in descending order
  std::sort(indexed_values.begin(), indexed_values.end(),
            [](const std::pair<double, int>& a,
               const std::pair<double, int>& b) { return a.first < b.first; });

  // Print the top 10 elements
  LOG_INFO << "Top 10 elements in vec:";
  for (int i = 0; i < std::min(10, static_cast<int>(indexed_values.size()));
       ++i) {
    LOG_INFO << "Index: " << indexed_values[i].second
             << ", Value: " << indexed_values[i].first;
  }
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

  double scale = 1.0;
  J_vector = J_vector * scale;  // convert to mA

#if !CUDA_IR_SOLVER
  Eigen::SparseMatrix<double> A = G_matrix;

  // call the eigen CG solver
  // Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
  //                          Eigen::Lower | Eigen::Upper>
  //     cg;
  // cg.compute(A);
  // cg.setTolerance(_tolerance);
  // cg.setMaxIterations(_max_iteration);

  // cg.solve(J_vector);

  Eigen::VectorXd X0 =
      Eigen::VectorXd::Constant(J_vector.size(), _nominal_voltage * scale);

  // Construct the diagonal preconditioner matrix
  Eigen::SparseMatrix<double> preconditioner(A.rows(), A.cols());
  for (int i = 0; i < A.rows(); ++i) {
    double diag = A.coeff(i, i);
    if (diag != 0) {
      preconditioner.insert(i, i) = 1.0 / 1.0;
      // LOG_INFO << "Diagonal element at index " << i << " : " << diag;
    } else {
      preconditioner.insert(i, i) = 0.0;  // Handle zero diagonal elements
    }
  }
  LOG_INFO << "Preconditioner matrix constructed.";

  auto B = preconditioner * A;           // Apply the preconditioner
  J_vector = preconditioner * J_vector;  // Apply the preconditioner to J_vector

  // for debug, calculate the condition number of matrix A
  // double condition_number = calculateConditionNumber(B);
  // LOG_INFO << "Condition number of matrix A: " << condition_number;

  auto max_iter = std::max((int)X0.size(), _max_iteration);
  Eigen::VectorXd v_vector = conjugateGradient(B, J_vector, preconditioner, X0,
                                               _tolerance, max_iter, _lambda);

  v_vector = v_vector / scale;

  // PrintVector(v_vector, "/home/taosimin/iEDA24/iEDA/bin/voltage.txt");

  LOG_INFO << "CPU solver X[0] result:" << v_vector(0) << std::endl;

#else
  Eigen::SparseMatrix<double> A = G_matrix;
  Eigen::VectorXd X0 =
      Eigen::VectorXd::Constant(J_vector.size(), _nominal_voltage);

  auto max_iter = std::max((int)X0.size(), _max_iteration);
  auto X = ir_cg_solver(A, J_vector, X0, _tolerance, max_iter, _lambda);
  Eigen::VectorXd v_vector(X.size());
  for (decltype(X.size()) i = 0; i < X.size(); ++i) {
    v_vector(i) = X[i];
  }

  LOG_INFO << "GPU solver X[0] result:" << v_vector(0) << std::endl;
#endif

  Eigen::VectorXd residual = G_matrix * v_vector - J_vector;
  LOG_INFO << "residual norm: " << residual.norm() << std::endl;

  auto ir_drops = getIRDrop(v_vector);

  return ir_drops;
}

}  // namespace iir