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
 * @file ir_solver.cu
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The ir cuda solver.
 * @version 0.1
 * @date 2025-04-19
 *
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cmath>
#include <iostream>
#include <vector>

#include "gpu/cuda_common.cuh"
#include "gpu/kernel_common.h"
#include "ir_solver.cuh"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>


namespace iir {

/**
 * @brief The ir cg solver use cuda.
 *
 * @param A
 * @param b
 * @param x0
 * @param tol
 * @param max_iter
 * @return std::vector<double>
 */
std::vector<double> ir_cg_solver(Eigen::SparseMatrix<double> &A,
                                 Eigen::VectorXd &b, Eigen::VectorXd &x0,
                                 const double tol, const int max_iter, double lambda) {
  // Convert Eigen sparse matrix to CSR format
  A.makeCompressed();
  int num_rows = A.rows();
  int num_cols = A.cols();
  int nnz = A.nonZeros();
  const int *csrRowPtr = A.outerIndexPtr();
  const int *csrColInd = A.innerIndexPtr();
  const double *csrVal = A.valuePtr();

  Eigen::VectorXd x0_new = x0 * 0.95;

  // Allocate device memory
  double *d_csrVal, *d_b, *d_x, *d_x0, *d_r, *d_p, *d_Ap;
  int *d_csrRowPtr, *d_csrColInd;
  CUDA_CHECK(cudaMalloc((void **)&d_csrVal, nnz * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&d_csrRowPtr, (num_rows + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&d_csrColInd, nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&d_b, num_rows * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&d_x, num_rows * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&d_x0, num_rows * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&d_r, num_rows * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&d_p, num_rows * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&d_Ap, num_rows * sizeof(double)));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_csrVal, csrVal, nnz * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_csrRowPtr, csrRowPtr, (num_rows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_csrColInd, csrColInd, nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), num_rows * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x, x0_new.data(), num_rows * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x0, x0.data(), num_rows * sizeof(double),
                        cudaMemcpyHostToDevice));

  // for debug
  // print_device_array(d_b, num_rows);
  // print_device_array(d_x, num_rows);
  // print_device_array(d_x0, num_rows);

  // cuSPARSE handle
  cusparseHandle_t handle;
  cusparseCreate(&handle);

  // Create matrix descriptor
  cusparseSpMatDescr_t matA;
  cusparseCreateCsr(&matA, num_rows, num_cols, nnz, d_csrRowPtr, d_csrColInd,
                    d_csrVal, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

  // Create vector descriptors
  cusparseDnVecDescr_t vecX, vecR, vecP, vecAp;
  cusparseCreateDnVec(&vecX, num_cols, d_x, CUDA_R_64F);
  cusparseCreateDnVec(&vecR, num_rows, d_r, CUDA_R_64F);
  cusparseCreateDnVec(&vecP, num_rows, d_p, CUDA_R_64F);
  cusparseCreateDnVec(&vecAp, num_rows, d_Ap, CUDA_R_64F);

  // Temporary variables
  double alpha = 1.0, beta = 0.0, r_dot_r, r_dot_r_new;

  // Initialize r = A * x
  size_t bufferSize = 0;
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                          matA, vecX, &beta, vecR, CUDA_R_64F,
                          CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
  void *dBuffer;
  cudaMalloc(&dBuffer, bufferSize);

  cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX,
               &beta, vecR, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

  // for debug
  // print_csr_matrix(num_rows, num_cols, nnz, d_csrRowPtr, d_csrColInd,
  // d_csrVal); print_device_array(d_r, num_rows);

  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);

  // Perform r = b - A * x - lambda * x (L2 regularization)
  double neg_lambda = -lambda; // Example value for lambda
  double neg_one = -1.0;
  cublasDaxpy(cublasHandle, num_rows, &neg_one, d_r, 1, d_b, 1);  // r = b - A * x
  cublasDaxpy(cublasHandle, num_rows, &(neg_lambda), d_x, 1, d_b, 1); // Add L2 regularization
  CUDA_CHECK(cudaMemcpy(d_r, d_b, num_rows * sizeof(double), cudaMemcpyDeviceToDevice));  // Copy b to r

  // for debug
  // print_device_array(d_r, num_rows);
  // Copy r to p
  CUDA_CHECK(cudaMemcpy(d_p, d_r, num_rows * sizeof(double),
                        cudaMemcpyDeviceToDevice));

  // print_device_array(d_p, num_rows);

  // Compute initial r_dot_r
  cublasDdot(cublasHandle, num_rows, d_r, 1, d_r, 1, &r_dot_r);

  // Initialize variables to track minimum residual and corresponding x
  double min_r_dot_r_new = r_dot_r;
  double *d_min_x;
  CUDA_CHECK(cudaMalloc((void **)&d_min_x, num_rows * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_min_x, d_x, num_rows * sizeof(double), cudaMemcpyDeviceToDevice));

  int k = 0;
  int min_residual_iter = 0;
  double one = 1.0;
  double zero = 0.0;
  while (k < max_iter && sqrt(r_dot_r) > tol) {
    // CUDA_LOG_INFO("CG iteration num : %d, residual: %f", k + 1 , sqrt(r_dot_r));

    // Ap = A * p + lambda * p (L2 regularization)
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecP,
                 &zero, vecAp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    cublasDaxpy(cublasHandle, num_rows, &lambda, d_p, 1, d_Ap, 1);

    // alpha = r_dot_r / (p^T * Ap)
    double p_dot_Ap;
    cublasDdot(cublasHandle, num_rows, d_p, 1, d_Ap, 1, &p_dot_Ap);
    alpha = r_dot_r / p_dot_Ap;

    // for debug
    // print_device_array(d_x, num_rows);
    // print_device_array(d_p, num_rows);

    // x = x + alpha * p
    cublasDaxpy(cublasHandle, num_rows, &alpha, d_p, 1, d_x, 1);
    
    // for debug
    // print_device_array(d_x, num_rows);

    // r = r - alpha * Ap
    double neg_alpha = -alpha;
    cublasDaxpy(cublasHandle, num_rows, &neg_alpha, d_Ap, 1, d_r, 1);

    // r_dot_r_new = r^T * r
    cublasDdot(cublasHandle, num_rows, d_r, 1, d_r, 1, &r_dot_r_new);

    // Update minimum residual and corresponding x
    if (r_dot_r_new < min_r_dot_r_new) {
      min_r_dot_r_new = r_dot_r_new;
      CUDA_CHECK(cudaMemcpy(d_min_x, d_x, num_rows * sizeof(double), cudaMemcpyDeviceToDevice));
      min_residual_iter = k;
    }

    // beta = r_dot_r_new / r_dot_r
    beta = r_dot_r_new / r_dot_r;

    // p = r + beta * p
    cublasDscal(cublasHandle, num_rows, &beta, d_p, 1);
    cublasDaxpy(cublasHandle, num_rows, &one, d_r, 1, d_p, 1);

    // print_device_array(d_p, num_rows);

    r_dot_r = r_dot_r_new;
    k++;
  }
  // for debug
  // print_device_array(d_x, num_rows);
  // Copy result back to host
  std::vector<double> x(num_rows);
  CUDA_CHECK(cudaMemcpy(x.data(), d_min_x, num_rows * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_LOG_INFO("Last 20 elements of x:");
  int size = x.size();
  int start_index = std::max(0, size - 20);
  for (int i = start_index; i < size; ++i) {
    CUDA_LOG_INFO("x[%d]: %f", i, x[i]);
  }
  
  CUDA_LOG_INFO("GPU CG iteration num: %d, minum residual iter: %d", k, min_residual_iter + 1);
  CUDA_LOG_INFO("Final Residual Norm: %f, minimum Residual Norm: %f", sqrt(r_dot_r), sqrt(min_r_dot_r_new));

  // Free resources
  cudaFree(d_csrVal);
  cudaFree(d_csrRowPtr);
  cudaFree(d_csrColInd);
  cudaFree(d_b);
  cudaFree(d_x);
  cudaFree(d_x0);
  cudaFree(d_r);
  cudaFree(d_p);
  cudaFree(d_Ap);
  cudaFree(dBuffer);
  cudaFree(d_min_x);
  cusparseDestroySpMat(matA);
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecR);
  cusparseDestroyDnVec(vecAp);
  cusparseDestroy(handle);
  cublasDestroy(cublasHandle);

  CUDA_CHECK_ERROR();

  return x;
}



}  // namespace iir