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
                                 const double tol, const int max_iter) {
  // Convert Eigen sparse matrix to CSR format
  A.makeCompressed();
  int num_rows = A.rows();
  int num_cols = A.cols();
  int nnz = A.nonZeros();
  const int *csrRowPtr = A.outerIndexPtr();
  const int *csrColInd = A.innerIndexPtr();
  const double *csrVal = A.valuePtr();

  // Allocate device memory
  double *d_csrVal, *d_b, *d_x, *d_r, *d_p, *d_Ap;
  int *d_csrRowPtr, *d_csrColInd;
  CUDA_CHECK(cudaMalloc((void **)&d_csrVal, nnz * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&d_csrRowPtr, (num_rows + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&d_csrColInd, nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&d_b, num_rows * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&d_x, num_rows * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&d_r, num_rows * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&d_p, num_rows * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&d_Ap, num_rows * sizeof(double)));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_csrVal, csrVal, nnz * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_csrRowPtr, csrRowPtr, (num_rows + 1) * sizeof(int),
             cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_csrColInd, csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x, x0.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));
  
  // for debug
  // print_device_array(d_b, num_rows);
  // print_device_array(d_x, num_rows);

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
  // print_csr_matrix(num_rows, num_cols, nnz, d_csrRowPtr, d_csrColInd, d_csrVal);
  // print_device_array(d_r, num_rows);

  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);

  // Perform r = b - A * x
  double neg_one = -1.0;
  cublasDaxpy(cublasHandle, num_rows, &neg_one, d_r, 1, d_b, 1);  // r = b - vecR
  CUDA_CHECK(cudaMemcpy(d_r, d_b, num_rows * sizeof(double), cudaMemcpyDeviceToDevice)); // Copy b to r

  // for debug
  // print_device_array(d_r, num_rows);

  // Copy r to p
  CUDA_CHECK(cudaMemcpy(d_p, d_r, num_rows * sizeof(double), cudaMemcpyDeviceToDevice));

  // Compute initial r_dot_r
  cublasDdot(cublasHandle, num_rows, d_r, 1, d_r, 1, &r_dot_r);

  int k = 0;
  double one = 1.0;
  double zero = 0.0;
  while (k < max_iter && sqrt(r_dot_r) > tol) {
    // Ap = A * p
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecP,
                 &zero, vecAp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

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

    // beta = r_dot_r_new / r_dot_r
    beta = r_dot_r_new / r_dot_r;

    // p = r + beta * p
    cublasDscal(cublasHandle, num_rows, &beta, d_p, 1);
    double one = 1.0;
    cublasDaxpy(cublasHandle, num_rows, &one, d_r, 1, d_p, 1);

    r_dot_r = r_dot_r_new;
    k++;
  }
  
  CUDA_LOG_INFO("CG Iterations: %d", k - 1);
  CUDA_LOG_INFO("Final Residual Norm: %f", sqrt(r_dot_r));

  // for debug
  // print_device_array(d_x, num_rows);

  // Copy result back to host
  std::vector<double> x(num_rows);
  CUDA_CHECK(cudaMemcpy(x.data(), d_x, num_rows * sizeof(double), cudaMemcpyDeviceToHost));

  // Free resources
  cudaFree(d_csrVal);
  cudaFree(d_csrRowPtr);
  cudaFree(d_csrColInd);
  cudaFree(d_b);
  cudaFree(d_x);
  cudaFree(d_r);
  cudaFree(d_p);
  cudaFree(d_Ap);
  cudaFree(dBuffer);
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