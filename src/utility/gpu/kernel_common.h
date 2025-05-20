/**
 * @file kernel_common.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief gpu kernel common functions file.
 * @version 0.1
 * @date 2024-10-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#pragma once

#include <cstdlib>
#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <fstream>

inline void *malloc_check(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        printf("error: cpu memory allocation failed at %s:%d\n", file, line);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

#define MALLOC_Check(size) malloc_check(size, __FILE__, __LINE__)

/**
 * @brief print gpu device array.
 * 
 * @param d_data 
 * @param num_elements 
 */
inline void print_device_array(double* d_data, size_t num_elements) {
    std::ofstream out("/home/taosimin/iEDA24/iEDA/bin/device_array.txt", std::ios::trunc);

    thrust::device_ptr<double> dev_ptr(d_data);
    thrust::host_vector<double> h_data(dev_ptr, dev_ptr + num_elements);

    for (size_t i = 0; i < h_data.size(); ++i) {
        out << h_data[i] << "\n";
    }
    out << std::endl;

    out.close();
}

#include <iostream>

// Function to print the CSR matrix
inline void print_csr_matrix(int num_rows, int num_cols, int nnz, const int *csrRowPtr, const int *csrColInd, const double *csrVal) {
  // Allocate host memory for CSR matrix data
  int *h_csrRowPtr = new int[num_rows + 1];
  int *h_csrColInd = new int[nnz];
  double *h_csrVal = new double[nnz];

  // Copy CSR matrix data from device to host
  cudaMemcpy(h_csrRowPtr, csrRowPtr, (num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_csrColInd, csrColInd, nnz * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_csrVal, csrVal, nnz * sizeof(double), cudaMemcpyDeviceToHost);

  std::ofstream out("/home/taosimin/iEDA24/iEDA/bin/device_matrix.txt", std::ios::trunc);

  out << "CSR Matrix (rows: " << num_rows << ", cols: " << num_cols << ", nnz: " << nnz << ")\n";
  for (int i = 0; i < num_rows; ++i) {
      out << "Row " << i << ": ";
      for (int j = h_csrRowPtr[i]; j < h_csrRowPtr[i + 1]; ++j) {
          out << "(" << h_csrColInd[j] << ", " << h_csrVal[j] << ") ";
      }
      out << "\n";
  }

  out.close();
}


