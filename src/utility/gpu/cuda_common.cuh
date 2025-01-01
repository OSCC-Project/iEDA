/**
 * @file kernel_common.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief gpu cuda common functions file.
 * @version 0.1
 * @date 2024-10-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#pragma once

#include <cuda_runtime.h>
#include "kernel_common.h"


inline void cuda_check(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("CUDA error at file %s:%d:\n%s\n", file, line, cudaGetErrorString(error));
  }
};
#define CUDA_CHECK(err) (cuda_check(err, __FILE__, __LINE__))

#define CUDA_CHECK_ERROR() do { \
  cudaError_t error = cudaGetLastError(); \
  if (error != cudaSuccess) { \
    printf("CUDA error at %s:%d:\n%s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
  } \
} while(0)
