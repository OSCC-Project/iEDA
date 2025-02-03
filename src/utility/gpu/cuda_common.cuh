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

/**
 * @brief cuda check error.
 * 
 * @param error 
 * @param file 
 * @param line 
 */
inline void cuda_check(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("CUDA error at file %s:%d: %s\n", file, line, cudaGetErrorString(error));
  }
};
#define CUDA_CHECK(err) (cuda_check(err, __FILE__, __LINE__))

#define CUDA_CHECK_ERROR() do { \
  cudaError_t error = cudaGetLastError(); \
  if (error != cudaSuccess) { \
    printf("CUDA error at %s:%d:%s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
  } \
} while(0)

#if 1
/**
 * @brief for log print.
 * 
 */
#define CUDA_LOG_INFO(msg, ...) do { \
    printf("INFO %s:%d] " msg "\n", __func__, __LINE__, ##__VA_ARGS__);\
} while (0)

#define CUDA_LOG_WARNING(msg, ...) do { \
    printf("Warning %s:%d] " msg "\n", __func__, __LINE__, ##__VA_ARGS__);\
} while (0)

#define CUDA_LOG_ERROR(msg, ...) do { \
    printf("ERROR %s:%d] " msg "\n", __func__, __LINE__, ##__VA_ARGS__);\
} while (0)

#else

#define CUDA_LOG_INFO(msg, ...)
#define CUDA_LOG_WARNING(msg, ...)
#define CUDA_LOG_ERROR(msg, ...)

#endif




