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

#if 0
/**
 * @brief for log print.
 * 
 */
#define CUDA_LOG_INFO(msg, ...) do { \
    printf("CUDA INFO %s:%d] " msg "\n", __func__, __LINE__, ##__VA_ARGS__);\
} while (0)

#define CUDA_LOG_WARNING(msg, ...) do { \
    printf("CUDA Warning %s:%d] " msg "\n", __func__, __LINE__, ##__VA_ARGS__);\
} while (0)

#define CUDA_LOG_ERROR(msg, ...) do { \
    printf("CUDA ERROR %s:%d] " msg "\n", __func__, __LINE__, ##__VA_ARGS__);\
} while (0)

#define CUDA_LOG_FATAL(msg, ...) do { \
    printf("CUDA ERROR %s:%d] " msg "\n", __func__, __LINE__, ##__VA_ARGS__);\
    assert(0);\
} while (0)

#else

#define CUDA_LOG_INFO(msg, ...)
#define CUDA_LOG_WARNING(msg, ...)
#define CUDA_LOG_ERROR(msg, ...)
#define CUDA_LOG_FATAL(msg, ...)

#endif

/**
 * @brief cuda check error.
 * 
 * @param error 
 * @param file 
 * @param line 
 */
#define CUDA_CHECK(error) do { \
  if (error != cudaSuccess) { \
    CUDA_LOG_ERROR("%s", cudaGetErrorString(error)); \
  } \
} while(0)

#define CUDA_CHECK_ERROR() do { \
  cudaError_t error = cudaGetLastError(); \
  if (error != cudaSuccess) { \
    CUDA_LOG_ERROR("%s", cudaGetErrorString(error)); \
  } \
} while(0)

/**
 * @brief prof the cuda execute time, start and end should the same pos.
 * 
 */
#define CUDA_PROF_START(pos) \
    cudaEvent_t start##pos, stop##pos; \
    float milliseconds##pos = 0; \
    cudaEventCreate(&start##pos); \
    cudaEventCreate(&stop##pos); \
    cudaEventRecord(start##pos, 0)

#define CUDA_PROF_END(pos, msg) \
    cudaEventRecord(stop##pos, 0); \
    cudaEventSynchronize(stop##pos); \
    cudaEventElapsedTime(&milliseconds##pos, start##pos, stop##pos); \
    printf("CUDA PROF %s:%d] %s %f s\n", __func__, __LINE__, msg, milliseconds##pos / 1000.0);

