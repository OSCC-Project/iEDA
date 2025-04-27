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

#if 1
/**
 * @brief for log print.
 * 
 */
# if 0
#define CUDA_LOG_DEBUG(msg, ...) do { \
    printf("CUDA DEBUG %s:%d] " msg "\n", __func__, __LINE__, ##__VA_ARGS__);\
} while (0)
#else 

#define CUDA_LOG_DEBUG(msg, ...)

#endif

#define CUDA_LOG_INFO(msg, ...) do { \
    printf("CUDA INFO %s:%d] " msg "\n", __func__, __LINE__, ##__VA_ARGS__);\
} while (0)

#define CUDA_LOG_WARNING(msg, ...) do { \
    printf("CUDA Warning %s:%d] " msg "\n", __func__, __LINE__, ##__VA_ARGS__);\
} while (0)

#define CUDA_LOG_ERROR(msg, ...) do { \
    printf("CUDA ERROR %s:%d] " msg "\n", __func__, __LINE__, ##__VA_ARGS__);\
} while (0)

#define CUDA_LOG_CRITICAL(msg, ...) do { \
    printf("CUDA CRITICAL %s:%d] " msg "\n", __func__, __LINE__, ##__VA_ARGS__);\
} while (0)

#define CUDA_LOG_FATAL(msg, ...) do { \
    printf("CUDA FATAL %s:%d] " msg "\n", __func__, __LINE__, ##__VA_ARGS__);\
    assert(0);\
} while (0)

#else

#define CUDA_LOG_INFO(msg, ...)
#define CUDA_LOG_WARNING(msg, ...)
#define CUDA_LOG_ERROR(msg, ...)
#define CUDA_LOG_CRITICAL(msg, ...)
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
    printf("CUDA ERROR %s:%d] %s\n", __func__, __LINE__, cudaGetErrorString(error)); \
  } \
} while(0)

#define CUDA_CHECK_ERROR() do { \
  cudaError_t error = cudaGetLastError(); \
  if (error != cudaSuccess) { \
    printf("CUDA ERROR %s:%d] %s\n", __func__, __LINE__, cudaGetErrorString(error)); \
  } \
} while(0)

/**
 * @brief macro for check memory.
 * 
 */
# define CUDA_CHECK_MEMORY(pos, msg) \
size_t free_mem##pos, total_mem##pos; \
cudaMemGetInfo(&free_mem##pos, &total_mem##pos); \
printf("CUDA MEMORY %s:%d] %s free memory: %fMB\n", __func__, __LINE__, msg, (float)free_mem##pos / (1024* 1024)); \
printf("CUDA MEMORY %s:%d] %s total memory: %fMB\n", __func__, __LINE__, msg, (float)total_mem##pos / (1024* 1024))

/**
 * @brief prof the cuda execute time, start and end should the same pos.
 * 
 */
#define CUDA_PROF_START(pos) \
    cudaEvent_t start##pos, stop##pos; \
    float milliseconds##pos = 0; \
    size_t start_free_mem##pos, start_total_mem##pos; \
    cudaEventCreate(&start##pos); \
    cudaEventCreate(&stop##pos); \
    cudaMemGetInfo(&start_free_mem##pos, &start_total_mem##pos); \
    cudaEventRecord(start##pos, 0)

#define CUDA_PROF_END(pos, msg) \
    size_t end_free_mem##pos, end_total_mem##pos; \
    cudaEventRecord(stop##pos, 0); \
    cudaEventSynchronize(stop##pos); \
    cudaMemGetInfo(&end_free_mem##pos, &end_total_mem##pos); \
    cudaEventElapsedTime(&milliseconds##pos, start##pos, stop##pos); \
    printf("CUDA PROF %s:%d] %s time elapsed: %fs\n", __func__, __LINE__, msg, milliseconds##pos / 1000.0); \
    printf("CUDA PROF %s:%d] %s memory usage: %fMB\n", __func__, __LINE__, msg, (float)(start_free_mem##pos - end_free_mem##pos) / (1024* 1024))




