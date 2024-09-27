/**
 * @file rc_tree.cu
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The rc tree datastrucure implemention for delay calculation.
 * @version 0.1
 * @date 2024-09-25
 */

#include <cuda_runtime.h>

#include "rc_tree.cuh"

namespace istagpu {

__global__ void add(int *a, int *b, int *c) {
  int tid = blockIdx.x;  // this thread handles the data at its thread id
  if (tid < 100) {
    c[tid] = a[tid] * b[tid];
  }

//   printf("thread id: %d \n", tid);
}

/**
 * @brief update the point load of the rc tree.
 *
 * @param rc_net
 */
float DelayRcNet::delay_update_point_load() {
  float a = 1.0;
  float b = 2.0;

  float c = a + b;

  return c;
}

}  // namespace istagpu