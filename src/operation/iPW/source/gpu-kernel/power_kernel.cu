/**
 * @file power_kernel.cu
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The power gpu kernel for speed up power tool.
 * @version 0.1
 * @date 2024-10-09
 *
 */
#include "cuda_common.cuh"
#include "power_kernel.cuh"

namespace ipower {

const int THREAD_PER_BLOCK_NUM = 2048;

/**
 * @brief use gpu to do bfs.
 * 
 * @param connection_points 
 * @param is_macros 
 * @param seq_arcs 
 * @param combine_depths 
 * @param snk_arcs 
 * @param connection_point_num 
 * @param num_seq_vertexes 
 * @param num_seq_arcs 
 * @param out_connection_points 
 * @return __global__ 
 */
__global__ void find_next_hop_macro(GPUConnectionPoint* connection_points,
                                    unsigned* is_macros, unsigned* seq_arcs,
                                    unsigned* combine_depths,
                                    unsigned* snk_arcs,
                                    int connection_point_num,
                                    int num_seq_vertexes, int num_seq_arcs,
                                    GPUConnectionPoint* out_connection_points) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < connection_point_num) {
    GPUConnectionPoint in_connection_point = connection_points[i];
    int seq_vertex_id = in_connection_point._snk_id;
    unsigned start = snk_arcs[seq_vertex_id * 2];
    unsigned end = snk_arcs[seq_vertex_id * 2 + 1];
    for (unsigned j = start; j < end; ++j) {
      unsigned snk_vertex = seq_arcs[j];
      out_connection_points[j]._src_id = in_connection_point._src_id;
      out_connection_points[j]._snk_id = snk_vertex;
    }
  }
}

/**
 * @brief build the macro connection map
 *
 * @param seq_vertexes the seq vertex index.
 * @param is_macros whether the seq vertex is macro.
 * @param seq_arcs the seq snk arcs for each seq vertex.
 * @param snk_arcs the snk arcs [start, end] for each seq vertex.
 * @param N the number of the seq vertex.
 * @param out_connection_points the output result of the macro connection
 * points.
 */
void build_macro_connection_map(GPUConnectionPoint* connection_points,
                                unsigned* is_macros, unsigned* seq_arcs,
                                unsigned* snk_depths, unsigned* snk_arcs,
                                int connection_point_num, int num_seq_vertexes,
                                int num_seq_arcs,
                                GPUConnectionPoint* out_connection_points) {
  cudaStream_t stream1 = nullptr;
  cudaStreamCreate(&stream1);

  // for copy to gpu memrory.
  GPUConnectionPoint* d_connection_points;
  unsigned* d_is_macros;
  unsigned* d_seq_arcs;
  unsigned* d_snk_depths;
  unsigned* d_snk_arcs;
  GPUConnectionPoint* d_out_connection_points;

  static bool is_init = false;

  auto init_memory = [&]() {
    cudaWithCheck(cudaMallocAsync(
        (void**)&d_connection_points,
        connection_point_num * sizeof(GPUConnectionPoint), stream1));
    cudaWithCheck(cudaMallocAsync(
        (void**)&d_is_macros, num_seq_vertexes * sizeof(unsigned), stream1));
    cudaWithCheck(cudaMallocAsync((void**)&d_seq_arcs,
                                  num_seq_arcs * sizeof(unsigned), stream1));
    cudaWithCheck(cudaMallocAsync((void**)&d_snk_depths,
                                  num_seq_arcs * sizeof(unsigned), stream1));
    // each point has start and end pair snk arc.
    cudaWithCheck(cudaMallocAsync((void**)&d_snk_arcs,
                                  connection_point_num * 2 * sizeof(unsigned),
                                  stream1));
    cudaWithCheck(cudaMallocAsync((void**)&d_out_connection_points,
                                  num_seq_arcs * sizeof(GPUConnectionPoint),
                                  stream1));

    cudaStreamSynchronize(stream1);

    cudaWithCheck(
        cudaMemcpyAsync(d_connection_points, connection_points,
                        connection_point_num * sizeof(GPUConnectionPoint),
                        cudaMemcpyHostToDevice, stream1));
    cudaWithCheck(cudaMemcpyAsync(d_is_macros, is_macros,
                                  num_seq_vertexes * sizeof(unsigned),
                                  cudaMemcpyHostToDevice, stream1));
    cudaWithCheck(cudaMemcpyAsync(d_seq_arcs, seq_arcs,
                                  num_seq_arcs * sizeof(unsigned),
                                  cudaMemcpyHostToDevice, stream1));
    cudaWithCheck(cudaMemcpyAsync(d_snk_depths, snk_depths,
                                  num_seq_arcs * sizeof(unsigned),
                                  cudaMemcpyHostToDevice, stream1));
    // each point has start and end pair snk arc.
    cudaWithCheck(cudaMemcpyAsync(d_snk_arcs, snk_arcs,
                                  connection_point_num * 2 * sizeof(unsigned),
                                  cudaMemcpyHostToDevice, stream1));
    cudaWithCheck(cudaMemcpyAsync(d_out_connection_points,
                                  out_connection_points,
                                  num_seq_arcs * sizeof(GPUConnectionPoint),
                                  cudaMemcpyHostToDevice, stream1));
    cudaStreamSynchronize(stream1);
  };

  if (!is_init) {
    init_memory();
    is_init = true;
  }

  int num_blocks =
      (connection_point_num + THREAD_PER_BLOCK_NUM - 1) / THREAD_PER_BLOCK_NUM;

  find_next_hop_macro<<<num_blocks, THREAD_PER_BLOCK_NUM>>>(
      d_connection_points, d_is_macros, d_seq_arcs, d_snk_depths, d_snk_arcs,
      connection_point_num, num_seq_vertexes, num_seq_arcs,
      d_out_connection_points);

  // TODO free memory.
  cudaStreamDestroy(stream1);
}

}  // namespace ipower