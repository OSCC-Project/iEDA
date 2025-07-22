/**
 * @file power_kernel.cu
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The power gpu kernel for speed up power tool.
 * @version 0.1
 * @date 2024-10-09
 *
 */
#include "gpu/cuda_common.cuh"
#include "power_kernel.cuh"

namespace ipower {

const int THREAD_PER_BLOCK_NUM = 1024;

/**
 * @brief use gpu to do seq graph bfs.
 *
 * @param connection_points the tranverse from connection point.
 * @param is_macros judge whether the seq vertex is a macro, each element is map to a seq vertex.
 * @param seq_arcs the snk point each seq vertex src arc reach to.
 * @param combine_depths each seq vertex src arc combine logic depth.
 * @param snk_arcs each seq vertex snk arc [start, end] pair.
 * @param connection_point_num the input connection point number.
 * @param num_seq_vertexes the seq graph vertex number.
 * @param num_seq_arcs the seq graph arc number.
 * @param out_connection_points out connection point number.
 * @return __global__
 */
__global__ void find_next_hop_macro(GPU_Connection_Point* connection_points,
                                    unsigned* is_macros, unsigned* seq_arcs,
                                    unsigned* combine_depths,
                                    unsigned* snk_arcs,
                                    int connection_point_num,
                                    int num_seq_vertexes, int num_seq_arcs,
                                    GPU_Connection_Point* out_connection_points) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < connection_point_num) {
    GPU_Connection_Point in_connection_point = connection_points[i];
    int seq_vertex_id = in_connection_point._snk_id;
    unsigned start = snk_arcs[seq_vertex_id * 2];
    unsigned end = snk_arcs[seq_vertex_id * 2 + 1];
    for (unsigned j = start; j <= end; ++j) {
      if (j >= num_seq_arcs) {
        printf(
            "thread %d seq vertex id %d seq index %d (start %d : end %d) "
            "beyond seq arc index.\n",
            i, seq_vertex_id, j, start, end);
      }
      unsigned snk_vertex = seq_arcs[j];
      out_connection_points[j]._src_id = in_connection_point._src_id;
      out_connection_points[j]._snk_id = snk_vertex;
      // printf(
      //     "thread %d traverse next hop src id %d -> snk id %d with out "
      //     "connection %d.\n",
      //     i, in_connection_point._src_id, snk_vertex, j);
    }

    // printf("thread %d traverse the connection point %d next hop.\n", i,
    //        seq_vertex_id);
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
void build_macro_connection_map(GPU_Connection_Point* connection_points,
                                unsigned* is_macros, unsigned* seq_arcs,
                                unsigned* snk_depths, unsigned* snk_arcs,
                                int connection_point_num, int num_seq_vertexes,
                                int num_seq_arcs,
                                GPU_Connection_Point* out_connection_points, bool is_free_memory) {
  cudaStream_t stream1 = nullptr;
  cudaStreamCreate(&stream1);

  // for copy to gpu memrory.
  static GPU_Connection_Point* d_connection_points;
  static unsigned* d_is_macros;
  static unsigned* d_seq_arcs;
  static unsigned* d_snk_depths;
  static unsigned* d_snk_arcs;
  static GPU_Connection_Point* d_out_connection_points;

  static bool is_init = false;

  // init gpu memory.
  auto init_memory = [&]() {

    CUDA_CHECK(cudaMallocAsync(
        (void**)&d_is_macros, num_seq_vertexes * sizeof(unsigned), stream1));
    CUDA_CHECK(cudaMallocAsync((void**)&d_seq_arcs,
                                  num_seq_arcs * sizeof(unsigned), stream1));
    CUDA_CHECK(cudaMallocAsync((void**)&d_snk_depths,
                                  num_seq_arcs * sizeof(unsigned), stream1));
    // each point has start and end pair snk arc.
    CUDA_CHECK(cudaMallocAsync(
        (void**)&d_snk_arcs, num_seq_vertexes * 2 * sizeof(unsigned), stream1));
    CUDA_CHECK(cudaMallocAsync((void**)&d_out_connection_points,
                                  num_seq_arcs * sizeof(GPU_Connection_Point),
                                  stream1));

    cudaStreamSynchronize(stream1);

    CUDA_CHECK(cudaMemcpyAsync(d_is_macros, is_macros,
                                  num_seq_vertexes * sizeof(unsigned),
                                  cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(d_seq_arcs, seq_arcs,
                                  num_seq_arcs * sizeof(unsigned),
                                  cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(d_snk_depths, snk_depths,
                                  num_seq_arcs * sizeof(unsigned),
                                  cudaMemcpyHostToDevice, stream1));
    // each point has start and end pair snk arc.
    CUDA_CHECK(cudaMemcpyAsync(d_snk_arcs, snk_arcs,
                                  num_seq_vertexes * 2 * sizeof(unsigned),
                                  cudaMemcpyHostToDevice, stream1));
  };

  if (!is_init) {
    init_memory();
    is_init = true;
  }

  // need alloc memory every time.
  CUDA_CHECK(cudaMallocAsync(
        (void**)&d_connection_points,
        connection_point_num * sizeof(GPU_Connection_Point), stream1));

  CUDA_CHECK(
      cudaMemcpyAsync(d_connection_points, connection_points,
                      connection_point_num * sizeof(GPU_Connection_Point),
                      cudaMemcpyHostToDevice, stream1));

  CUDA_CHECK(cudaMemcpyAsync(d_out_connection_points, out_connection_points,
                                num_seq_arcs * sizeof(GPU_Connection_Point),
                                cudaMemcpyHostToDevice, stream1));
  cudaStreamSynchronize(stream1);

  int num_blocks =
      (connection_point_num + THREAD_PER_BLOCK_NUM - 1) / THREAD_PER_BLOCK_NUM;

  printf("use block number %d per block num %d, start run gpu kernel.\n", num_blocks,
         THREAD_PER_BLOCK_NUM);

  find_next_hop_macro<<<num_blocks, THREAD_PER_BLOCK_NUM>>>(
      d_connection_points, d_is_macros, d_seq_arcs, d_snk_depths, d_snk_arcs,
      connection_point_num, num_seq_vertexes, num_seq_arcs,
      d_out_connection_points);

  // wait to finish.
  cudaDeviceSynchronize();

  printf("run gpu kernel finished.\n");

  // copy back to host.
  CUDA_CHECK(cudaMemcpyAsync(out_connection_points, d_out_connection_points,
                                num_seq_arcs * sizeof(GPU_Connection_Point),
                                cudaMemcpyDeviceToHost, stream1));

  cudaDeviceSynchronize();

  cudaStreamDestroy(stream1);

  // free memory.
  CUDA_CHECK(cudaFree(d_connection_points));
  if (is_free_memory) {
    CUDA_CHECK(cudaFree(d_is_macros));
    CUDA_CHECK(cudaFree(d_seq_arcs));
    CUDA_CHECK(cudaFree(d_snk_depths));
    CUDA_CHECK(cudaFree(d_snk_arcs));
    CUDA_CHECK(cudaFree(d_out_connection_points));
  }
}

}  // namespace ipower