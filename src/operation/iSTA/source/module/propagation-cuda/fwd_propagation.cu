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
 * @file fwd_propagation.cu
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The fwd propagation using GPU.
 * @version 0.1
 * @date 2025-01-15
 *
 */
#include <cuda_runtime.h>

#include <map>

#include "fwd_propagation.cuh"
#include "gpu/cuda_common.cuh"
#include "lib_arc.cuh"
#include "propagation.cuh"

namespace ista {

/**
 * @brief Get the one fwd data object accroding the analysis mode and trans type.
 * 
 * @param flatten_all_datas 
 * @param the_vertex_data 
 * @param analysis_mode 
 * @param trans_type 
 * @return GPU_Fwd_Data 
 */
__device__ GPU_Fwd_Data get_one_fwd_data(GPU_Fwd_Data* flatten_all_datas,
                              GPU_Vertex_Data* the_vertex_data,
                              GPU_Analysis_Mode analysis_mode,
                              GPU_Trans_Type trans_type) {
    for (unsigned i = 0; i < the_vertex_data->_num_fwd_data; ++i) {
      auto fwd_data = flatten_all_datas[the_vertex_data->_start_pos + i];
      if (fwd_data._analysis_mode == analysis_mode && fwd_data._trans_type == trans_type) {
        return fwd_data;
      }
    }

    GPU_Fwd_Data error_fwd_data;
    return error_fwd_data;

}

/**
 * @brief device function for lut delay arc, using slew and load
 *
 * @param trans_type
 * @param in_slew input slew
 * @param out_load output load
 * @param snk_slew the lut snk vertex slew for store value.
 * @param arc_delay the arc delay for store value
 * @return __device__
 */
__device__ void lut_inst_slew_delay(GPU_Graph* the_graph,
                                    Lib_Arc_GPU& the_lib_arc,
                                    GPU_Arc_Trans_Type the_arc_trans_type,
                                    GPU_Vertex_Data* in_slew,
                                    GPU_Vertex_Data* out_load,
                                    GPU_Vertex_Data* snk_slew,
                                    GPU_Vertex_Data* arc_delay) {
  // TODO(to taosimin), call gpu lut table.
  // store the lut value

  auto find_slew_delay = [the_lib_arc](auto in_trans_type,
                                       auto& one_src_slew_data,
                                       auto& one_snk_cap_data) {
    auto& the_slew_lib_table =
        the_lib_arc
            ._table[GPU_Table_Base_Index::kTransitionBase + in_trans_type];
    double slew_value = find_value(the_slew_lib_table, one_src_slew_data._data_value,
                                   one_snk_cap_data._data_value);
    auto& the_delay_lib_table =
        the_lib_arc._table[GPU_Table_Base_Index::kDelayBase + in_trans_type];
    double delay_value =
        find_value(the_delay_lib_table, one_src_slew_data._data_value,
                   one_snk_cap_data._data_value);

    return std::pair(slew_value, delay_value);
  };

  GPU_Fwd_Data one_src_slew_data;
  GPU_Fwd_Data one_snk_cap_data;
  FOREACH_GPU_FWD_DATA(the_graph->_flatten_slew_data, (*in_slew),
                       one_src_slew_data) {
    GPU_Trans_Type in_trans_type = one_src_slew_data._trans_type;
    GPU_Analysis_Mode analysis_mode = one_src_slew_data._analysis_mode;
    auto out_trans_type = in_trans_type;
    if (the_arc_trans_type == GPU_Arc_Trans_Type::kNegative) {
      out_trans_type = FLIP_TRANS(in_trans_type);
    }
    auto one_snk_cap_data =
        get_one_fwd_data(the_graph->_flatten_node_cap_data, out_load,
                         analysis_mode, out_trans_type);
    auto [slew, delay] =
        find_slew_delay(out_trans_type, one_src_slew_data, one_snk_cap_data);
    if (the_arc_trans_type == GPU_Arc_Trans_Type::kNonUnate) {
      // non unate split the trans type into two type.
      out_trans_type = FLIP_TRANS(in_trans_type);
      auto one_snk_cap_data =
          get_one_fwd_data(the_graph->_flatten_node_cap_data, out_load,
                           analysis_mode, out_trans_type);
      auto [slew1, delay1] =
          find_slew_delay(out_trans_type, one_src_slew_data, one_snk_cap_data);
    }
  }
}

/**
 * @brief device function for lut check arc, using slew and snk slew.
 *
 * @param in_slew
 * @param snk_slew
 * @param arc_delay
 * @return __device__
 */
__device__ void lut_constraint_delay(GPU_Graph* the_graph,
                                     Lib_Arc_GPU& the_lib_arc,
                                     GPU_Vertex_Data* in_slew,
                                     GPU_Vertex_Data* snk_slew,
                                     GPU_Vertex_Data* arc_delay) {
  // TODO(to taosimin), call gpu lut table.
  // store the lut value
  GPU_Fwd_Data one_src_slew_data;
  GPU_Fwd_Data one_snk_slew_data;
  FOREACH_GPU_FWD_DATA(the_graph->_flatten_slew_data, (*in_slew),
                       one_src_slew_data) {
    FOREACH_GPU_FWD_DATA(the_graph->_flatten_slew_data, (*snk_slew),
                         one_snk_slew_data) {
      auto snk_trans = one_snk_slew_data._trans_type;
      auto& the_lib_table = the_lib_arc._table[GPU_Table_Base_Index::kCheckBase + snk_trans];

      double delay_value =
        find_value(the_lib_table, one_src_slew_data._data_value,
                   one_snk_slew_data._data_value);
    }
  }
}

/**
 * @brief device function for lut net arc, using input slew and node table.
 *
 * @param in_slew
 * @param delay_data
 * @param impulse_data
 * @return __device__
 */
__device__ void lut_net_slew_delay(GPU_Vertex_Data* in_slew,
                                   GPU_Vertex_Data* delay_data,
                                   GPU_Vertex_Data* impulse_data,
                                   GPU_Vertex_Data* snk_slew,
                                   GPU_Vertex_Data* arc_delay) {}

/**
 * @brief propagate the bfs arcs in cuda kernal.
 *
 * @param the_graph
 * @param propagated_arcs
 * @return __global__
 */
__global__ void propagate_fwd(GPU_Graph the_graph, Lib_Data_GPU the_lib_data,
                              GPU_BFS_Propagated_Arc propagated_arcs) {
  // current thread id
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < propagated_arcs._num_arcs) {
    unsigned current_arc_id = propagated_arcs._arc_index[i];
    GPU_Arc current_arc = the_graph._arcs[current_arc_id];
    GPU_Arc_Type current_arc_type = current_arc._arc_type;
    GPU_Arc_Trans_Type current_arc_trans_type = current_arc._arc_trans_type;
    unsigned src_vertex_id = current_arc._src_vertex_id;
    unsigned snk_vertex_id = current_arc._snk_vertex_id;
    auto src_vertex = the_graph._vertices[src_vertex_id];
    auto snk_vertex = the_graph._vertices[snk_vertex_id];

    unsigned lib_arc_id = current_arc._lib_data_arc_id;
    auto lib_arc = the_lib_data._arcs_gpu[lib_arc_id];

    if (current_arc_type == kInstDelayArc) {
      lut_inst_slew_delay(&the_graph, lib_arc, current_arc_trans_type,
                          &src_vertex._slew_data, &snk_vertex._node_cap_data,
                          &snk_vertex._slew_data, &current_arc._delay_values);

    } else if (current_arc_type == kInstCheckArc) {
      // lut table for get constrain value for check arc.
      lut_constraint_delay(&the_graph, lib_arc, &src_vertex._slew_data,
                           &snk_vertex._slew_data, &current_arc._delay_values);
    } else {
      // for net arc
      // lut net output slew and delay.
      lut_net_slew_delay(&src_vertex._slew_data, &snk_vertex._node_delay_data,
                         &snk_vertex._node_impulse_data, &snk_vertex._slew_data,
                         &current_arc._delay_values);
    }
  }
}

/**
 * @brief copy sta graph to gpu sta graph.
 *
 */
GPU_Graph copy_from_sta_graph(GPU_Graph& the_cpu_graph,
                              unsigned vertex_data_size,
                              unsigned arc_data_size) {
  const unsigned num_stream = 7;
  cudaStream_t stream[num_stream];
  for (unsigned index = 0; index < num_stream; ++index) {
    cudaStreamCreate(&stream[index]);
  }

  GPU_Graph the_gpu_graph;

  CUDA_CHECK(cudaMallocAsync((void**)&the_gpu_graph._vertices,
                             the_gpu_graph._num_vertices * sizeof(GPU_Vertex),
                             stream[0]));

  CUDA_CHECK(cudaMallocAsync((void**)&the_gpu_graph._arcs,
                             the_gpu_graph._num_arcs * sizeof(GPU_Arc),
                             stream[1]));

  CUDA_CHECK(cudaMallocAsync((void**)&the_gpu_graph._flatten_slew_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data),
                             stream[2]));

  CUDA_CHECK(cudaMallocAsync((void**)&the_gpu_graph._flatten_at_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data),
                             stream[3]));

  CUDA_CHECK(cudaMallocAsync((void**)&the_gpu_graph._flatten_node_cap_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data),
                             stream[4]));

  CUDA_CHECK(cudaMallocAsync((void**)&the_gpu_graph._flatten_node_impulse_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data),
                             stream[5]));

  CUDA_CHECK(cudaMallocAsync((void**)&the_gpu_graph._flatten_arc_delay_data,
                             arc_data_size * sizeof(GPU_Fwd_Data), stream[6]));

  for (unsigned index = 0; index < num_stream; ++index) {
    cudaStreamSynchronize(stream[index]);
  }

  CUDA_CHECK(cudaMemcpyAsync(the_gpu_graph._vertices, the_cpu_graph._vertices,
                             the_gpu_graph._num_vertices * sizeof(GPU_Vertex),
                             cudaMemcpyHostToDevice, stream[0]));

  CUDA_CHECK(cudaMemcpyAsync(the_gpu_graph._arcs, the_cpu_graph._arcs,
                             the_gpu_graph._num_arcs * sizeof(GPU_Arc),
                             cudaMemcpyHostToDevice, stream[1]));

  CUDA_CHECK(cudaMemcpyAsync(the_gpu_graph._flatten_slew_data,
                             the_cpu_graph._flatten_slew_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data),
                             cudaMemcpyHostToDevice, stream[2]));

  CUDA_CHECK(cudaMemcpyAsync(the_gpu_graph._flatten_at_data,
                             the_cpu_graph._flatten_at_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data),
                             cudaMemcpyHostToDevice, stream[3]));

  CUDA_CHECK(cudaMemcpyAsync(the_gpu_graph._flatten_node_cap_data,
                             the_cpu_graph._flatten_node_cap_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data),
                             cudaMemcpyHostToDevice, stream[4]));

  CUDA_CHECK(cudaMemcpyAsync(the_gpu_graph._flatten_node_impulse_data,
                             the_cpu_graph._flatten_node_impulse_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data),
                             cudaMemcpyHostToDevice, stream[5]));

  CUDA_CHECK(cudaMemcpyAsync(the_gpu_graph._flatten_arc_delay_data,
                             the_cpu_graph._flatten_arc_delay_data,
                             arc_data_size * sizeof(GPU_Fwd_Data),
                             cudaMemcpyHostToDevice, stream[6]));

  for (unsigned index = 0; index < num_stream; ++index) {
    cudaStreamSynchronize(stream[index]);
  }

  for (unsigned index = 0; index < num_stream; ++index) {
    cudaStreamDestroy(stream[index]);
  }
  return the_gpu_graph;
}

/**
 * @brief copyback gpu data to cpu sta graph.
 *
 * @param the_cpu_graph
 * @param the_gpu_graph
 */
void copy_to_sta_graph(GPU_Graph& the_cpu_graph, GPU_Graph& the_gpu_graph,
                       unsigned vertex_data_size, unsigned arc_data_size) {
  const unsigned num_stream = 5;
  cudaStream_t stream[num_stream];
  for (unsigned index = 0; index < num_stream; ++index) {
    cudaStreamCreate(&stream[index]);
  }

  CUDA_CHECK(cudaMemcpyAsync(the_cpu_graph._flatten_slew_data,
                             the_gpu_graph._flatten_slew_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data),
                             cudaMemcpyHostToDevice, stream[0]));

  CUDA_CHECK(cudaMemcpyAsync(the_cpu_graph._flatten_at_data,
                             the_gpu_graph._flatten_at_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data),
                             cudaMemcpyHostToDevice, stream[1]));

  CUDA_CHECK(cudaMemcpyAsync(the_cpu_graph._flatten_node_cap_data,
                             the_gpu_graph._flatten_node_cap_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data),
                             cudaMemcpyHostToDevice, stream[2]));

  CUDA_CHECK(cudaMemcpyAsync(the_cpu_graph._flatten_node_impulse_data,
                             the_gpu_graph._flatten_node_impulse_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data),
                             cudaMemcpyHostToDevice, stream[3]));

  CUDA_CHECK(cudaMemcpyAsync(the_cpu_graph._flatten_arc_delay_data,
                             the_gpu_graph._flatten_arc_delay_data,
                             arc_data_size * sizeof(GPU_Fwd_Data),
                             cudaMemcpyHostToDevice, stream[4]));

  for (unsigned index = 0; index < num_stream; ++index) {
    cudaStreamSynchronize(stream[index]);
  }

  for (unsigned index = 0; index < num_stream; ++index) {
    cudaStreamDestroy(stream[index]);
  }
}

/**
 * @brief The interface function for the fwd function.
 * first, build gpu graph
 * for vertex, copy slew data, load, at data, node delay, node impulse
 * for arc, set src and snk id
 * then, propagate level by level.
 */
void gpu_propagate_fwd(
    GPU_Graph& the_cpu_graph, unsigned vertex_data_size, unsigned arc_data_size,
    std::map<unsigned, GPU_BFS_Propagated_Arc>& level_to_arcs,
    Lib_Data_GPU& lib_data) {
  auto the_gpu_graph =
      copy_from_sta_graph(the_cpu_graph, vertex_data_size, arc_data_size);

  // TODO(to taosimin), copy arc id to gpu bfs propagated arc.
  for (auto& [level, the_arcs] : level_to_arcs) {
    propagate_fwd<<<1, 1000>>>(the_gpu_graph, lib_data, the_arcs);
  }

  copy_to_sta_graph(the_cpu_graph, the_gpu_graph, vertex_data_size,
                    arc_data_size);
}

}  // namespace ista