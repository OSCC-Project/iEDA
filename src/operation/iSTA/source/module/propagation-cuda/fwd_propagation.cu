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
#include "include/Type.hh"
#include "propagation.cuh"

namespace ista {

/**
 * @brief Get the one fwd data object accroding the analysis mode and trans
 * type.
 *
 * @param flatten_all_datas
 * @param the_vertex_data
 * @param analysis_mode
 * @param trans_type
 * @return GPU_Fwd_Data and index
 */
template <typename T>
__device__ std::pair<GPU_Fwd_Data<T>, int> get_one_fwd_data(
    GPU_Fwd_Data<T>* flatten_all_datas, GPU_Vertex_Data* the_vertex_data,
    GPU_Analysis_Mode analysis_mode, GPU_Trans_Type trans_type) {
  for (unsigned i = 0; i < the_vertex_data->_num_fwd_data; ++i) {
    int data_index = the_vertex_data->_start_pos + i;
    auto fwd_data = flatten_all_datas[data_index];
    if (fwd_data._analysis_mode == analysis_mode &&
        fwd_data._trans_type == trans_type) {
      return std::make_pair(fwd_data, data_index);
    }
  }

  GPU_Fwd_Data<T> error_fwd_data;
  error_fwd_data._data_value = -1.0;
  printf("not found the fwd data.");
  return std::make_pair(error_fwd_data, -1);
}

/**
 * @brief Set the one fwd data object
 *
 * @param flatten_all_datas
 * @param the_vertex_data
 * @param analysis_mode
 * @param trans_type
 * @param data_value
 * @return __device__
 */
template <GPU_OP_TYPE op>
__device__ void set_one_fwd_data(GPU_Graph* the_graph, GPU_Arc& the_arc,
                                 GPU_Analysis_Mode analysis_mode,
                                 GPU_Trans_Type in_trans_type,
                                 GPU_Trans_Type out_trans_type,
                                 int64_t data_value, bool is_force = false) {
  GPU_Fwd_Data<int64_t>* flatten_all_datas;
  if (op == GPU_OP_TYPE::kSlew) {
    flatten_all_datas = the_graph->_flatten_slew_data;
  } else if (op == GPU_OP_TYPE::kDelay) {
    flatten_all_datas = the_graph->_flatten_arc_delay_data;
  } else {
    flatten_all_datas = the_graph->_flatten_at_data;
  }

  unsigned src_vertex_id = the_arc._src_vertex_id;
  unsigned snk_vertex_id = the_arc._snk_vertex_id;

  auto src_vertex = the_graph->_vertices[src_vertex_id];
  auto snk_vertex = the_graph->_vertices[snk_vertex_id];

  GPU_Vertex_Data* src_vertex_data;
  GPU_Vertex_Data* snk_vertex_data;
  if (op == GPU_OP_TYPE::kSlew) {
    src_vertex_data = &src_vertex._slew_data;
    snk_vertex_data = &snk_vertex._slew_data;
  } else if (op == GPU_OP_TYPE::kDelay) {
    src_vertex_data = &the_arc._delay_values;
    snk_vertex_data = &the_arc._delay_values;
  } else {
    src_vertex_data = &src_vertex._at_data;
    snk_vertex_data = &snk_vertex._at_data;
  }

  auto [src_fwd_data, src_data_index] = get_one_fwd_data(
      flatten_all_datas, src_vertex_data, analysis_mode, in_trans_type);

  auto [snk_fwd_data, snk_data_index] = get_one_fwd_data(
      flatten_all_datas, snk_vertex_data, analysis_mode, out_trans_type);

  if (is_force) {
    snk_fwd_data._data_value = data_value;
  }

  // lock the data.
  int expected = 0;
  while (atomicCAS(&snk_fwd_data._is_lock, expected, 1) != expected) {
    expected = 0;
  }

  if (GPU_Analysis_Mode::kMax == analysis_mode) {
    if (op != GPU_OP_TYPE::kAT) {
      if (snk_fwd_data._data_value < data_value) {
        snk_fwd_data._src_vertex_id = src_vertex_id;
        snk_fwd_data._src_data_index = src_data_index;
        snk_fwd_data._data_value = data_value;
      }
    } else {
      if (snk_fwd_data._data_value < (src_fwd_data._data_value + data_value)) {
        snk_fwd_data._src_vertex_id = src_vertex_id;
        snk_fwd_data._src_data_index = src_data_index;
        snk_fwd_data._data_value = src_fwd_data._data_value + data_value;
      }
    }
  } else {
    if (op != GPU_OP_TYPE::kAT) {
      if (snk_fwd_data._data_value > data_value) {
        snk_fwd_data._src_vertex_id = src_vertex_id;
        snk_fwd_data._src_data_index = src_data_index;
        snk_fwd_data._data_value = data_value;
      }
    } else {
      if (snk_fwd_data._data_value > (src_fwd_data._data_value + data_value)) {
        snk_fwd_data._src_vertex_id = src_vertex_id;
        snk_fwd_data._src_data_index = src_data_index;
        snk_fwd_data._data_value = src_fwd_data._data_value + data_value;
      }
    }
  }

  // release the lock.
  atomicExch(&snk_fwd_data._is_lock, 0);
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
__device__ void propagate_inst_slew_delay(GPU_Graph* the_graph,
                                          GPU_Arc& the_arc,
                                          Lib_Arc_GPU& the_lib_arc) {
  unsigned src_vertex_id = the_arc._src_vertex_id;
  unsigned snk_vertex_id = the_arc._snk_vertex_id;
  auto src_vertex = the_graph->_vertices[src_vertex_id];
  auto snk_vertex = the_graph->_vertices[snk_vertex_id];

  GPU_Arc_Trans_Type the_arc_trans_type = the_arc._arc_trans_type;
  GPU_Vertex_Data* in_slew = &src_vertex._slew_data;
  GPU_Vertex_Data* out_load = &snk_vertex._node_cap_data;

  auto find_slew_delay = [the_lib_arc](auto in_trans_type,
                                       auto& one_src_slew_data,
                                       auto& one_snk_cap_data) {
    auto& the_slew_lib_table =
        the_lib_arc
            ._table[GPU_Table_Base_Index::kTransitionBase + in_trans_type];
    float slew_value =
        find_value(the_slew_lib_table, one_src_slew_data._data_value,
                   one_snk_cap_data._data_value);
    auto& the_delay_lib_table =
        the_lib_arc._table[GPU_Table_Base_Index::kDelayBase + in_trans_type];
    float delay_value =
        find_value(the_delay_lib_table, one_src_slew_data._data_value,
                   one_snk_cap_data._data_value);

    return std::pair(int64_t(NS_TO_FS(slew_value)),
                     int64_t(NS_TO_FS(delay_value)));
  };

  GPU_Fwd_Data<int64_t> one_src_slew_data;
  GPU_Fwd_Data<double> one_snk_cap_data;
  FOREACH_GPU_FWD_DATA(the_graph->_flatten_slew_data, (*in_slew),
                       one_src_slew_data) {
    GPU_Trans_Type in_trans_type = one_src_slew_data._trans_type;
    GPU_Analysis_Mode analysis_mode = one_src_slew_data._analysis_mode;
    auto out_trans_type = in_trans_type;
    if (the_arc_trans_type == GPU_Arc_Trans_Type::kNegative) {
      out_trans_type = GPU_FLIP_TRANS(in_trans_type);
    }
    auto [one_snk_cap_data, snk_cap_index] =
        get_one_fwd_data(the_graph->_flatten_node_cap_data, out_load,
                         analysis_mode, out_trans_type);
    auto [slew, delay] =
        find_slew_delay(out_trans_type, one_src_slew_data, one_snk_cap_data);

    set_one_fwd_data<GPU_OP_TYPE::kSlew>(the_graph, the_arc, analysis_mode,
                                         in_trans_type, out_trans_type, slew);
    set_one_fwd_data<GPU_OP_TYPE::kDelay>(the_graph, the_arc, analysis_mode,
                                          in_trans_type, out_trans_type, delay);
    // update at data.
    set_one_fwd_data<GPU_OP_TYPE::kAT>(the_graph, the_arc, analysis_mode,
                                       in_trans_type, out_trans_type, delay);
    if (the_arc_trans_type == GPU_Arc_Trans_Type::kNonUnate) {
      // non unate split the trans type into two type.
      out_trans_type = GPU_FLIP_TRANS(in_trans_type);
      auto [one_snk_cap_data1, snk_cap_index1] =
          get_one_fwd_data(the_graph->_flatten_node_cap_data, out_load,
                           analysis_mode, out_trans_type);
      auto [slew1, delay1] =
          find_slew_delay(out_trans_type, one_src_slew_data, one_snk_cap_data1);

      set_one_fwd_data<GPU_OP_TYPE::kSlew>(the_graph, the_arc, analysis_mode,
                                           in_trans_type, out_trans_type,
                                           slew1);
      set_one_fwd_data<GPU_OP_TYPE::kDelay>(the_graph, the_arc, analysis_mode,
                                            in_trans_type, out_trans_type,
                                            delay1);
      set_one_fwd_data<GPU_OP_TYPE::kAT>(the_graph, the_arc, analysis_mode,
                                         in_trans_type, out_trans_type, delay1);
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
__device__ void lut_constraint_delay(GPU_Graph* the_graph, GPU_Arc& the_arc,
                                     Lib_Arc_GPU& the_lib_arc) {
  unsigned src_vertex_id = the_arc._src_vertex_id;
  unsigned snk_vertex_id = the_arc._snk_vertex_id;
  auto src_vertex = the_graph->_vertices[src_vertex_id];
  auto snk_vertex = the_graph->_vertices[snk_vertex_id];

  GPU_Vertex_Data* in_slew = &src_vertex._slew_data;
  GPU_Vertex_Data* snk_slew = &snk_vertex._slew_data;

  GPU_Fwd_Data<int64_t> one_src_slew_data;
  GPU_Fwd_Data<int64_t> one_snk_slew_data;
  FOREACH_GPU_FWD_DATA(the_graph->_flatten_slew_data, (*in_slew),
                       one_src_slew_data) {
    // TODO(to taosimin), need judge the clock trigger type.
    FOREACH_GPU_FWD_DATA(the_graph->_flatten_slew_data, (*snk_slew),
                         one_snk_slew_data) {
      auto snk_trans = one_snk_slew_data._trans_type;

      auto& the_lib_table =
          the_lib_arc._table[GPU_Table_Base_Index::kCheckBase + snk_trans];

      float delay_value_ns =
          find_value(the_lib_table, one_src_slew_data._data_value,
                     one_snk_slew_data._data_value);
      int64_t delay_value = NS_TO_FS(delay_value_ns);

      auto analysis_mode = one_snk_slew_data._analysis_mode;

      set_one_fwd_data<GPU_OP_TYPE::kDelay>(the_graph, the_arc, analysis_mode,
                                            snk_trans, snk_trans, delay_value,
                                            true);
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
__device__ void propagate_net_slew_delay(GPU_Graph* the_graph,
                                         GPU_Arc& the_arc) {
  unsigned src_vertex_id = the_arc._src_vertex_id;
  unsigned snk_vertex_id = the_arc._snk_vertex_id;
  auto src_vertex = the_graph->_vertices[src_vertex_id];
  auto snk_vertex = the_graph->_vertices[snk_vertex_id];

  GPU_Vertex_Data* in_slew = &src_vertex._slew_data;
  GPU_Vertex_Data* delay_data = &snk_vertex._node_delay_data;
  GPU_Vertex_Data* impulse_data = &snk_vertex._node_impulse_data;

  // slew
  {
    GPU_Fwd_Data<int64_t> one_src_slew_data;
    FOREACH_GPU_FWD_DATA(the_graph->_flatten_slew_data, (*in_slew),
                         one_src_slew_data) {
      auto in_slew_value = FS_TO_PS(one_src_slew_data._data_value);
      GPU_Trans_Type in_trans_type = one_src_slew_data._trans_type;
      GPU_Analysis_Mode analysis_mode = one_src_slew_data._analysis_mode;
      auto [one_snk_impulse_data, impulse_data_index] =
          get_one_fwd_data(the_graph->_flatten_node_impulse_data, impulse_data,
                           analysis_mode, in_trans_type);

      float out_slew = in_slew_value < 0.0
                           ? -std::sqrt(in_slew_value * in_slew_value +
                                        one_snk_impulse_data._data_value)
                           : std::sqrt(in_slew_value * in_slew_value +
                                       one_snk_impulse_data._data_value);

      set_one_fwd_data<GPU_OP_TYPE::kSlew>(the_graph, the_arc, analysis_mode,
                                           in_trans_type, in_trans_type,
                                           PS_TO_FS(out_slew));
    }

    // delay
    {
      GPU_Fwd_Data<double> one_snk_delay_data;
      FOREACH_GPU_FWD_DATA(the_graph->_flatten_node_delay_data, (*delay_data),
                           one_snk_delay_data) {
        float delay_value = one_snk_delay_data._data_value;
        auto analysis_mode = one_snk_delay_data._analysis_mode;
        auto in_trans_type = one_snk_delay_data._trans_type;
        // net out trans type is the same with the in trans type.
        set_one_fwd_data<GPU_OP_TYPE::kDelay>(the_graph, the_arc, analysis_mode,
                                              in_trans_type, in_trans_type,
                                              delay_value);
        set_one_fwd_data<GPU_OP_TYPE::kAT>(the_graph, the_arc, analysis_mode,
                                           in_trans_type, in_trans_type,
                                           PS_TO_FS(delay_value));
      }
    }
  }
}

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

    unsigned lib_arc_id = current_arc._lib_data_arc_id;
    auto lib_arc = the_lib_data._arcs_gpu[lib_arc_id];

    if (current_arc_type == kInstDelayArc) {
      // for inst delay arc.
      propagate_inst_slew_delay(&the_graph, current_arc, lib_arc);
    } else if (current_arc_type == kInstCheckArc) {
      // lut table for get constrain value for check arc.
      lut_constraint_delay(&the_graph, current_arc, lib_arc);
    } else {
      // for net arc
      // lut net output slew and delay.
      propagate_net_slew_delay(&the_graph, current_arc);
    }
  }
}

/**
 * @brief copy sta graph to gpu sta graph.
 *
 */
GPU_Graph copy_from_host_graph(GPU_Graph& the_host_graph,
                              unsigned vertex_data_size,
                              unsigned arc_data_size) {
  const unsigned num_stream = 7;
  cudaStream_t stream[num_stream];
  for (unsigned index = 0; index < num_stream; ++index) {
    cudaStreamCreate(&stream[index]);
  }

  GPU_Graph the_device_graph;

  CUDA_CHECK(cudaMallocAsync((void**)&the_device_graph._vertices,
                             the_device_graph._num_vertices * sizeof(GPU_Vertex),
                             stream[0]));

  CUDA_CHECK(cudaMallocAsync((void**)&the_device_graph._arcs,
                             the_device_graph._num_arcs * sizeof(GPU_Arc),
                             stream[1]));

  CUDA_CHECK(cudaMallocAsync((void**)&the_device_graph._flatten_slew_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data<int64_t>),
                             stream[2]));

  CUDA_CHECK(cudaMallocAsync((void**)&the_device_graph._flatten_at_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data<int64_t>),
                             stream[3]));

  CUDA_CHECK(cudaMallocAsync((void**)&the_device_graph._flatten_node_cap_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data<double>),
                             stream[4]));

  CUDA_CHECK(cudaMallocAsync((void**)&the_device_graph._flatten_node_impulse_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data<double>),
                             stream[5]));

  CUDA_CHECK(cudaMallocAsync((void**)&the_device_graph._flatten_arc_delay_data,
                             arc_data_size * sizeof(GPU_Fwd_Data<int64_t>),
                             stream[6]));

  for (unsigned index = 0; index < num_stream; ++index) {
    cudaStreamSynchronize(stream[index]);
  }

  CUDA_CHECK(cudaMemcpyAsync(the_device_graph._vertices, the_host_graph._vertices,
                             the_device_graph._num_vertices * sizeof(GPU_Vertex),
                             cudaMemcpyHostToDevice, stream[0]));

  CUDA_CHECK(cudaMemcpyAsync(the_device_graph._arcs, the_host_graph._arcs,
                             the_device_graph._num_arcs * sizeof(GPU_Arc),
                             cudaMemcpyHostToDevice, stream[1]));

  CUDA_CHECK(cudaMemcpyAsync(the_device_graph._flatten_slew_data,
                             the_host_graph._flatten_slew_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data<int64_t>),
                             cudaMemcpyHostToDevice, stream[2]));

  CUDA_CHECK(cudaMemcpyAsync(the_device_graph._flatten_at_data,
                             the_host_graph._flatten_at_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data<int64_t>),
                             cudaMemcpyHostToDevice, stream[3]));

  CUDA_CHECK(cudaMemcpyAsync(the_device_graph._flatten_node_cap_data,
                             the_host_graph._flatten_node_cap_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data<double>),
                             cudaMemcpyHostToDevice, stream[4]));

  CUDA_CHECK(cudaMemcpyAsync(the_device_graph._flatten_node_impulse_data,
                             the_host_graph._flatten_node_impulse_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data<double>),
                             cudaMemcpyHostToDevice, stream[5]));

  CUDA_CHECK(cudaMemcpyAsync(the_device_graph._flatten_arc_delay_data,
                             the_host_graph._flatten_arc_delay_data,
                             arc_data_size * sizeof(GPU_Fwd_Data<int64_t>),
                             cudaMemcpyHostToDevice, stream[6]));

  for (unsigned index = 0; index < num_stream; ++index) {
    cudaStreamSynchronize(stream[index]);
  }

  for (unsigned index = 0; index < num_stream; ++index) {
    cudaStreamDestroy(stream[index]);
  }
  return the_device_graph;
}

/**
 * @brief copyback gpu data to cpu sta graph.
 *
 * @param the_host_graph
 * @param the_device_graph
 */
void copy_to_host_graph(GPU_Graph& the_host_graph, GPU_Graph& the_device_graph,
                       unsigned vertex_data_size, unsigned arc_data_size) {
  const unsigned num_stream = 5;
  cudaStream_t stream[num_stream];
  for (unsigned index = 0; index < num_stream; ++index) {
    cudaStreamCreate(&stream[index]);
  }

  CUDA_CHECK(cudaMemcpyAsync(the_host_graph._flatten_slew_data,
                             the_device_graph._flatten_slew_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data<int64_t>),
                             cudaMemcpyDeviceToHost, stream[0]));

  CUDA_CHECK(cudaMemcpyAsync(the_host_graph._flatten_at_data,
                             the_device_graph._flatten_at_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data<int64_t>),
                             cudaMemcpyDeviceToHost, stream[1]));

  CUDA_CHECK(cudaMemcpyAsync(the_host_graph._flatten_node_cap_data,
                             the_device_graph._flatten_node_cap_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data<double>),
                             cudaMemcpyDeviceToHost, stream[2]));

  CUDA_CHECK(cudaMemcpyAsync(the_host_graph._flatten_node_impulse_data,
                             the_device_graph._flatten_node_impulse_data,
                             vertex_data_size * sizeof(GPU_Fwd_Data<double>),
                             cudaMemcpyDeviceToHost, stream[3]));

  CUDA_CHECK(cudaMemcpyAsync(the_host_graph._flatten_arc_delay_data,
                             the_device_graph._flatten_arc_delay_data,
                             arc_data_size * sizeof(GPU_Fwd_Data<int64_t>),
                             cudaMemcpyDeviceToHost, stream[4]));

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
void gpu_propagate_fwd(GPU_Graph& the_host_graph, unsigned vertex_data_size,
                       unsigned arc_data_size,
                       std::map<unsigned, std::vector<unsigned>>& level_to_arcs,
                       Lib_Data_GPU& lib_data) {
  auto the_device_graph =
      copy_from_host_graph(the_host_graph, vertex_data_size, arc_data_size);

  for (auto& [level, the_arcs] : level_to_arcs) {
    GPU_BFS_Propagated_Arc bfs_arcs;
    bfs_arcs._num_arcs = the_arcs.size();
    CUDA_CHECK(cudaMalloc((void**)&bfs_arcs._arc_index,
                          bfs_arcs._num_arcs * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(bfs_arcs._arc_index, the_arcs.data(),
                          bfs_arcs._num_arcs * sizeof(unsigned),
                          cudaMemcpyHostToDevice));

    propagate_fwd<<<1, 1>>>(the_device_graph, lib_data, bfs_arcs);
    // wait to finish.
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaFree(bfs_arcs._arc_index));
  }

  copy_to_host_graph(the_host_graph, the_device_graph, vertex_data_size,
                    arc_data_size);
  CUDA_CHECK_ERROR();
}

}  // namespace ista