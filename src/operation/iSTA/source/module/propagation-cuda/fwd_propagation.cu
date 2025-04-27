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
#include <thrust/device_vector.h>

#include <map>

#include "fwd_propagation.cuh"
#include "gpu/cuda_common.cuh"
#include "include/Type.hh"
#include "propagation-cuda/lib_arc.cuh"
#include "propagation.cuh"

namespace ista {

/// each block thread num.
static const int THREAD_PER_BLOCK_NUM = 512;
static const unsigned max_arc_per_epoch = 25600;

/**
 * @brief copy host graph to gpu device graph.
 *
 */
GPU_Graph copy_from_host_graph(GPU_Graph& the_host_graph,
                               unsigned vertex_data_size,
                               unsigned arc_data_size) {
  CUDA_LOG_INFO("copy from host graph start");
  CUDA_PROF_START(0);

  CUDA_LOG_INFO("malloc device graph start");
  CUDA_CHECK_MEMORY(0, "before copy host graph to device graph");

  const unsigned num_stream = 8;
  cudaStream_t stream[num_stream];
  for (unsigned index = 0; index < num_stream; ++index) {
    cudaStreamCreate(&stream[index]);
  }

  GPU_Graph the_device_graph;

  the_device_graph._num_vertices = the_host_graph._num_vertices;
  the_device_graph._num_arcs = the_host_graph._num_arcs;

  unsigned stream_index = 0;
  CUDA_CHECK(cudaMallocAsync((void**)&the_device_graph._vertices,
                             the_host_graph._num_vertices * sizeof(GPU_Vertex),
                             stream[stream_index++]));

  CUDA_CHECK(cudaMallocAsync((void**)&the_device_graph._arcs,
                             the_host_graph._num_arcs * sizeof(GPU_Arc),
                             stream[stream_index++]));

  CUDA_CHECK(cudaMallocAsync(
      (void**)&the_device_graph._flatten_slew_data,
      the_host_graph._num_slew_data * sizeof(GPU_Fwd_Data<int64_t>),
      stream[stream_index++]));

  CUDA_CHECK(cudaMallocAsync(
      (void**)&the_device_graph._flatten_at_data,
      the_host_graph._num_at_data * sizeof(GPU_Fwd_Data<int64_t>),
      stream[stream_index++]));

  CUDA_CHECK(cudaMallocAsync(
      (void**)&the_device_graph._flatten_node_cap_data,
      the_host_graph._num_node_cap_data * sizeof(GPU_Fwd_Data<float>),
      stream[stream_index++]));

  CUDA_CHECK(cudaMallocAsync(
      (void**)&the_device_graph._flatten_node_delay_data,
      the_host_graph._num_node_delay_data * sizeof(GPU_Fwd_Data<float>),
      stream[stream_index++]));

  CUDA_CHECK(cudaMallocAsync(
      (void**)&the_device_graph._flatten_node_impulse_data,
      the_host_graph._num_node_impulse_data * sizeof(GPU_Fwd_Data<float>),
      stream[stream_index++]));

  CUDA_CHECK(cudaMallocAsync(
      (void**)&the_device_graph._flatten_arc_delay_data,
      the_host_graph._num_arc_delay_data * sizeof(GPU_Fwd_Data<int64_t>),
      stream[stream_index++]));

  assert(stream_index == num_stream);

  for (unsigned index = 0; index < num_stream; ++index) {
    cudaStreamSynchronize(stream[index]);
  }

  CUDA_LOG_INFO("malloc device graph end");
  CUDA_CHECK_MEMORY(1, "after copy host graph to device graph");

  stream_index = 0;
  CUDA_CHECK(cudaMemcpyAsync(the_device_graph._vertices,
                             the_host_graph._vertices,
                             the_host_graph._num_vertices * sizeof(GPU_Vertex),
                             cudaMemcpyHostToDevice, stream[stream_index++]));

  CUDA_CHECK(cudaMemcpyAsync(the_device_graph._arcs, the_host_graph._arcs,
                             the_host_graph._num_arcs * sizeof(GPU_Arc),
                             cudaMemcpyHostToDevice, stream[stream_index++]));

  CUDA_CHECK(cudaMemcpyAsync(
      the_device_graph._flatten_slew_data, the_host_graph._flatten_slew_data,
      the_host_graph._num_slew_data * sizeof(GPU_Fwd_Data<int64_t>),
      cudaMemcpyHostToDevice, stream[stream_index++]));

  CUDA_CHECK(cudaMemcpyAsync(
      the_device_graph._flatten_at_data, the_host_graph._flatten_at_data,
      the_host_graph._num_at_data * sizeof(GPU_Fwd_Data<int64_t>),
      cudaMemcpyHostToDevice, stream[stream_index++]));

  CUDA_CHECK(cudaMemcpyAsync(
      the_device_graph._flatten_node_cap_data,
      the_host_graph._flatten_node_cap_data,
      the_host_graph._num_node_cap_data * sizeof(GPU_Fwd_Data<float>),
      cudaMemcpyHostToDevice, stream[stream_index++]));

  CUDA_CHECK(cudaMemcpyAsync(
      the_device_graph._flatten_node_delay_data,
      the_host_graph._flatten_node_delay_data,
      the_host_graph._num_node_delay_data * sizeof(GPU_Fwd_Data<float>),
      cudaMemcpyHostToDevice, stream[stream_index++]));

  CUDA_CHECK(cudaMemcpyAsync(
      the_device_graph._flatten_node_impulse_data,
      the_host_graph._flatten_node_impulse_data,
      the_host_graph._num_node_impulse_data * sizeof(GPU_Fwd_Data<float>),
      cudaMemcpyHostToDevice, stream[stream_index++]));

  CUDA_CHECK(cudaMemcpyAsync(
      the_device_graph._flatten_arc_delay_data,
      the_host_graph._flatten_arc_delay_data,
      the_host_graph._num_arc_delay_data * sizeof(GPU_Fwd_Data<int64_t>),
      cudaMemcpyHostToDevice, stream[stream_index++]));

  assert(stream_index == num_stream);

  for (unsigned index = 0; index < num_stream; ++index) {
    cudaStreamSynchronize(stream[index]);
  }

  for (unsigned index = 0; index < num_stream; ++index) {
    cudaStreamDestroy(stream[index]);
  }

  CUDA_CHECK_ERROR();

  CUDA_LOG_INFO("copy from host graph end");
  CUDA_PROF_END(0, "host data copy to gpu");

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
  CUDA_LOG_INFO("copy to host graph start");
  CUDA_PROF_START(0);

  const unsigned num_stream = 3;
  cudaStream_t stream[num_stream];
  for (unsigned index = 0; index < num_stream; ++index) {
    cudaStreamCreate(&stream[index]);
  }

  CUDA_CHECK(cudaMemcpyAsync(
      the_host_graph._flatten_slew_data, the_device_graph._flatten_slew_data,
      the_host_graph._num_slew_data * sizeof(GPU_Fwd_Data<int64_t>),
      cudaMemcpyDeviceToHost, stream[0]));

  CUDA_CHECK(cudaMemcpyAsync(
      the_host_graph._flatten_at_data, the_device_graph._flatten_at_data,
      the_host_graph._num_at_data * sizeof(GPU_Fwd_Data<int64_t>),
      cudaMemcpyDeviceToHost, stream[1]));

  CUDA_CHECK(cudaMemcpyAsync(
      the_host_graph._flatten_arc_delay_data,
      the_device_graph._flatten_arc_delay_data,
      the_host_graph._num_arc_delay_data * sizeof(GPU_Fwd_Data<int64_t>),
      cudaMemcpyDeviceToHost, stream[2]));

  for (unsigned index = 0; index < num_stream; ++index) {
    cudaStreamSynchronize(stream[index]);
  }

  for (unsigned index = 0; index < num_stream; ++index) {
    cudaStreamDestroy(stream[index]);
  }

  CUDA_LOG_INFO("copy to host graph end");
  CUDA_PROF_END(0, "gpu data copy to host");

  cudaFree(the_device_graph._vertices);
  cudaFree(the_device_graph._arcs);
  cudaFree(the_device_graph._flatten_slew_data);
  cudaFree(the_device_graph._flatten_at_data);
  cudaFree(the_device_graph._flatten_node_cap_data);
  cudaFree(the_device_graph._flatten_node_delay_data);
  cudaFree(the_device_graph._flatten_node_impulse_data);
  cudaFree(the_device_graph._flatten_arc_delay_data);

  CUDA_CHECK_ERROR();

}

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
__device__ inline std::pair<GPU_Fwd_Data<T>*, int> get_one_fwd_data(
    GPU_Fwd_Data<T>* flatten_all_datas, GPU_Vertex_Data* the_vertex_data,
    GPU_Analysis_Mode analysis_mode, GPU_Trans_Type trans_type,
    int own_clock_index) {
  for (unsigned i = 0; i < the_vertex_data->_num_fwd_data; ++i) {
    int data_index = the_vertex_data->_start_pos + i;
    auto& fwd_data = flatten_all_datas[data_index];
    if (fwd_data._analysis_mode == analysis_mode &&
        fwd_data._trans_type == trans_type) {
      if (own_clock_index == -1 ||
          fwd_data._own_clock_index == own_clock_index) {
        return std::make_pair(&fwd_data, data_index);
      }
    }
  }

  CUDA_LOG_ERROR("not found the fwd data start pos %d, num data %d",
                 the_vertex_data->_start_pos, the_vertex_data->_num_fwd_data);

  return std::make_pair(nullptr, -1);
}

/**
 * @brief update the snk fwd data.
 * 
 * @tparam op 
 * @param src_fwd_data 
 * @param snk_fwd_data 
 * @param src_vertex_id 
 * @param src_data_index 
 * @param analysis_mode 
 * @param data_value 
 * @return __device__ 
 */
template <GPU_OP_TYPE op>
__device__ void update_fwd_data(GPU_Fwd_Data<int64_t>& src_fwd_data,
                                GPU_Fwd_Data<int64_t>& snk_fwd_data,
                                unsigned src_vertex_id, int src_data_index,
                                int snk_data_index,
                                GPU_Analysis_Mode analysis_mode,
                                int64_t data_value) {
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
        src_fwd_data._snk_data_index = snk_data_index;
        snk_fwd_data._data_value = data_value;
      }
    } else {
      if (snk_fwd_data._data_value < (src_fwd_data._data_value + data_value)) {
        snk_fwd_data._src_vertex_id = src_vertex_id;
        snk_fwd_data._src_data_index = src_data_index;
        src_fwd_data._snk_data_index = snk_data_index;
        snk_fwd_data._data_value = src_fwd_data._data_value + data_value;

        CUDA_LOG_DEBUG("update max src vertex %d -> snk vertex %d at %lld",
                       src_vertex_id, snk_vertex_id, snk_fwd_data._data_value);
      }
    }
  } else {
    if (op != GPU_OP_TYPE::kAT) {
      if ((snk_fwd_data._data_value == 0.0) ||
          (snk_fwd_data._data_value > data_value)) {
        snk_fwd_data._src_vertex_id = src_vertex_id;
        snk_fwd_data._src_data_index = src_data_index;
        src_fwd_data._snk_data_index = snk_data_index;
        snk_fwd_data._data_value = data_value;
      }
    } else {
      if ((snk_fwd_data._data_value == 0.0) ||
          (snk_fwd_data._data_value >
           (src_fwd_data._data_value + data_value))) {
        snk_fwd_data._src_vertex_id = src_vertex_id;
        snk_fwd_data._src_data_index = src_data_index;
        src_fwd_data._snk_data_index = snk_data_index;
        snk_fwd_data._data_value = src_fwd_data._data_value + data_value;

        CUDA_LOG_DEBUG("update min src vertex %d -> snk vertex %d at %lld",
                       src_vertex_id, snk_vertex_id, snk_fwd_data._data_value);
      }
    }
  }

  // release the lock.
  atomicExch(&snk_fwd_data._is_lock, 0);
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

  if (is_force) {
    // update check value should be direct.
    auto [snk_fwd_data_ptr, snk_data_index] = get_one_fwd_data(
        flatten_all_datas, snk_vertex_data, analysis_mode, out_trans_type, -1);
    if (!snk_fwd_data_ptr) {
      CUDA_LOG_ERROR("the arc %d -> %d, snk fwd data is null.", src_vertex_id,
                     snk_vertex_id);
    } else {
      snk_fwd_data_ptr->_data_value = data_value;
    }

    return;
  }

  for (unsigned i = 0; i < src_vertex_data->_num_fwd_data; ++i) {
    int src_data_index = src_vertex_data->_start_pos + i;
    auto& src_fwd_data = flatten_all_datas[src_data_index];
    if (src_fwd_data._analysis_mode == analysis_mode &&
        src_fwd_data._trans_type == in_trans_type) {
      int own_clock_index = src_fwd_data._own_clock_index;

      auto [snk_fwd_data_ptr, snk_data_index] = get_one_fwd_data(
          flatten_all_datas, snk_vertex_data, analysis_mode, out_trans_type, own_clock_index);
      if (!snk_fwd_data_ptr) {
        CUDA_LOG_ERROR("the arc %d -> %d, snk fwd data is null.", src_vertex_id,
                       snk_vertex_id);
        continue;
      }

      update_fwd_data<op>(src_fwd_data, *snk_fwd_data_ptr, src_vertex_id,
                          src_data_index, snk_data_index, analysis_mode, data_value);
    }
  }
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

  auto find_slew_delay = [the_lib_arc, the_arc](auto in_trans_type,
                                                auto& one_src_slew_data,
                                                auto& one_snk_cap_data) {
    unsigned table_index =
        GPU_Table_Base_Index::kTransitionBase + in_trans_type;
    if (table_index >= the_lib_arc._num_table) {
      CUDA_LOG_FATAL(
          "table index %d beyond num table %d table line no %d lib arc id %d",
          table_index, the_lib_arc._num_table, the_lib_arc._line_no,
          the_arc._lib_data_arc_id);
    }
    auto& the_slew_lib_table = the_lib_arc._table[table_index];

    auto src_slew_ns = FS_TO_NS(one_src_slew_data._data_value);
    float snk_cap_load = one_snk_cap_data._data_value;  // default is PF
    if (the_lib_arc._cap_unit == Lib_Cap_unit::kFF) {
      snk_cap_load = float(PF_TO_FF(snk_cap_load));  // change to FF
    }
    float found_slew_value =
        find_value(the_slew_lib_table, src_slew_ns, snk_cap_load);
    CUDA_LOG_DEBUG(
        "find slew value %f src slew %f snk cap %f table line no %d "
        "lib arc id %d",
        found_slew_value, src_slew_ns, snk_cap_load, the_lib_arc._line_no,
        the_arc._lib_data_arc_id);

    table_index = GPU_Table_Base_Index::kDelayBase + in_trans_type;
    if (table_index >= the_lib_arc._num_table) {
      CUDA_LOG_FATAL(
          "table index %d beyond num table %d table line no %d lib arc id %d",
          table_index, the_lib_arc._num_table, the_lib_arc._line_no,
          the_arc._lib_data_arc_id);
    }
    auto& the_delay_lib_table = the_lib_arc._table[table_index];
    float found_delay_value =
        find_value(the_delay_lib_table, src_slew_ns, snk_cap_load);

    CUDA_LOG_DEBUG(
        "find delay value %f src slew %f snk cap %f table line no %d "
        "lib arc id %d",
        found_delay_value, src_slew_ns, snk_cap_load, the_lib_arc._line_no,
        the_arc._lib_data_arc_id);
    
    if (the_lib_arc._time_unit == Lib_Time_unit::kNS) {
        return std::pair(int64_t(NS_TO_FS(found_slew_value)),
                     int64_t(NS_TO_FS(found_delay_value)));
    } else if (the_lib_arc._time_unit == Lib_Time_unit::kPS) {
        return std::pair(int64_t(PS_TO_FS(found_slew_value)),
                     int64_t(PS_TO_FS(found_delay_value)));
    } else {
      // should be FS
        return std::pair(int64_t(found_slew_value),
                     int64_t(found_delay_value));
    }

  };

  GPU_Fwd_Data<int64_t> one_src_slew_data;
  GPU_Fwd_Data<float> one_snk_cap_data;
  FOREACH_GPU_FWD_DATA(the_graph->_flatten_slew_data, (*in_slew),
                       one_src_slew_data) {
    GPU_Trans_Type in_trans_type = one_src_slew_data._trans_type;
    GPU_Analysis_Mode analysis_mode = one_src_slew_data._analysis_mode;
    auto out_trans_type = in_trans_type;
    if (the_arc_trans_type == GPU_Arc_Trans_Type::kNegative) {
      out_trans_type = GPU_FLIP_TRANS(in_trans_type);
    }
    auto [one_snk_cap_data_ptr, snk_cap_index] =
        get_one_fwd_data(the_graph->_flatten_node_cap_data, out_load,
                         analysis_mode, out_trans_type, -1);
    auto& one_snk_cap_data = *one_snk_cap_data_ptr;
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
      auto [one_snk_cap_data1_ptr, snk_cap_index1] =
          get_one_fwd_data(the_graph->_flatten_node_cap_data, out_load,
                           analysis_mode, out_trans_type, -1);
      auto& one_snk_cap_data1 = *one_snk_cap_data1_ptr;
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

      unsigned table_index = snk_trans;
      auto& the_lib_table = the_lib_arc._table[snk_trans];

      if (table_index >= the_lib_arc._num_table) {
        CUDA_LOG_FATAL(
            "table index %d beyond num table %d table line no %d lib arc id %d",
            table_index, the_lib_arc._num_table, the_lib_arc._line_no,
            the_arc._lib_data_arc_id);
      }

      float src_slew_ns = FS_TO_NS(one_src_slew_data._data_value);
      float snk_slew_ns = FS_TO_NS(one_snk_slew_data._data_value);

      float found_delay_value =
          find_value(the_lib_table, src_slew_ns, snk_slew_ns);
      CUDA_LOG_DEBUG(
          "find check value %f src slew %f snk slew %f table line no %d "
          "lib arc id %d",
          found_delay_value, src_slew_ns, snk_slew_ns, the_lib_arc._line_no,
          the_arc._lib_data_arc_id);

      int64_t delay_value = (the_lib_arc._time_unit == Lib_Time_unit::kNS)
                                ? NS_TO_FS(found_delay_value)
                            : (the_lib_arc._time_unit == Lib_Time_unit::kPS)
                                ? PS_TO_FS(found_delay_value)
                                : found_delay_value;
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
      auto [one_snk_impulse_data_ptr, impulse_data_index] =
          get_one_fwd_data(the_graph->_flatten_node_impulse_data, impulse_data,
                           analysis_mode, in_trans_type, -1);
      auto& one_snk_impulse_data = *one_snk_impulse_data_ptr;
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
      GPU_Fwd_Data<float> one_snk_delay_data;
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
    // CUDA_LOG_DEBUG("GPU thread %d propagate fwd in gpu kernel", i);

    unsigned current_arc_id = propagated_arcs._arc_index[i];
    // CUDA_LOG_DEBUG("current arc id %d", current_arc_id);

    GPU_Arc current_arc = the_graph._arcs[current_arc_id];
    GPU_Arc_Type current_arc_type = current_arc._arc_type;

    int lib_arc_id = current_arc._lib_data_arc_id;
    // CUDA_LOG_DEBUG("current lib arc id %d", lib_arc_id);

    if (current_arc_type == kInstDelayArc) {
      CUDA_LOG_DEBUG("process inst delay arc id %d lib arc id %d",
                     current_arc_id, lib_arc_id);
      auto lib_arc = the_lib_data._arcs_gpu[lib_arc_id];
      // for inst delay arc.
      propagate_inst_slew_delay(&the_graph, current_arc, lib_arc);
    } else if (current_arc_type == kInstCheckArc) {
      CUDA_LOG_DEBUG("process inst check arc id %d lib arc id %d",
                     current_arc_id, lib_arc_id);
      auto lib_arc = the_lib_data._arcs_gpu[lib_arc_id];
      // for inst check arc, lut table for get constrain value.
      lut_constraint_delay(&the_graph, current_arc, lib_arc);
    } else {
      CUDA_LOG_DEBUG("process net arc");
      // for net arc, lut net output slew and delay.
      propagate_net_slew_delay(&the_graph, current_arc);
    }
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

  int device_id = 0;
  CUDA_CHECK(cudaSetDevice(device_id));
  auto the_device_graph =
      copy_from_host_graph(the_host_graph, vertex_data_size, arc_data_size);

  CUDA_PROF_START(0);
  for (auto& [level, the_arcs] : level_to_arcs) {
    unsigned the_level_arc_size = the_arcs.size();
    CUDA_LOG_INFO("propagate fwd level %d level size %d", level,
                  the_level_arc_size);

    int num_epoch =
        (the_level_arc_size + max_arc_per_epoch - 1) / max_arc_per_epoch;
    // split to epoch for large arc size.
    for (int epoch_index = 0; epoch_index < num_epoch; ++epoch_index) {
      CUDA_LOG_INFO("propagate fwd epoch %d total epoch %d", epoch_index,
                    num_epoch);

      GPU_BFS_Propagated_Arc bfs_arcs_epoch;
      if (num_epoch == 1) {
        bfs_arcs_epoch._num_arcs = the_level_arc_size;
      } else {
        bfs_arcs_epoch._num_arcs =
            std::min(max_arc_per_epoch,
                     the_level_arc_size - epoch_index * max_arc_per_epoch);
      }

      thrust::device_vector<unsigned> bfs_arc_vec(bfs_arcs_epoch._num_arcs);
      unsigned i = 0;
      std::for_each_n(
          the_arcs.begin() + epoch_index * max_arc_per_epoch,
          bfs_arcs_epoch._num_arcs,
          [&bfs_arc_vec, &i](auto arc_index) { bfs_arc_vec[i++] = arc_index; });

      bfs_arcs_epoch._arc_index = thrust::raw_pointer_cast(bfs_arc_vec.data());

      int num_blocks = (bfs_arcs_epoch._num_arcs + THREAD_PER_BLOCK_NUM - 1) /
                       THREAD_PER_BLOCK_NUM;

      CUDA_LOG_INFO(
          "propagate fwd kernal num blocks %d per each block %d threads start",
          num_blocks, THREAD_PER_BLOCK_NUM);

      propagate_fwd<<<num_blocks, THREAD_PER_BLOCK_NUM>>>(
          the_device_graph, lib_data, bfs_arcs_epoch);
      // wait to finish.
      cudaDeviceSynchronize();

      CUDA_LOG_INFO(
          "propagate fwd kernal num blocks %d per each block %d threads end",
          num_blocks, THREAD_PER_BLOCK_NUM);

      CUDA_CHECK_ERROR();
    }
  }

  CUDA_PROF_END(0, "propagate fwd kernel");

  copy_to_host_graph(the_host_graph, the_device_graph, vertex_data_size,
                     arc_data_size);
  CUDA_CHECK_ERROR();
}

}  // namespace ista