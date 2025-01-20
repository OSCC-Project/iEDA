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

#include "fwd_propagation.cuh"
// #include "sta/StaGraph.hh"

namespace ista {

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
__device__ void lut_inst_slew_delay(GPU_Vertex_Data* in_slew,
                                    GPU_Vertex_Data* out_load,
                                    GPU_Vertex_Data* snk_slew,
                                    GPU_Vertex_Data* arc_delay) {
  // TODO(to taosimin), call gpu lut table.
  // store the lut value
}

/**
 * @brief device function for lut check arc, using slew and snk slew.
 *
 * @param in_slew
 * @param snk_slew
 * @param arc_delay
 * @return __device__
 */
__device__ void lut_constraint_delay(GPU_Vertex_Data* in_slew,
                                     GPU_Vertex_Data* snk_slew,
                                     GPU_Vertex_Data* arc_delay) {
  // TODO(to taosimin), call gpu lut table.
  // store the lut value
  GPU_Fwd_Data one_src_slew_data;
  FOREACH_GPU_FWD_DATA((*in_slew), one_src_slew_data) {}
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
__global__ void propagate_fwd(GPU_Graph the_graph,
                              GPU_BFS_Propagated_Arc propagated_arcs) {
  // current thread id
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < propagated_arcs._num_arcs) {
    unsigned current_arc_id = propagated_arcs._arc_start_addr[i];
    GPU_Arc current_arc = the_graph._arcs[current_arc_id];
    GPU_Arc_Type current_arc_type = current_arc._arc_type;
    unsigned src_vertex_id = current_arc._src_vertex_id;
    unsigned snk_vertex_id = current_arc._snk_vertex_id;
    auto src_vertex = the_graph._vertices[src_vertex_id];
    auto snk_vertex = the_graph._vertices[snk_vertex_id];

    if (current_arc_type == kInstDelayArc) {
      // lut table for snk arc slew and arc delay use src slew and out cap.
      lut_inst_slew_delay(&src_vertex._slew_data, &snk_vertex._node_cap_data,
                          &snk_vertex._slew_data, &current_arc._delay_values);
    } else if (current_arc_type == kInstCheckArc) {
      // lut table for get constrain value for check arc.
      lut_constraint_delay(&src_vertex._slew_data, &snk_vertex._slew_data,
                           &current_arc._delay_values);
    } else {
      // for net arc
      // lut net output slew and delay.
      lut_net_slew_delay(&src_vertex._slew_data, &snk_vertex._node_delay_data,
                         &snk_vertex._node_impulse_data, &snk_vertex._slew_data,
                         &current_arc._delay_values);
    }
  }
}
#if 0
/**
 * @brief copy sta graph to gpu sta graph.
 *
 */
GPU_Graph build_gpu_sta_graph(StaGraph* the_cpu_graph) {
  GPU_Graph the_gpu_graph;
  return the_gpu_graph;
}

/**
 * @brief copyback gpu data to cpu sta graph.
 *
 * @param the_cpu_graph
 * @param the_gpu_graph
 */
void update_sta_graph(StaGraph* the_cpu_graph, GPU_Graph the_gpu_graph) {}

/**
 * @brief The interface function for the fwd function.
 * first, build gpu graph
 * for vertex, copy slew data, load, at data, node delay, node impulse
 * for arc, set src and snk id
 * then, propagate level by level.
 */
void gpu_propagate_fwd(
    StaGraph* the_cpu_graph,
    std::map<unsigned, std::vector<StaArc*>>& level_to_arcs) {
  auto the_gpu_graph = build_gpu_sta_graph(the_cpu_graph);
  GPU_BFS_Propagated_Arc propagate_arcs;
  //TODO(to taosimin), copy arc id to gpu bfs propagated arc.
  for (auto& [level, the_arcs] : level_to_arcs) {
    propagate_fwd<<<1, 1000>>>(the_gpu_graph, propagate_arcs);
  }

  update_sta_graph(the_cpu_graph, the_gpu_graph);
}
#endif

}  // namespace ista