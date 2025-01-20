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

namespace ista {

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

    GPU_Fwd_Data one_src_slew_data;
    FOREACH_GPU_FWD_DATA((src_vertex._slew_data), one_src_slew_data) {
      if (current_arc_type == kInstDelayArc) {
        // lut table for snk arc slew and arc delay use src slew and out cap.
      } else if (current_arc_type == kInstCheckArc) {
        GPU_Fwd_Data one_snk_slew_data;
        FOREACH_GPU_FWD_DATA((snk_vertex._slew_data), one_snk_slew_data) {
            // lut table for get constrain value for check arc
        }
      } else {
        // for net arc
        // lut net output slew and delay.

      }
    }
  }
}



}  // namespace ista