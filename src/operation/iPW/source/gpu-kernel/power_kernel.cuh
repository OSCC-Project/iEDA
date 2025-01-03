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
 * @file power_kernel.cuh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The power gpu kernel for speed up power tool.
 * @version 0.1
 * @date 2024-10-09
 * 
 */
#pragma once

#include <cuda_runtime.h>
#include "gpu/kernel_common.h"

namespace ipower {

/**
 * @brief for gpu kernel function used to store connection point.
 * 
 */
struct GPU_Connection_Point {
    int _src_id = -1;
    int _snk_id = -1;
    unsigned _last_depth; // The last hop combine depth.
};


/**
 * @brief build macro connection map with gpu kernel function.
 * 
 * @param connection_points 
 * @param is_macros 
 * @param seq_arcs 
 * @param snk_depths 
 * @param snk_arcs 
 * @param connection_point_num 
 * @param num_seq_vertexes 
 * @param num_seq_arcs 
 * @param out_connection_points 
 */
void build_macro_connection_map(GPU_Connection_Point* connection_points,
                                unsigned* is_macros, unsigned* seq_arcs,
                                unsigned* snk_depths, unsigned* snk_arcs,
                                int connection_point_num, int num_seq_vertexes,
                                int num_seq_arcs,
                                GPU_Connection_Point* out_connection_points, bool is_free_memory);

}