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
 * @file fwd_propagation.cuh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The fwd propagation using GPU. 
 * @version 0.1
 * @date 2025-01-15
 *
 */
#pragma once

#include <variant>

namespace ista {

/**
 * @brief The signal trans type.
 *
 */
enum GPU_Trans_Type { kRise = 0, kFall = 1 };

/**
 * @brief The analysis mode.
 *
 */
enum GPU_Analysis_Mode { kMax = 0, kMin = 0 };

/**
 * @brief The arc type.
 *
 */
enum GPU_Arc_Type { kInst = 0, kNet = 1 };

/**
 * @brief The fwd data common type.
 *
 */
struct GPU_Fwd_Data {
  double _data_value;
  GPU_Trans_Type _trans_type;
  GPU_Analysis_Mode _analysis_mode;
};

/**
 * @brief The vertex fwd data vector.
 *
 */
struct GPU_Vertex_Data {
  GPU_Fwd_Data* _fwd_data;
  unsigned _num_fwd_data;
};

/**
 * @brief The vertex in GPU mapping with StaVertex.
 *
 */
struct GPU_Vertex {
  GPU_Vertex_Data _slew_data;          //!< The slew data of the vertex.
  GPU_Vertex_Data _at_data;            //!< The arrive data of the vertex.
  GPU_Vertex_Data _load_cap_data;      //!< The driver load cap.
  GPU_Vertex_Data _node_delay_data;    //!< The load pin node delay data.
  GPU_Vertex_Data _node_impulse_data;  //!< The load pin node impulse data for
                                       //!< calculate impulse data.
};

/**
 * @brief The arc in GPU mapping with the StaArc.
 *
 */
struct GPU_Arc {
  GPU_Arc_Type _arc_type;  //!< The arc type inst or net arc.
  unsigned _arc_id;  //!< inst arc id or net arc id mapping the host StaArc.
  unsigned _src_vertex_id;  //!< The src vertex id mapping the host StaVertex.
  unsigned _snk_vertex_id;  //!< The snk vertex id mapping the host StaVertex.
};

/**
 * @brief The gpu graph mapping wht the StaGraph.
 * 
 */
struct GPU_Graph {
    GPU_Vertex* _vertices;  //!< The vertex data on GPU.
    GPU_Arc* _arcs;        //!< The arc data on GPU.
    unsigned _num_vertices;  //!< The number of vertices.
    unsigned _num_arcs;    //!< The number of arcs.
};


}  // namespace ista