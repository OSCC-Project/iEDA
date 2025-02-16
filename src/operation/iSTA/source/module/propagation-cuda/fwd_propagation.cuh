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
#include <map>
#include "propagation-cuda/propagation.cuh"

namespace ista {

/**
 * @brief The signal trans type.
 *
 */
enum GPU_Trans_Type { kRise = 0, kFall = 1 };

#define GPU_FLIP_TRANS(trans)                                  \
  (trans == GPU_Trans_Type::kRise) ? GPU_Trans_Type::kFall \
                                   : GPU_Trans_Type::kRise

/**
 * @brief The analysis mode.
 *
 */
enum GPU_Analysis_Mode { kMax = 0, kMin };

/**
 * @brief The arc type.
 *
 */
enum GPU_Arc_Type { kInstDelayArc = 0, kInstCheckArc, kNet };

enum GPU_Arc_Trans_Type { kPositive = 0, kNegative, kNonUnate };

enum GPU_Table_Type {
  kCellRise = 0,
  kCellFall = 1,
  kRiseTransition = 2,
  kFallTransition = 3,
  kRiseConstrain = 4,
  kFallConstrain = 5,
  kRiseCurrent = 6,
  kFallCurrent = 7,
  // power
  kRisePower = 8,
  kFallPower = 9,
  // sigma
  kCellRiseSigma = 10,
  kCellFallSigma = 12,
  kRiseTransitionSigma = 14,
  kFallTransitionSigma = 16
};

enum GPU_Table_Base_Index {
  kDelayBase = GPU_Table_Type::kCellRise,
  kTransitionBase = GPU_Table_Type::kRiseTransition,
  kCheckBase = GPU_Table_Type::kRiseConstrain
};

/**
 * @brief The operator for gpu data update.
 * 
 */
enum GPU_OP_TYPE {
  kSlew = 0,
  kDelay,
  kAT
};

/**
 * @brief The fwd data common type.
 *
 */
template<typename T>
struct GPU_Fwd_Data {
  T _data_value = 0.0; //!< unit is FS.
  GPU_Trans_Type _trans_type;  //!< for purposes of more gpu fwd data, so we
                               //!< record trans_type and analysis mode.
  GPU_Analysis_Mode _analysis_mode;
  int _src_vertex_id = -1; //! The src vertex id for backtrace analysis.
  int _src_data_index = -1; //!< The src data index in flatten datas.
  int _snk_data_index = -1; //!< The snk data index in flatten datas.
  int _own_clock_index = -1; //!< The launch clock index of path data.

  int _is_lock = 0; //!< 0 is not lock.
};

/**
 * @brief The vertex fwd data vector.
 *
 */
struct GPU_Vertex_Data {
  unsigned _start_pos = 0;
  unsigned _num_fwd_data = 0;
};

/**
 * @brief The macro of foreach gpu vertex, usage:
 * GPU_Vertex_Data* the_datas;
 * GPU_Fwd_Data one_data;
 * FOREACH_GPU_FWD_DATA(the_datas, one_data)
 * {
 *    do_something_for_data();
 * }
 */
#define FOREACH_GPU_FWD_DATA(all_datas, the_datas, one_data)             \
  for (unsigned i = 0; (i < the_datas._num_fwd_data)                     \
                       ? one_data = all_datas[the_datas._start_pos + i], \
                true : false;                                            \
       ++i)



constexpr unsigned c_gpu_num_vertex_data = 4;
constexpr unsigned c_gpu_num_node_data = 4;
/**
 * @brief The vertex in GPU mapping with StaVertex.
 *
 */
struct GPU_Vertex {
  GPU_Vertex_Data _slew_data;          //!< The slew data of the vertex.
  GPU_Vertex_Data _at_data;            //!< The arrive data of the vertex.
  GPU_Vertex_Data _node_cap_data;      //!< The driver load cap.
  GPU_Vertex_Data _node_delay_data;    //!< The load pin node delay data.
  GPU_Vertex_Data _node_impulse_data;  //!< The load pin node impulse data
                                       //!< for calculate impulse data.
};

constexpr unsigned c_gpu_num_arc_delay = 4;
using GPU_ARC_DATA = GPU_Vertex_Data;
/**
 * @brief The arc in GPU mapping with the StaArc.
 *
 */
struct GPU_Arc {
  GPU_Arc_Type _arc_type;              //!< The arc type inst or net arc.
  GPU_Arc_Trans_Type _arc_trans_type;  //!< The arc trans type.
  unsigned _src_vertex_id;  //!< The src vertex id mapping the host StaVertex.
  unsigned _snk_vertex_id;  //!< The snk vertex id mapping the host StaVertex.
  GPU_ARC_DATA _delay_values;

  int _lib_data_arc_id = -1;  //!< The lib data arc id.
};

/**
 * @brief struct for collect all data.
 *
 */
struct GPU_Flatten_Data {
  std::vector<GPU_Fwd_Data<int64_t>> _flatten_slew_data;
  std::vector<GPU_Fwd_Data<int64_t>> _flatten_at_data;
  std::vector<GPU_Fwd_Data<float>> _flatten_node_cap_data;
  std::vector<GPU_Fwd_Data<float>> _flatten_node_delay_data;
  std::vector<GPU_Fwd_Data<float>> _flatten_node_impulse_data;
  std::vector<GPU_Fwd_Data<int64_t>> _flatten_arc_delay_data;
};

/**
 * @brief The bfs propagated arcs.
 *
 */
struct GPU_BFS_Propagated_Arc {
  unsigned* _arc_index =
      nullptr;  //!< The arc start address, each one is arc id.
  unsigned _num_arcs;
};

/**
 * @brief The gpu graph mapping wht the StaGraph.
 *
 */
struct GPU_Graph {
  GPU_Vertex* _vertices = nullptr;  //!< The vertex data on GPU.
  GPU_Arc* _arcs = nullptr;         //!< The arc data on GPU.
  unsigned _num_vertices = 0;       //!< The number of vertices.
  unsigned _num_arcs = 0;           //!< The number of arcs.

  // flatten data for copy data from cpu to gpu faster.
  GPU_Fwd_Data<int64_t>* _flatten_slew_data = nullptr;  //!< The all slew data of the vertex.
  unsigned _num_slew_data = 0; //!< The number of slew data.
  GPU_Fwd_Data<int64_t>* _flatten_at_data = nullptr;    //!< The all arrive data of the vertex.
  unsigned _num_at_data = 0; //!< The number of arrive data.
  GPU_Fwd_Data<float>*
      _flatten_node_cap_data = nullptr;  //!< The all node cap data of the vertex.
  unsigned _num_node_cap_data = 0; //!< The number of node cap data.
  GPU_Fwd_Data<float>*
      _flatten_node_delay_data = nullptr;  //!< The all node delay data of the vertex.
  unsigned _num_node_delay_data = 0; //!< The number of node delay data.
  GPU_Fwd_Data<float>*
      _flatten_node_impulse_data = nullptr;  //!< The all node impulse data of the vertex.
  unsigned _num_node_impulse_data = 0; //!< The number of node impulse data.

  GPU_Fwd_Data<int64_t>* _flatten_arc_delay_data = nullptr;  //!< The all arc delay data.
  unsigned _num_arc_delay_data = 0; //!< The number of arc delay data.
};

/**
 * @brief The gpu fwd propagation top api.
 * 
 * @param the_cpu_graph  The gpu graph in cpu.
 * @param vertex_data_size The flatten vertex data size.
 * @param arc_data_size The flatten arc data size.
 * @param level_to_arcs The BFS propagation arcs for each level.
 * @param lib_data The lut lib data.
 */
void gpu_propagate_fwd(
    GPU_Graph& the_cpu_graph, unsigned vertex_data_size, unsigned arc_data_size,
    std::map<unsigned, std::vector<unsigned>>& level_to_arcs,
    Lib_Data_GPU& lib_data);

}  // namespace ista