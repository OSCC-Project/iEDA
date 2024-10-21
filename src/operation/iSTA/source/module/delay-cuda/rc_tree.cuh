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
 * @file rc_tree.cuh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The rc tree datastrucure for delay calculation.
 * @version 0.1
 * @date 2024-09-25
 */

#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <variant>
#include <vector>

namespace istagpu {

struct DelayRcEdge;

/**
 * @brief The rc node for rc tree, represent for capacitance.
 *
 */
struct DelayRcPoint {
  const char* _node_name = nullptr;
  float _cap = 0.0;
  float _nload =
      0.0;  //!< The nload is sum of the node cap and downstream node cap.
  float _ndelay = 0.0;  //!< The ndelay is the time from root to this node.
  float _ldelay = 0.0;  //!< The load is sum of the node cap*ndelay and
  //!< downstream node cap*ndelay.
  float _beta = 0.0;
  float _impulse = 0.0;  //!< The delay is the output slew.

  std::size_t _flatten_pos = 0;

  bool _is_update_load = false;
  bool _is_update_delay = false;
  bool _is_update_ldelay = false;
  bool _reserved = false;

  DelayRcPoint* _parent =
      nullptr;  //!< For the tree, the root is the top parent.
  std::vector<DelayRcEdge*> _fanin_edges;  //!< The fanin edge to the rc point.
  std::vector<DelayRcEdge*>
      _fanout_edges;  //!< The fanout edge from the rc point.
};

/**
 * @brief The rc edge for rc tree, represent for resitance.
 *
 */
struct DelayRcEdge {
  DelayRcPoint* _from;  // The from node id.
  DelayRcPoint* _to;    // The to node id.

  float _resistance;
};

/**
 * @brief The rc tree of one net.
 *
 */
struct DelayRcNetwork {
  DelayRcPoint* _root{nullptr};
  std::vector<std::unique_ptr<DelayRcPoint>> _nodes;
  std::vector<std::unique_ptr<DelayRcEdge>> _edges;

  std::vector<float> _cap_array;
  std::vector<float>
      _load_array;  // TODO(to taosimin), load and delay may be use one array.
  std::vector<float> _resistance_array;
  std::vector<float> _delay_array;
  std::vector<float> _ldelay_array;
  std::vector<float> _beta_array;
  std::vector<float> _impulse_array;
  std::vector<int> _parent_pos_array;
  std::vector<int> _children_pos_array;

  float* _gpu_cap_array = nullptr;
  float* _gpu_load_array = nullptr;
  float* _gpu_resistance_array = nullptr;
  float* _gpu_delay_array = nullptr;
  float* _gpu_ldelay_array = nullptr;
  float* _gpu_beta_array = nullptr;
  float* _gpu_impulse_array = nullptr;
  int* _gpu_parent_pos_array = nullptr;
  int* _gpu_children_pos_array = nullptr;

  std::size_t get_node_num() { return _nodes.size(); }
};

/**
 * @brief The rc net wrap for rc tree.
 *
 */
struct DelayRcNet {
  DelayRcNetwork _rc_network;
};

#if 1
std::vector<std::vector<DelayRcPoint*>> delay_levelization(
    DelayRcNetwork* rc_network);
void delay_update_point_load(std::vector<std::vector<DelayRcPoint*>>);

#else
float delay_update_point_load(DelayRcPoint* parent, DelayRcPoint* rc_point);
void delay_update_point_load(DelayRcNet* rc_net);

void delay_update_point_delay(DelayRcPoint* parent, DelayRcPoint* rc_net);
void delay_update_point_delay(DelayRcNet* rc_net);

#endif

////////////////////////////////////////////////////

int test();

}  // namespace istagpu
