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

#include <vector>
#include <memory>
#include <variant>

namespace istagpu {

    /**
 * @brief The rc node for rc tree, represent for capacitance.
 *
 */
struct DelayRcPoint {
  const char* _node_name;
  float _cap;
  float _load;  //!< The load is sum of the node cap and downstream node cap.
};

typedef size_t DelayRcPointId;

/**
 * @brief The rc edge for rc tree, represent for resitance.
 *
 */
struct DelayRcEdge {
  DelayRcPointId _from;  // The from node id.
  DelayRcPointId _to;    // The to node id.

  float _resistance;
};

/**
 * @brief The rc tree of one net.
 *
 */
struct DelayRcNetwork {
    DelayRcPoint* _root{nullptr};
    std::vector<std::unique_ptr<DelayRcPoint>> _nodes;
    std::vector<DelayRcEdge> _edges;
};



/**
 * @brief The rc net wrap for rc tree.
 * 
 */
struct DelayRcNet {
    DelayRcNetwork _rc_network;

    float delay_update_point_load();

};




}

