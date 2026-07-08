// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file FastStaClockNetParasitic.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief CTS fast STA clock-net RC tree and Pi parasitic records.
 */

#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "FastSta.hh"

namespace icts {

struct FastStaPointKeyHash
{
  auto operator()(const std::pair<int, int>& key) const -> std::size_t
  {
    const auto x_hash = std::hash<int>{}(key.first);
    const auto y_hash = std::hash<int>{}(key.second);
    return x_hash ^ (y_hash + 0x9e3779b9U + (x_hash << 6U) + (x_hash >> 2U));
  }
};

struct FastStaPiModel
{
  double near_cap_pf = 0.0;
  double resistance_ohm = 0.0;
  double far_cap_pf = 0.0;
};

struct FastStaRcNode
{
  std::string name;
  double wire_cap_pf = 0.0;
  double pin_cap_pf = 0.0;
  double cap_pf = 0.0;
  double downstream_cap_pf = 0.0;
  double elmore_delay_ns = 0.0;
  FastStaNodeId terminal_node_id = kInvalidFastStaNodeId;
};

struct FastStaRcEdge
{
  FastStaRcNodeId from = kInvalidFastStaRcNodeId;
  FastStaRcNodeId to = kInvalidFastStaRcNodeId;
  double resistance_ohm = 0.0;
};

struct FastStaNetParasitic
{
  std::vector<FastStaRcNode> rc_nodes;
  std::vector<FastStaRcEdge> rc_edges;
  std::unordered_map<std::string, FastStaRcNodeId> rc_node_id_by_name;
  FastStaRcNodeId root_rc_node_id = kInvalidFastStaRcNodeId;
  FastStaPiModel pi;
  double total_cap_pf = 0.0;
  bool pre_reduced_pi_elmore = false;
  bool valid = false;
};

}  // namespace icts
