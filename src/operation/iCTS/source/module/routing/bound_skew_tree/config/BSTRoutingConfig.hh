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
 * @file BSTRoutingConfig.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-03-16
 * @brief Bounded-skew routing topology mode, RC pattern, and electrical config.
 */

#pragma once
// IWYU pragma: private, include "bound_skew_tree/config/BSTRoutingConfig.hh"

#include <optional>

#include "Point.hh"

namespace icts {

enum class BSTRoutingRCPattern
{
  kSingle,
  kHV,
  kVH,
};

enum class BSTRoutingTopologyMode
{
  kGreedyDistance,
  kGreedyMerge,
  kBiPartition,
  kBiCluster,
  kSourceRouteTree,
};

struct BSTRoutingConfig
{
  double skew_bound = 0.0;
  int dbu_per_um = 0;
  double unit_h_cap = 0.0;
  double unit_h_res = 0.0;
  double unit_v_cap = 0.0;
  double unit_v_res = 0.0;
  BSTRoutingTopologyMode topology_mode = BSTRoutingTopologyMode::kGreedyDistance;
  BSTRoutingRCPattern rc_pattern = BSTRoutingRCPattern::kHV;
  std::optional<Point<int>> root_guide = std::nullopt;
};

}  // namespace icts
