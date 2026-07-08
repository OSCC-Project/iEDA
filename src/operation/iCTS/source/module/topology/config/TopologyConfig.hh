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
 * @file TopologyConfig.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-03-24
 * @brief Shared configuration types for topology module algorithms.
 */

#pragma once

#include <cstddef>
#include <limits>
#include <unordered_map>

#include "routing/ClockRouteSegmentRc.hh"

namespace icts {

class Pin;

struct BiPartitionConfig
{
  double max_ratio = 0.6;
  int max_iter = 10;
  int converge_threshold = 1000;
  std::size_t kmeans_iter_count = 5;
  double htree_topology_tolerance = 0.1;
  std::size_t max_cluster_size = 0;     // 0 means unconstrained.
  std::size_t max_leaf_load_count = 0;  // 0 means unconstrained.
};

enum class ClusterRouterKind
{
  kFlute,
  kSalt,
  kBst,
  kCbs,
};

enum class ClusterRootPolicy
{
  kMedian,
  kCenter,
};

enum class ClusterScoringStrategy
{
  kMaxDiameter,      // Sum of per-cluster max internal diameter (default)
  kTotalWirelength,  // Sum of per-cluster routed wirelength
};

struct ClusterConfig
{
  std::size_t max_fanout = 32;
  int max_diameter = std::numeric_limits<int>::max();
  double max_cap = std::numeric_limits<double>::infinity();

  ClusterRouterKind router_kind = ClusterRouterKind::kFlute;
  ClusterRootPolicy root_policy = ClusterRootPolicy::kMedian;
  ClusterScoringStrategy scoring_strategy = ClusterScoringStrategy::kMaxDiameter;

  bool enable_exact_cap = true;
  bool always_build_exact_cap = false;

  double wirelength_weight = 1.0;

  ClockRouteSegmentRc clock_route_segment_rc;
  std::unordered_map<const Pin*, double> sink_pin_cap_pf_by_pin;
};

}  // namespace icts
