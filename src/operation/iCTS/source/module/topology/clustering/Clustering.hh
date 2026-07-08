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
 * @file Clustering.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-16
 * @brief Clustering for topology embedding.
 */

#pragma once

#include <cstddef>
#include <vector>

#include "Point.hh"

namespace icts {

class Pin;
struct BiPartitionConfig;
struct ClusterConfig;

struct ClusterElectricalSummary
{
  bool exact = false;
  bool route_success = false;
  std::size_t sink_count = 0;
  int diameter_dbu = 0;
  double pin_cap_pf = 0.0;
  double wire_cap_pf = 0.0;
  double total_cap_pf = 0.0;
  double wirelength_dbu = 0.0;
};

enum class ClusterElectricalViolation
{
  kNone,
  kEmptyCluster,
  kFanout,
  kDiameter,
  kCapacitance,
  kRoutingFailed,
};

struct ClusterElectricalEvaluation
{
  bool legal = false;
  ClusterElectricalViolation violation = ClusterElectricalViolation::kEmptyCluster;
  ClusterElectricalSummary summary;
};

struct ClusterOutput
{
  std::vector<std::vector<Pin*>> clusters;
  std::vector<Point<int>> centers;
  std::vector<ClusterElectricalSummary> electrical_summaries;
};

class Clustering
{
 public:
  Clustering() = delete;
  ~Clustering() = default;

  static auto biPartition(const std::vector<Pin*>& loads, std::size_t min_cluster_size) -> ClusterOutput;
  static auto biPartition(const std::vector<Pin*>& loads, std::size_t min_cluster_size, const BiPartitionConfig& config) -> ClusterOutput;
  static auto fastClustering(const std::vector<Pin*>& loads) -> ClusterOutput;
  static auto defaultFastClustering(const std::vector<Pin*>& loads, const ClusterConfig& base_config) -> ClusterOutput;
  static auto fastClustering(const std::vector<Pin*>& loads, const ClusterConfig& config) -> ClusterOutput;
  static auto evaluateClusterElectrical(const std::vector<Pin*>& loads, const Point<int>& anchor, const ClusterConfig& config)
      -> ClusterElectricalEvaluation;
  static auto evaluateClusterElectrical(const std::vector<Pin*>& loads, const Point<int>& anchor, const ClusterConfig& config,
                                        bool need_exact_cap) -> ClusterElectricalEvaluation;
};

}  // namespace icts
