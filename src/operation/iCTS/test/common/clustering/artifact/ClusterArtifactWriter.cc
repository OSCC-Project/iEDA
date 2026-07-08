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
 * @file ClusterArtifactWriter.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Shared cluster artifact construction helpers for clustering tests.
 */

#include "common/clustering/artifact/ClusterArtifactWriter.hh"

#include <cstddef>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/clustering/metrics/ClusterGeometryMetrics.hh"
#include "database/spatial/Point.hh"
#include "module/topology/clustering/Clustering.hh"

namespace icts_test::common::clustering {

auto BuildClusterArtifacts(const icts::ClusterOutput& result, const std::vector<icts::Pin*>& loads,
                           std::unordered_map<const icts::Pin*, std::size_t>& cluster_map, std::vector<icts::Point<int>>& centers,
                           std::vector<std::size_t>& cluster_sizes, std::string& error) -> bool
{
  cluster_map.clear();
  centers.clear();
  cluster_sizes.clear();

  if (result.clusters.empty()) {
    error = "result has no clusters";
    return false;
  }

  cluster_map.reserve(loads.size());
  for (const auto& raw_cluster : result.clusters) {
    if (raw_cluster.empty()) {
      continue;
    }

    const std::size_t compact_cluster_id = centers.size();
    for (const auto* pin : raw_cluster) {
      if (pin == nullptr) {
        error = "null pin in cluster result";
        return false;
      }
      if (cluster_map.contains(pin)) {
        error = "duplicate pin assignment in cluster result";
        return false;
      }
      cluster_map.emplace(pin, compact_cluster_id);
    }
    cluster_sizes.push_back(raw_cluster.size());

    icts::Point<int> center;
    if (compact_cluster_id < result.centers.size()) {
      center = result.centers.at(compact_cluster_id);
    } else {
      center = CalcClusterCenter(raw_cluster);
    }
    if (center.get_x() < 0 || center.get_y() < 0) {
      center = CalcClusterCenter(raw_cluster);
    }
    centers.push_back(center);
  }

  if (cluster_map.size() != loads.size()) {
    std::ostringstream stream;
    stream << "cluster map size mismatch: expected " << loads.size() << ", got " << cluster_map.size();
    error = stream.str();
    return false;
  }

  return true;
}

}  // namespace icts_test::common::clustering
