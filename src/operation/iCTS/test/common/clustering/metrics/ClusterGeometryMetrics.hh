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
 * @file ClusterGeometryMetrics.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Shared cluster geometry and metric helpers for clustering tests.
 */

#pragma once

#include <cstddef>
#include <vector>

namespace icts {
class Pin;
template <typename T>
class Point;
struct ClusterOutput;
}  // namespace icts

namespace icts_test::common::clustering {

struct ClusterMetrics
{
  std::size_t cluster_count = 0;
  std::size_t singleton_cluster_count = 0;
  std::size_t min_cluster_size = 0;
  std::size_t max_cluster_size = 0;
  double avg_cluster_size = 0.0;
  int min_cluster_diameter = 0;
  int max_cluster_diameter = 0;
};

auto CalcClusterDiameter(const std::vector<icts::Pin*>& cluster) -> int;
auto CalcClusterCenter(const std::vector<icts::Pin*>& cluster) -> icts::Point<int>;
auto CollectClusterMetrics(const icts::ClusterOutput& result) -> ClusterMetrics;

}  // namespace icts_test::common::clustering
