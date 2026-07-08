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
 * @file TestVisualization.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Shared SVG visualization helpers for iCTS tests.
 */

#include "common/visualization/TestVisualization.hh"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "visualization/cluster/ClusterSvgWriter.hh"
#include "visualization/topology/TopologySvgWriter.hh"

namespace icts_test::common::visualization {

auto WriteClusterSvg(const std::string& path, const std::vector<icts::Pin*>& loads,
                     const std::unordered_map<const icts::Pin*, std::size_t>& cluster_map, const std::vector<icts::Point<int>>& centers)
    -> bool
{
  return icts::visualization::cluster::WriteClusterSvg(path, loads, cluster_map, centers);
}

auto WriteClusterComparisonSvg(const std::string& path, const std::vector<icts::Pin*>& loads, const std::string& left_title,
                               const std::unordered_map<const icts::Pin*, std::size_t>& left_cluster_map,
                               const std::vector<icts::Point<int>>& left_centers, const std::string& right_title,
                               const std::unordered_map<const icts::Pin*, std::size_t>& right_cluster_map,
                               const std::vector<icts::Point<int>>& right_centers) -> bool
{
  return icts::visualization::cluster::WriteClusterComparisonSvg(path, loads, left_title, left_cluster_map, left_centers, right_title,
                                                                 right_cluster_map, right_centers);
}

auto WriteTopologySvg(const std::string& path, const icts::Tree& tree, const std::vector<icts::Pin*>& loads) -> bool
{
  return icts::visualization::topology::WriteTopologySvg(path, tree, loads);
}

}  // namespace icts_test::common::visualization
