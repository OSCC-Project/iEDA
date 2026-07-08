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
 * @file TopologyGenArtifactWriter.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Artifact emission for topology generation tests.
 */

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <ostream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#include "Log.hh"
#include "Point.hh"
#include "common/topology/TopologyAnalysis.hh"
#include "common/visualization/TestVisualization.hh"
#include "module/topology/topology_gen/fixture/TopologyGenCaseFixture.hh"
#include "module/topology/topology_gen/fixture/TopologyGenScenario.hh"

namespace icts {
class Pin;
class Tree;
}  // namespace icts

namespace icts_test::topology_gen::detail {

auto ValidateOutputDir(const std::filesystem::path& output_dir) -> void
{
  std::error_code error_code;
  std::filesystem::create_directories(output_dir, error_code);
  ASSERT_FALSE(error_code) << "Failed to create output dir: " << output_dir.string() << " (" << error_code.message() << ")";
}

auto WriteArtifacts(const std::filesystem::path& output_dir, const TopologyCase& test_case, const icts::Tree& tree,
                    const std::vector<icts::Pin*>& loads) -> void
{
  std::unordered_map<const icts::Pin*, std::size_t> first_level_cluster_map;
  std::vector<icts::Point<int>> first_level_centers;
  std::string first_level_error;
  ASSERT_TRUE(common::topology::AnalyzeFirstLevelClusters(tree, loads, first_level_cluster_map, first_level_centers, first_level_error))
      << first_level_error;

  const std::string cluster_name = "cluster_" + test_case.name + "_" + std::to_string(test_case.count) + ".svg";
  const auto cluster_path = output_dir / cluster_name;
  EXPECT_TRUE(common::visualization::WriteClusterSvg(cluster_path.string(), loads, first_level_cluster_map, first_level_centers))
      << "Failed to write cluster svg: " << cluster_path.string();
  LOG_INFO << "Cluster svg saved (first-level only): " << cluster_path.string();

  const std::string topo_name = "topology_" + test_case.name + "_" + std::to_string(test_case.count) + ".svg";
  const auto topo_path = output_dir / topo_name;
  EXPECT_TRUE(common::visualization::WriteTopologySvg(topo_path.string(), tree, loads))
      << "Failed to write topology svg: " << topo_path.string();
  LOG_INFO << "Topology svg saved: " << topo_path.string();
}

}  // namespace icts_test::topology_gen::detail
