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
 * @file TopologyRealTechArtifactAssertions.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Real-tech Topology artifact assertions.
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "TopologyRealTechScenario.hh"
#include "flow/synthesis/TopologyArtifactWriter.hh"
#include "synthesis/topology/Topology.hh"

namespace icts {
class Pin;
}  // namespace icts

namespace icts_test::synthesis_realtech_smoke {
namespace {

auto ReadTextFile(const std::filesystem::path& path) -> std::string
{
  std::ifstream input_stream(path);
  std::ostringstream buffer;
  buffer << input_stream.rdbuf();
  return buffer.str();
}

auto RemovedClusterDistanceArtifactName() -> std::string
{
  return std::string{"cluster_leaf_distance"} + ".csv";
}

auto AssertTopologyPhysicalMetricUnits(const std::string& cts_log_content, bool sink_clustering_enabled) -> void
{
  EXPECT_NE(cts_log_content.find("TopologyGen Load Distribution Overview"), std::string::npos);
  EXPECT_NE(cts_log_content.find("TopologyGen Root-To-Leaf Path Overview"), std::string::npos);
  if (sink_clustering_enabled) {
    EXPECT_NE(cts_log_content.find("load_count (local buffer)"), std::string::npos);
    EXPECT_EQ(cts_log_content.find("load_count (sink)"), std::string::npos);
  } else {
    EXPECT_NE(cts_log_content.find("load_count (sink)"), std::string::npos);
    EXPECT_EQ(cts_log_content.find("load_count (local buffer)"), std::string::npos);
  }
  EXPECT_TRUE(std::regex_search(cts_log_content, std::regex(R"(\|\s*span_width_height\s*\|\s*[^|\n]*x[^|\n]*um\s*\|)")));
  EXPECT_TRUE(std::regex_search(cts_log_content, std::regex(R"(\|\s*area\s*\|\s*[^|\n]*um\^2\s*\|)")));
  EXPECT_TRUE(std::regex_search(cts_log_content, std::regex(R"(\|\s*sqrt_area\s*\|\s*[^|\n]*um\s*\|)")));
  EXPECT_TRUE(std::regex_search(cts_log_content, std::regex(R"(\|\s*half_perimeter\s*\|\s*[^|\n]*um\s*\|)")));
  EXPECT_TRUE(std::regex_search(cts_log_content, std::regex(R"(\|\s*min_path_length\s*\|\s*[^|\n]*um\s*\|)")));
  EXPECT_TRUE(std::regex_search(cts_log_content, std::regex(R"(\|\s*max_path_length\s*\|\s*[^|\n]*um\s*\|)")));
  EXPECT_TRUE(std::regex_search(cts_log_content, std::regex(R"(\|\s*avg_path_length\s*\|\s*[^|\n]*um\s*\|)")));
  EXPECT_FALSE(std::regex_search(cts_log_content, std::regex(R"(\|\s*span_width_height\s*\|[^\n]*DBU)")));
  EXPECT_FALSE(std::regex_search(cts_log_content, std::regex(R"(\|\s*area\s*\|[^\n]*DBU)")));
  EXPECT_FALSE(std::regex_search(cts_log_content, std::regex(R"(\|\s*sqrt_area\s*\|[^\n]*DBU)")));
  EXPECT_FALSE(std::regex_search(cts_log_content, std::regex(R"(\|\s*half_perimeter\s*\|[^\n]*DBU)")));
  EXPECT_FALSE(std::regex_search(cts_log_content, std::regex(R"(\|\s*min_path_length\s*\|[^\n]*DBU)")));
  EXPECT_FALSE(std::regex_search(cts_log_content, std::regex(R"(\|\s*max_path_length\s*\|[^\n]*DBU)")));
  EXPECT_FALSE(std::regex_search(cts_log_content, std::regex(R"(\|\s*avg_path_length\s*\|[^\n]*DBU)")));
}

}  // namespace

auto WriteAndAssertSynthesisArtifacts(const std::string& case_name, const std::string& scenario_name, const std::string& clock_name,
                                      const synthesis::TopologyArtifactPaths& artifact_paths, icts::Pin* source,
                                      const std::vector<icts::Pin*>& sinks, const icts::Topology::Build& result)
    -> synthesis::TopologyArtifactPaths
{
  if (artifact_paths.output_dir.empty()) {
    ADD_FAILURE() << "Failed to prepare synthesis artifact output dir for case " << case_name;
    return artifact_paths;
  }
  EXPECT_TRUE(synthesis::WriteTopologyArtifacts(artifact_paths, scenario_name, clock_name, source, sinks, result));
  EXPECT_TRUE(std::filesystem::exists(artifact_paths.cts_log));
  EXPECT_TRUE(std::filesystem::exists(artifact_paths.synthesis_svg));
  EXPECT_TRUE(std::filesystem::exists(artifact_paths.report_log));
  return artifact_paths;
}

auto AssertClusteredArtifacts(const synthesis::TopologyArtifactPaths& artifact_paths) -> void
{
  const auto cts_log_content = ReadTextFile(artifact_paths.cts_log);
  ASSERT_FALSE(cts_log_content.empty());
  AssertTopologyPhysicalMetricUnits(cts_log_content, true);
  EXPECT_NE(cts_log_content.find("Cluster Center vs H-Tree Leaf Distance Overview"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("Cluster Center vs H-Tree Leaf Distance Details"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("Cluster Center vs H-Tree Leaf Distance Top Worst Clusters"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("Notes"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("Cluster center to H-tree leaf distance is summarized here"), std::string::npos);
  EXPECT_NE(cts_log_content.find("mean_distance"), std::string::npos);
  EXPECT_NE(cts_log_content.find("median_distance"), std::string::npos);
  EXPECT_TRUE(std::regex_search(cts_log_content, std::regex(R"(\|\s*min_distance\s*\|\s*[^|\n]*um\s*\|)")));
  EXPECT_TRUE(std::regex_search(cts_log_content, std::regex(R"(\|\s*max_distance\s*\|\s*[^|\n]*um\s*\|)")));
  EXPECT_TRUE(std::regex_search(cts_log_content, std::regex(R"(\|\s*mean_distance\s*\|\s*[^|\n]*um\s*\|)")));
  EXPECT_TRUE(std::regex_search(cts_log_content, std::regex(R"(\|\s*median_distance\s*\|\s*[^|\n]*um\s*\|)")));
  EXPECT_EQ(cts_log_content.find("min_distance_dbu"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("max_distance_dbu"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("mean_distance_dbu"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("median_distance_dbu"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("p90_distance_dbu"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("p95_distance_dbu"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("p99_distance_dbu"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("count_over_50000_dbu"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("count_over_80000_dbu"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("count_over_100000_dbu"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("detail_artifact"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("detail_row_count"), std::string::npos);
  EXPECT_EQ(cts_log_content.find(RemovedClusterDistanceArtifactName()), std::string::npos);
  EXPECT_NE(cts_log_content.find("htree_load_group_count"), std::string::npos);
  EXPECT_NE(cts_log_content.find("htree_load_cap_min"), std::string::npos);
  EXPECT_NE(cts_log_content.find("htree_load_cap_max"), std::string::npos);
  EXPECT_NE(cts_log_content.find("htree_load_cap_mean"), std::string::npos);
  EXPECT_NE(cts_log_content.find("htree_load_cap_median"), std::string::npos);

  EXPECT_FALSE(std::filesystem::exists(artifact_paths.output_dir / RemovedClusterDistanceArtifactName()));

  const auto svg_content = ReadTextFile(artifact_paths.synthesis_svg);
  ASSERT_FALSE(svg_content.empty());
  EXPECT_EQ(svg_content.find("cts_clock_source_to_htree_root"), std::string::npos);
  EXPECT_NE(svg_content.find("sink-level net"), std::string::npos);
  EXPECT_TRUE(std::regex_search(svg_content, std::regex(R"(<line [^>]*stroke="#2ca25f"[^>]*><title>net cts_htree_net_)")));
  EXPECT_TRUE(std::regex_search(svg_content, std::regex(R"(<line [^>]*stroke="#0f766e"[^>]*><title>sink-level net )")));
}

auto AssertNonClusteredArtifacts(const synthesis::TopologyArtifactPaths& artifact_paths) -> void
{
  const auto cts_log_content = ReadTextFile(artifact_paths.cts_log);
  ASSERT_FALSE(cts_log_content.empty());
  AssertTopologyPhysicalMetricUnits(cts_log_content, false);
  EXPECT_EQ(cts_log_content.find("Cluster Center vs H-Tree Leaf Distance Overview"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("Cluster Center vs H-Tree Leaf Distance Details"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("Cluster Center vs H-Tree Leaf Distance Top Worst Clusters"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("Notes"), std::string::npos);
  EXPECT_EQ(cts_log_content.find(RemovedClusterDistanceArtifactName()), std::string::npos);
  EXPECT_FALSE(std::filesystem::exists(artifact_paths.output_dir / RemovedClusterDistanceArtifactName()));
  EXPECT_NE(cts_log_content.find("htree_load_group_count"), std::string::npos);
  EXPECT_NE(cts_log_content.find("htree_load_cap_min"), std::string::npos);
  EXPECT_NE(cts_log_content.find("htree_load_cap_max"), std::string::npos);
  EXPECT_NE(cts_log_content.find("htree_load_cap_mean"), std::string::npos);
  EXPECT_NE(cts_log_content.find("htree_load_cap_median"), std::string::npos);

  const auto svg_content = ReadTextFile(artifact_paths.synthesis_svg);
  ASSERT_FALSE(svg_content.empty());
  EXPECT_EQ(svg_content.find("cts_clock_source_to_htree_root"), std::string::npos);
  EXPECT_EQ(svg_content.find("sink-level net"), std::string::npos);
  EXPECT_TRUE(std::regex_search(svg_content, std::regex(R"(<line [^>]*stroke="#2ca25f"[^>]*><title>net cts_htree_net_)")));
}

}  // namespace icts_test::synthesis_realtech_smoke
