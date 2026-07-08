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
 * @file TopologyNonClusteredRealTechSmokeTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Real-tech smoke coverage for non-clustered Topology flow.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Clustering.hh"
#include "Net.hh"
#include "TopologyRealTechScenario.hh"
#include "common/logging/ScopedLogFile.hh"
#include "common/realtech/setup/RealTechDesignSetup.hh"
#include "database/config/Config.hh"
#include "flow/synthesis/TopologyArtifactWriter.hh"
#include "flow/synthesis/topology/Topology.hh"
#include "module/characterization/fixture/CharacterizationRealTechFixture.hh"
#include "utils/logger/Schema.hh"

namespace icts_test {
namespace {

namespace common_realtech = common::realtech;
namespace realtech_fixture = characterization::realtech;
namespace smoke = synthesis_realtech_smoke;

TEST(TopologyRealTechSmokeTest, NonClusteredModeSkipsClusterBuffersAndUsesUnrestrictedHTreeFrontier)
{
  const auto& setup_state = common_realtech::EnsureRealTechSetup();
  if (setup_state.mode != common_realtech::RealTechMode::kRealTech || !setup_state.setup_succeeded) {
    GTEST_SKIP() << setup_state.summary;
    return;
  }

  const auto selected_clock = smoke::SelectLargestRealClock(std::numeric_limits<std::size_t>::max(), 2U);
  if (!selected_clock.has_value()) {
    GTEST_SKIP() << "No DEF-derived clock net exposes source plus at least two sinks.";
    return;
  }
  const auto& selected_clock_data = selected_clock.value();

  realtech_fixture::RealTechCharFixture char_fixture;
  if (const auto prepare_error = char_fixture.prepare("topology_non_clustered_smoke", std::nullopt, smoke::kSynthesisSmokeMaxSlewNs,
                                                      smoke::kSynthesisSmokeMaxCapPf, true);
      prepare_error.has_value()) {
    GTEST_SKIP() << *prepare_error;
    return;
  }

  icts_test::runtime::CurrentRuntime().config.set_enable_sink_clustering(false);
  icts_test::runtime::CurrentRuntime().config.set_htree_topology_tolerance(0.1);
  ASSERT_FALSE(icts_test::runtime::CurrentRuntime().config.is_enable_sink_clustering());
  ASSERT_DOUBLE_EQ(icts_test::runtime::CurrentRuntime().config.get_htree_topology_tolerance(), 0.1);

  const auto artifact_paths = synthesis::PrepareTopologyArtifactPaths("non_clustered_mode_realtech_smoke");
  ASSERT_FALSE(artifact_paths.output_dir.empty());
  const common::logging::ScopedLogFile cts_log_guard(artifact_paths.cts_log, "Clock Synthesis Test Report");
  icts_test::runtime::CurrentRuntime().reporter.emitKeyValueTable("Clock Synthesis Smoke Scenario",
                                                                  {
                                                                      {"scenario", "non_clustered_mode"},
                                                                      {"clock_name", selected_clock_data.clock_name},
                                                                      {"clock_net", selected_clock_data.net_name},
                                                                      {"load_count", std::to_string(selected_clock_data.sinks.size())},
                                                                      {"enable_sink_clustering", "false"},
                                                                      {"omit_wirelength_unit", "true"},
                                                                  });

  icts::Topology::Config config;
  smoke::SetEnableSinkClustering(config, false);
  icts::Net root_net(selected_clock_data.net_name + "_synthesis_root_non_clustered");
  smoke::ConnectRootNet(root_net, selected_clock_data.source, selected_clock_data.sinks);
  const auto result = smoke::BuildTopology(root_net, config);

  ASSERT_TRUE(result.summary.success);
  EXPECT_FALSE(result.summary.sink_clustering_enabled);
  EXPECT_FALSE(result.output.cluster_output.has_value());

  ASSERT_EQ(result.output.htree_output.root_net, &root_net);
  EXPECT_EQ(root_net.get_driver(), selected_clock_data.source);
  EXPECT_FALSE(root_net.get_loads().empty());

  smoke::AssertTopologyHTreePayload(result);
  smoke::AssertNoSingleLoadExternalLeafBuffer(result.output.htree_output);
  smoke::AssertSelectedTopologyDepth(result);
  EXPECT_TRUE(result.output.cluster_buffers.empty());
  EXPECT_EQ(smoke::CountTopologyLeafNodes(result.output.htree_output.topology),
            smoke::CalcFloorPowerOfTwo(selected_clock_data.sinks.size()));

  smoke::WriteAndAssertSynthesisArtifacts("non_clustered_mode_realtech_smoke", "non_clustered_mode", selected_clock_data.clock_name,
                                          artifact_paths, selected_clock_data.source, selected_clock_data.sinks, result);
  smoke::AssertNonClusteredArtifacts(artifact_paths);
}

}  // namespace
}  // namespace icts_test
