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
 * @file TopologyGenCaseFixture.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Case fixture declarations for topology generation tests.
 */

#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "Pin.hh"
#include "Point.hh"
#include "common/dataset/TestDataset.hh"
#include "database/spatial/Tree.hh"
#include "module/topology/topology_gen/fixture/TopologyGenScenario.hh"

namespace icts_test::topology_gen::detail {

struct TopologyArtifacts
{
  TopologyStats stats;
  std::unordered_map<const icts::Pin*, std::size_t> cluster_map;
  std::vector<icts::Point<int>> centers;
};

auto GenerateCase(const TopologyCase& test_case) -> GeneratedPins;
auto ValidateOutputDir(const std::filesystem::path& output_dir) -> void;
auto AnalyzeBuiltTopology(const icts::Tree& tree, const std::vector<icts::Pin*>& loads, TopologyArtifacts& artifacts) -> void;
auto LogTopologySummary(const TopologyStats& stats) -> void;
auto ValidateTreeEdges(const icts::Tree& tree) -> void;
auto WriteArtifacts(const std::filesystem::path& output_dir, const TopologyCase& test_case, const icts::Tree& tree,
                    const std::vector<icts::Pin*>& loads) -> void;

}  // namespace icts_test::topology_gen::detail
