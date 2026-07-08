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
 * @file TopologyArtifactWriter.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-17
 * @brief Artifact and SVG emission helpers for Topology real-tech smoke tests.
 */

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "flow/synthesis/topology/Topology.hh"

namespace icts {
class Pin;
}  // namespace icts

namespace icts_test::synthesis {

struct TopologyArtifactPaths
{
  std::filesystem::path output_dir;
  std::filesystem::path cts_log;
  std::filesystem::path synthesis_svg;
  std::filesystem::path report_log;
};

auto PrepareTopologyArtifactPaths(const std::string& case_name) -> TopologyArtifactPaths;
auto WriteTopologyArtifacts(const TopologyArtifactPaths& paths, const std::string& scenario_name, const std::string& clock_name,
                            icts::Pin* source, const std::vector<icts::Pin*>& original_sinks, const icts::Topology::Build& result) -> bool;

}  // namespace icts_test::synthesis
