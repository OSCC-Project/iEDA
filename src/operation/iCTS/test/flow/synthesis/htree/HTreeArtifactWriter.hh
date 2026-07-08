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
 * @file HTreeArtifactWriter.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-14
 * @brief SVG artifact emission helpers for HTree tests.
 */

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "flow/synthesis/htree/HTree.hh"
#include "flow/synthesis/htree/diagnostic/HTreeDiagnostic.hh"

namespace icts {
class Pin;
}  // namespace icts

namespace icts_test::htree {

struct HTreeArtifactPaths
{
  std::filesystem::path output_dir;
  std::filesystem::path cts_log;
  std::filesystem::path topology_svg;
  std::filesystem::path materialized_svg;
  std::filesystem::path pareto_svg;
  std::filesystem::path report_log;
};

auto PrepareHTreeArtifactPaths(const std::string& case_name) -> HTreeArtifactPaths;
auto WriteHTreeArtifacts(const HTreeArtifactPaths& paths, const std::string& scenario_name, const std::string& input_summary,
                         const std::vector<icts::Pin*>& loads, const icts::htree::DiagnosticBuild& result) -> bool;

}  // namespace icts_test::htree
