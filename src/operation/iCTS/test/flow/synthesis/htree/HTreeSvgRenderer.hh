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
 * @file HTreeSvgRenderer.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Private SVG and report rendering helpers for H-tree tests.
 */

#pragma once

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "HTreeTopologyChar.hh"
#include "Inst.hh"
#include "flow/synthesis/htree/HTreeArtifactWriter.hh"
#include "flow/synthesis/htree/diagnostic/HTreeDiagnostic.hh"
#include "visualization/core/SvgCommon.hh"

namespace icts {
class Pin;
}  // namespace icts

namespace icts_test::htree {

struct BufferMasterSummary
{
  std::string cell_master;
  std::size_t count = 0;
  int drive_strength = -1;
};

struct BufferRenderStyle
{
  std::string fill_color = icts::visualization::detail::kSvgColorBufferFillDefault;
  std::string stroke_color = icts::visualization::detail::kSvgColorBufferStrokeDefault;
  double half_size = 6.0;
};

struct DelayPowerPoint
{
  const icts::HTreeTopologyChar* entry = nullptr;
  double normalized_delay = 0.0;
  double normalized_power = 0.0;
  bool is_pareto = false;
  bool is_selected = false;
};

struct DelayPowerChoiceSummary
{
  std::size_t frontier_selection_pool_size = 0U;
  std::size_t selection_pool_size = 0U;
  std::size_t pareto_solution_count = 0U;
  std::optional<std::size_t> pareto_power_median_index = std::nullopt;
  std::optional<std::size_t> selected_pareto_power_order_index = std::nullopt;
  bool median_uses_lower_index = false;
  bool selected_on_pareto_front = false;
};

auto FormatSvgNumber(double value) -> std::string;
auto EscapeXml(const std::string& text) -> std::string;
auto WriteTooltip(std::ofstream& output_stream, const std::string& text) -> void;
auto CollectBufferMasterSummaries(const std::vector<std::unique_ptr<icts::Inst>>& inserted_insts) -> std::vector<BufferMasterSummary>;
auto BuildDelayPowerPoints(const icts::htree::DiagnosticBuild& result) -> std::vector<DelayPowerPoint>;
auto BuildDelayPowerChoiceSummary(const icts::htree::DiagnosticBuild& result) -> std::optional<DelayPowerChoiceSummary>;

auto WriteMaterializedSvg(const std::filesystem::path& path, const std::vector<icts::Pin*>& loads,
                          const icts::htree::DiagnosticBuild& result) -> bool;
auto WriteDelayPowerParetoSvg(const std::filesystem::path& path, const icts::htree::DiagnosticBuild& result) -> bool;
auto BuildReport(const std::string& scenario_name, const std::string& input_summary, const HTreeArtifactPaths& paths,
                 const std::vector<icts::Pin*>& loads, const icts::htree::DiagnosticBuild& result) -> std::string;

}  // namespace icts_test::htree
