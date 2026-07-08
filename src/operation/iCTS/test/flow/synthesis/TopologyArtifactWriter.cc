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
 * @file TopologyArtifactWriter.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-17
 * @brief Artifact and SVG emission helpers for Topology real-tech smoke tests.
 */

#include "flow/synthesis/TopologyArtifactWriter.hh"

#include <algorithm>
#include <cctype>
#include <compare>
#include <cstddef>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Inst.hh"
#include "Pin.hh"
#include "Point.hh"
#include "common/io/TestArtifactIO.hh"
#include "flow/synthesis/TopologySvgRenderer.hh"
#include "synthesis/topology/Topology.hh"
#include "utils/logger/Schema.hh"
#include "visualization/core/SvgCommon.hh"

namespace icts_test::synthesis {
namespace {

auto ParseDriveStrength(const std::string& cell_master) -> int
{
  const std::size_t x_pos = cell_master.find('X');
  if (x_pos == std::string::npos || x_pos + 1 >= cell_master.size()) {
    return -1;
  }

  int drive_strength = 0;
  bool saw_digit = false;
  for (std::size_t index = x_pos + 1; index < cell_master.size(); ++index) {
    const auto ch = static_cast<unsigned char>(cell_master[index]);
    if (std::isdigit(ch) == 0) {
      break;
    }

    saw_digit = true;
    drive_strength = (drive_strength * 10) + static_cast<int>(cell_master[index] - '0');
  }

  return saw_digit ? drive_strength : -1;
}

}  // namespace

auto HasValidLocation(const icts::Point<int>& location) -> bool
{
  return location.get_x() >= 0 && location.get_y() >= 0;
}

auto FindRenderableLocation(const icts::Pin* pin) -> icts::Point<int>
{
  if (pin == nullptr) {
    return {-1, -1};
  }
  const auto pin_location = pin->get_location();
  if (HasValidLocation(pin_location)) {
    return pin_location;
  }
  if (pin->get_inst() != nullptr && HasValidLocation(pin->get_inst()->get_location())) {
    return pin->get_inst()->get_location();
  }
  return {-1, -1};
}

auto CollectExtraPoints(const icts::Topology::Build& result) -> std::vector<icts::Point<int>>
{
  std::vector<icts::Point<int>> extra_points;
  extra_points.reserve(result.output.htree_output.topology.get_size() + result.output.inserted_insts.size()
                       + result.output.inserted_pins.size());

  for (std::size_t node_id = 0; node_id < result.output.htree_output.topology.get_size(); ++node_id) {
    const auto* node = result.output.htree_output.topology.get_node(node_id);
    if (node == nullptr || !HasValidLocation(node->get_position())) {
      continue;
    }
    extra_points.push_back(node->get_position());
  }

  for (const auto& inst_owner : result.output.inserted_insts) {
    const auto* inst = inst_owner.get();
    if (inst == nullptr || !HasValidLocation(inst->get_location())) {
      continue;
    }
    extra_points.push_back(inst->get_location());
  }

  for (const auto& pin_owner : result.output.inserted_pins) {
    const auto* pin = pin_owner.get();
    const auto location = FindRenderableLocation(pin);
    if (!HasValidLocation(location)) {
      continue;
    }
    extra_points.push_back(location);
  }

  return extra_points;
}

auto CollectBufferMasterSummaries(const std::vector<std::unique_ptr<icts::Inst>>& inserted_insts) -> std::vector<BufferMasterSummary>
{
  std::map<std::string, std::size_t> master_counts;
  for (const auto& inst_owner : inserted_insts) {
    const auto* inst = inst_owner.get();
    if (inst == nullptr || inst->get_cell_master().empty()) {
      continue;
    }
    ++master_counts[inst->get_cell_master()];
  }

  std::vector<BufferMasterSummary> summaries;
  summaries.reserve(master_counts.size());
  for (const auto& [cell_master, count] : master_counts) {
    summaries.push_back(BufferMasterSummary{cell_master, count, ParseDriveStrength(cell_master)});
  }

  std::ranges::sort(summaries, [](const BufferMasterSummary& lhs, const BufferMasterSummary& rhs) -> bool {
    const bool lhs_has_strength = lhs.drive_strength >= 0;
    const bool rhs_has_strength = rhs.drive_strength >= 0;
    if (lhs_has_strength != rhs_has_strength) {
      return lhs_has_strength;
    }
    if (lhs_has_strength && lhs.drive_strength != rhs.drive_strength) {
      return lhs.drive_strength < rhs.drive_strength;
    }
    return lhs.cell_master < rhs.cell_master;
  });

  return summaries;
}

auto BuildBufferRenderStyles(const std::vector<BufferMasterSummary>& summaries) -> std::unordered_map<std::string, BufferRenderStyle>
{
  std::unordered_map<std::string, BufferRenderStyle> styles;
  styles.reserve(summaries.size());
  if (summaries.empty()) {
    return styles;
  }

  constexpr double min_half_size = kTopologyVisualizationBufferHalfSize - 1.5;
  constexpr double max_half_size = kTopologyVisualizationBufferHalfSize + 2.5;
  const std::size_t rank_count = summaries.size() > 1 ? summaries.size() - 1 : 1;

  for (std::size_t index = 0; index < summaries.size(); ++index) {
    const double rank_ratio = summaries.size() == 1 ? 0.5 : static_cast<double>(index) / static_cast<double>(rank_count);
    BufferRenderStyle style;
    style.fill_color
        = icts::visualization::detail::kSvgBufferFillPalette[index % icts::visualization::detail::kSvgBufferFillPalette.size()];
    style.stroke_color
        = icts::visualization::detail::kSvgBufferStrokePalette[index % icts::visualization::detail::kSvgBufferStrokePalette.size()];
    style.half_size = min_half_size + ((max_half_size - min_half_size) * rank_ratio);
    styles.emplace(summaries[index].cell_master, style);
  }

  return styles;
}

auto CollectSinkLevelSegments(const icts::Topology::Build& result) -> std::vector<LineSegment>
{
  std::vector<LineSegment> segments;
  for (const auto& cluster_buffer : result.output.cluster_buffers) {
    if (cluster_buffer.sink_net == nullptr || cluster_buffer.output_pin == nullptr) {
      continue;
    }

    const auto source_location = FindRenderableLocation(cluster_buffer.output_pin);
    if (!HasValidLocation(source_location)) {
      continue;
    }

    for (const auto* load : cluster_buffer.sink_net->get_loads()) {
      const auto target_location = FindRenderableLocation(load);
      if (!HasValidLocation(target_location)) {
        continue;
      }
      segments.push_back(LineSegment{.start = source_location, .end = target_location});
    }
  }
  return segments;
}

auto BuildReport(const std::string& scenario_name, const std::string& clock_name, const TopologyArtifactPaths& paths, icts::Pin* source,
                 const std::vector<icts::Pin*>& original_sinks, const icts::Topology::Build& result) -> std::string
{
  const auto sink_level_segments = CollectSinkLevelSegments(result);
  std::ostringstream report;
  report << "scenario=" << scenario_name << "\n";
  report << "clock=" << clock_name << "\n";
  report << "success=" << (result.summary.success ? "true" : "false") << "\n";
  report << "sink_clustering_enabled=" << (result.summary.sink_clustering_enabled ? "true" : "false") << "\n";
  report << "input_sink_count=" << original_sinks.size() << "\n";
  report << "htree_node_count=" << result.output.htree_output.topology.get_size() << "\n";
  report << "inserted_inst_count=" << result.output.inserted_insts.size() << "\n";
  report << "inserted_net_count=" << result.output.inserted_nets.size() << "\n";
  report << "cluster_buffer_count=" << result.output.cluster_buffers.size() << "\n";
  report << "sink_level_edge_count=" << sink_level_segments.size() << "\n";
  report << "source_pin=" << (source != nullptr ? source->get_name() : "<null>") << "\n";
  report << "root_net=" << (result.output.htree_output.root_net != nullptr ? result.output.htree_output.root_net->get_name() : "<null>")
         << "\n";
  report << "artifacts=cts.log,synthesis_topology.svg,report.log\n";
  report << "output_dir=" << paths.output_dir.string() << "\n";
  return report.str();
}

auto PrepareTopologyArtifactPaths(const std::string& case_name) -> TopologyArtifactPaths
{
  TopologyArtifactPaths paths;
  paths.output_dir = common::io::PrepareCleanOutputDir(common::io::ResolveOutputDir() / "flow" / "synthesis"
                                                       / common::io::SanitizeOutputName(case_name));
  if (paths.output_dir.empty()) {
    return paths;
  }

  paths.cts_log = paths.output_dir / "cts.log";
  paths.synthesis_svg = paths.output_dir / "synthesis_topology.svg";
  paths.report_log = paths.output_dir / "report.log";
  return paths;
}

auto WriteTopologyArtifacts(const TopologyArtifactPaths& paths, const std::string& scenario_name, const std::string& clock_name,
                            icts::Pin* source, const std::vector<icts::Pin*>& original_sinks, const icts::Topology::Build& result) -> bool
{
  if (paths.output_dir.empty()) {
    return false;
  }

  const bool wrote_svg = WriteSynthesisSvg(paths.synthesis_svg, original_sinks, result);
  const bool wrote_report
      = common::io::WriteTextLog(paths.report_log, BuildReport(scenario_name, clock_name, paths, source, original_sinks, result));
  auto& reporter = icts_test::runtime::CurrentRuntime().reporter;
  if (wrote_svg) {
    icts::EmitArtifact(reporter, "Clock synthesis topology svg", paths.synthesis_svg);
  }
  if (wrote_report) {
    icts::EmitArtifact(reporter, "Clock synthesis report", paths.report_log);
  }
  return wrote_svg && wrote_report;
}

}  // namespace icts_test::synthesis
