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
 * @file HTreeVisualizationDelayPower.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Delay-power Pareto SVG and report rendering helpers for H-tree artifacts.
 */

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "HTreeTopologyChar.hh"
#include "PatternId.hh"
#include "Pin.hh"
#include "Tree.hh"
#include "flow/synthesis/htree/HTreeArtifactWriter.hh"
#include "flow/synthesis/htree/HTreeBuildObservation.hh"
#include "flow/synthesis/htree/HTreeSvgRenderer.hh"
#include "synthesis/htree/HTree.hh"
#include "visualization/core/SvgCommon.hh"

namespace icts_test::htree {
namespace {

auto NormalizeMetric(double value, double min_value, double max_value) -> double
{
  if (max_value <= min_value) {
    return 0.0;
  }
  return std::clamp((value - min_value) / (max_value - min_value), 0.0, 1.0);
}

}  // namespace

auto BuildDelayPowerPoints(const icts::htree::DiagnosticBuild& result) -> std::vector<DelayPowerPoint>
{
  if (!result.output.best_char.has_value()) {
    return {};
  }

  std::vector<DelayPowerPoint> points;
  points.push_back(DelayPowerPoint{
      .entry = &(*result.output.best_char),
      .normalized_delay
      = NormalizeMetric(result.output.best_char->get_delay(), result.output.best_char->get_delay(), result.output.best_char->get_delay()),
      .normalized_power
      = NormalizeMetric(result.output.best_char->get_power(), result.output.best_char->get_power(), result.output.best_char->get_power()),
      .is_pareto = true,
      .is_selected = true,
  });

  std::ranges::sort(points, [](const DelayPowerPoint& lhs, const DelayPowerPoint& rhs) -> bool {
    if (lhs.normalized_delay != rhs.normalized_delay) {
      return lhs.normalized_delay < rhs.normalized_delay;
    }
    if (lhs.normalized_power != rhs.normalized_power) {
      return lhs.normalized_power < rhs.normalized_power;
    }
    return lhs.entry->get_pattern_id().pack() < rhs.entry->get_pattern_id().pack();
  });
  return points;
}

auto BuildDelayPowerChoiceSummary(const icts::htree::DiagnosticBuild& result) -> std::optional<DelayPowerChoiceSummary>
{
  if (!result.output.best_char.has_value()) {
    return std::nullopt;
  }

  const auto observation = ObserveHTreeBuild(result);
  const std::size_t frontier_pool_size
      = observation.used_boundary_relaxation ? observation.selected_candidate_solution_count : observation.selected_feasible_solution_count;
  const std::size_t selection_pool_size = observation.used_boundary_relaxation ? observation.selected_candidate_frontier_entry_count
                                                                               : observation.selected_feasible_frontier_entry_count;

  DelayPowerChoiceSummary summary{
      .frontier_selection_pool_size = frontier_pool_size,
      .selection_pool_size = selection_pool_size,
      .pareto_solution_count = result.output.best_char.has_value() ? 1U : 0U,
      .median_uses_lower_index = false,
  };
  summary.pareto_power_median_index = 0U;
  summary.selected_on_pareto_front = true;
  summary.selected_pareto_power_order_index = 0U;

  return summary;
}

namespace {

auto BuildDelayPowerTooltip(const DelayPowerPoint& point) -> std::string
{
  std::ostringstream tooltip;
  tooltip.setf(std::ostringstream::fixed, std::ostringstream::floatfield);
  tooltip << std::setprecision(6);
  tooltip << "pattern_id=" << point.entry->get_pattern_id().local_id;
  tooltip << ", delay=" << point.entry->get_delay();
  tooltip << ", power=" << point.entry->get_power();
  tooltip << ", norm_delay=" << point.normalized_delay;
  tooltip << ", norm_power=" << point.normalized_power;
  tooltip << ", in_slew_idx=" << point.entry->get_input_slew_idx();
  tooltip << ", out_slew_idx=" << point.entry->get_output_slew_idx();
  tooltip << ", driven_cap_idx=" << point.entry->get_driven_cap_idx();
  tooltip << ", load_cap_idx=" << point.entry->get_load_cap_idx();
  if (point.is_pareto) {
    tooltip << ", pareto=true";
  }
  if (point.is_selected) {
    tooltip << ", selected=true";
  }
  return tooltip.str();
}

}  // namespace

auto WriteDelayPowerParetoSvg(const std::filesystem::path& path, const icts::htree::DiagnosticBuild& result) -> bool
{
  const auto points = BuildDelayPowerPoints(result);
  if (points.empty()) {
    std::ofstream empty_stream(path);
    if (!empty_stream.is_open()) {
      return false;
    }
    empty_stream << R"(<svg xmlns="http://www.w3.org/2000/svg" width="720" height="520" viewBox="0 0 720 520"></svg>)";
    return true;
  }

  constexpr double width = 720.0;
  constexpr double height = 520.0;
  constexpr double plot_left = 88.0;
  constexpr double plot_right = 660.0;
  constexpr double plot_top = 54.0;
  constexpr double plot_bottom = 434.0;
  const double plot_width = plot_right - plot_left;
  const double plot_height = plot_bottom - plot_top;

  auto map_x = [plot_width](double normalized_delay) -> double { return plot_left + (normalized_delay * plot_width); };
  auto map_y = [plot_height](double normalized_power) -> double { return plot_bottom - (normalized_power * plot_height); };

  std::ofstream output_stream(path);
  if (!output_stream.is_open()) {
    return false;
  }

  output_stream << icts::visualization::detail::kSvgOpenTagPrefix << FormatSvgNumber(width) << icts::visualization::detail::kSvgHeightTag
                << FormatSvgNumber(height) << icts::visualization::detail::kSvgViewBoxPrefix << FormatSvgNumber(width) << ' '
                << FormatSvgNumber(height) << icts::visualization::detail::kSvgOpenTagSuffix;
  output_stream << icts::visualization::detail::kSvgBackgroundRect;

  output_stream << R"(<g font-family="monospace" fill="#1f2933">)";
  output_stream << R"(<text x="36" y="28" font-size="16" font-weight="700">Sink-Load-Region-Legal Delay-Power Frontier</text>)";
  output_stream
      << R"(<text x="36" y="46" font-size="12">Each point is one sink-load-region-legal frontier entry. Lower-left is better.</text>)";
  output_stream << R"(<rect x=")" << FormatSvgNumber(plot_left) << R"(" y=")" << FormatSvgNumber(plot_top) << R"(" width=")"
                << FormatSvgNumber(plot_width) << R"(" height=")" << FormatSvgNumber(plot_height)
                << R"(" fill="#ffffff" fill-opacity="0.85" stroke="#d0d7de" />)";

  for (int tick = 0; tick <= 5; ++tick) {
    const double normalized_value = static_cast<double>(tick) / 5.0;
    const double grid_x = map_x(normalized_value);
    const double grid_y = map_y(normalized_value);

    output_stream << R"(<line x1=")" << FormatSvgNumber(grid_x) << R"(" y1=")" << FormatSvgNumber(plot_top) << R"(" x2=")"
                  << FormatSvgNumber(grid_x) << R"(" y2=")" << FormatSvgNumber(plot_bottom) << R"(" stroke="#edf2f7" stroke-width="1" />)";
    output_stream << R"(<line x1=")" << FormatSvgNumber(plot_left) << R"(" y1=")" << FormatSvgNumber(grid_y) << R"(" x2=")"
                  << FormatSvgNumber(plot_right) << R"(" y2=")" << FormatSvgNumber(grid_y) << R"(" stroke="#edf2f7" stroke-width="1" />)";

    output_stream << R"(<text x=")" << FormatSvgNumber(grid_x - 8.0) << R"(" y="456" font-size="11">)" << FormatSvgNumber(normalized_value)
                  << R"(</text>)";
    output_stream << R"(<text x="52" y=")" << FormatSvgNumber(grid_y + 4.0) << R"(" font-size="11">)" << FormatSvgNumber(normalized_value)
                  << R"(</text>)";
  }

  output_stream << R"(<line x1=")" << FormatSvgNumber(plot_left) << R"(" y1=")" << FormatSvgNumber(plot_bottom) << R"(" x2=")"
                << FormatSvgNumber(plot_right) << R"(" y2=")" << FormatSvgNumber(plot_bottom)
                << R"(" stroke="#4a5568" stroke-width="1.4" />)";
  output_stream << R"(<line x1=")" << FormatSvgNumber(plot_left) << R"(" y1=")" << FormatSvgNumber(plot_bottom) << R"(" x2=")"
                << FormatSvgNumber(plot_left) << R"(" y2=")" << FormatSvgNumber(plot_top) << R"(" stroke="#4a5568" stroke-width="1.4" />)";
  output_stream << R"(<text x=")" << FormatSvgNumber((plot_left + plot_right) * 0.5 - 58.0)
                << R"(" y="488" font-size="13">normalized delay</text>)";
  const double power_label_y = (plot_top + plot_bottom) * 0.5;
  output_stream << R"SVG(<text x="22" y=")SVG" << FormatSvgNumber(power_label_y) << R"SVG(" font-size="13" transform="rotate(-90 22 )SVG"
                << FormatSvgNumber(power_label_y) << R"SVG()">normalized power</text>)SVG";

  std::vector<const DelayPowerPoint*> pareto_points;
  pareto_points.reserve(points.size());
  for (const auto& point : points) {
    if (point.is_pareto) {
      pareto_points.push_back(&point);
    }
  }

  if (!pareto_points.empty()) {
    std::ranges::sort(pareto_points, [](const DelayPowerPoint* lhs, const DelayPowerPoint* rhs) -> bool {
      if (lhs->normalized_delay != rhs->normalized_delay) {
        return lhs->normalized_delay < rhs->normalized_delay;
      }
      return lhs->normalized_power < rhs->normalized_power;
    });

    output_stream << R"(<polyline fill="none" stroke="#e76f51" stroke-width="2.2" points=")";
    for (const auto* point : pareto_points) {
      output_stream << FormatSvgNumber(map_x(point->normalized_delay)) << "," << FormatSvgNumber(map_y(point->normalized_power)) << " ";
    }
    output_stream << R"("/>)";
  }

  for (const auto& point : points) {
    const double center_x = map_x(point.normalized_delay);
    const double center_y = map_y(point.normalized_power);
    std::string fill_color = "#94a3b8";
    std::string stroke_color = "#64748b";
    double radius = 3.8;
    if (point.is_selected) {
      fill_color = "#6a3d9a";
      stroke_color = "#2d1b69";
      radius = 6.5;
    } else if (point.is_pareto) {
      fill_color = "#e76f51";
      stroke_color = "#91361f";
      radius = 5.0;
    }

    output_stream << R"(<circle cx=")" << FormatSvgNumber(center_x) << R"(" cy=")" << FormatSvgNumber(center_y) << R"(" r=")"
                  << FormatSvgNumber(radius) << R"(" fill=")" << fill_color << R"(" stroke=")" << stroke_color
                  << R"(" stroke-width="1.2">)";
    WriteTooltip(output_stream, BuildDelayPowerTooltip(point));
    output_stream << "</circle>\n";
  }

  output_stream << R"(<g font-size="12">)";
  output_stream << R"(<rect x="478" y="60" width="200" height="92" rx="6" fill="#ffffff" fill-opacity="0.9" stroke="#d0d7de" />)";
  output_stream << R"(<circle cx="494" cy="84" r="4.5" fill="#94a3b8" stroke="#64748b" stroke-width="1.2" />)";
  output_stream << R"(<text x="508" y="88">frontier entry</text>)";
  output_stream << R"(<circle cx="494" cy="108" r="5.0" fill="#e76f51" stroke="#91361f" stroke-width="1.2" />)";
  output_stream << R"(<text x="508" y="112">Pareto front</text>)";
  output_stream << R"(<circle cx="494" cy="132" r="6.5" fill="#6a3d9a" stroke="#2d1b69" stroke-width="1.2" />)";
  output_stream << R"(<text x="508" y="136">selected solution</text>)";
  output_stream << R"(</g>)";
  output_stream << R"(</g>)";

  output_stream << icts::visualization::detail::kSvgClosingTag;
  return true;
}

auto BuildReport(const std::string& scenario_name, const std::string& input_summary, const HTreeArtifactPaths& paths,
                 const std::vector<icts::Pin*>& loads, const icts::htree::DiagnosticBuild& result) -> std::string
{
  const auto delay_power_points = BuildDelayPowerPoints(result);
  const auto observation = ObserveHTreeBuild(result);

  std::ostringstream report;
  report.setf(std::ostringstream::fixed, std::ostringstream::floatfield);
  report << std::setprecision(3);
  report << "scenario=" << scenario_name << "\n";
  report << "input=" << input_summary << "\n";
  report << "load_count=" << loads.size() << "\n";
  report << "topology_nodes=" << result.output.topology.get_size() << "\n";
  report << "htree_levels=" << result.output.levels.size() << "\n";
  report << "inserted_buffers=" << result.output.inserted_insts.size() << "\n";
  report << "inserted_nets=" << result.output.inserted_nets.size() << "\n";
  report << "char_grid_unit_um=" << result.diagnostics.char_wirelength_unit_um
         << ", iterations=" << result.diagnostics.char_wirelength_iterations
         << ", distinct_level_bins=" << result.diagnostics.char_unique_level_bins
         << ", adapted=" << (result.diagnostics.char_grid_adapted ? "true" : "false") << "\n";
  report << "char_limits max_slew_ns=" << result.diagnostics.char_max_slew_ns << ", max_cap_pf=" << result.diagnostics.char_max_cap_pf
         << "\n";
  report << "frontier_candidate_solution_count=" << observation.selected_candidate_solution_count
         << ", candidate_frontier_entry_count=" << observation.selected_candidate_frontier_entry_count << "\n";
  report << "frontier_feasible_solution_count=" << observation.selected_feasible_solution_count
         << ", feasible_frontier_entry_count=" << observation.selected_feasible_frontier_entry_count << "\n";
  report << "frontier_solution_count=" << delay_power_points.size() << ", pareto_solution_count="
         << std::ranges::count_if(delay_power_points, [](const DelayPowerPoint& point) -> bool { return point.is_pareto; }) << "\n";
  const auto selection_summary = BuildDelayPowerChoiceSummary(result);
  if (selection_summary.has_value()) {
    report << std::setprecision(9);
    report << "delay_power_selection_summary selection_policy="
           << (observation.used_boundary_relaxation ? "boundary_relaxation" : "global_frontier_pareto_power_median")
           << ", frontier_selection_pool_size=" << selection_summary->frontier_selection_pool_size
           << ", selection_pool_size=" << selection_summary->selection_pool_size
           << ", pareto_solution_count=" << selection_summary->pareto_solution_count << ", pareto_power_median_index=";
    if (selection_summary->pareto_power_median_index.has_value()) {
      report << *selection_summary->pareto_power_median_index;
    } else {
      report << "none";
    }
    report << ", selected_pareto_power_order_index=";
    if (selection_summary->selected_pareto_power_order_index.has_value()) {
      report << *selection_summary->selected_pareto_power_order_index;
    } else {
      report << "none";
    }
    report << ", median_uses_lower_index=" << (selection_summary->median_uses_lower_index ? "true" : "false")
           << ", selected_on_pareto_front=" << (selection_summary->selected_on_pareto_front ? "true" : "false");
    if (result.output.best_char.has_value()) {
      report << ", selected_pattern_id=" << result.output.best_char->get_pattern_id().local_id
             << ", selected_delay=" << result.output.best_char->get_delay() << ", selected_power=" << result.output.best_char->get_power();
    }
    report << "\n";
    report << std::setprecision(3);
  }
  for (const auto& summary : CollectBufferMasterSummaries(result.output.inserted_insts)) {
    report << "buffer_master=" << summary.cell_master << ", count=" << summary.count;
    if (summary.drive_strength >= 0) {
      report << ", drive_strength=" << summary.drive_strength;
    }
    report << "\n";
  }
  report << "root_output_pin=" << (result.output.root_output_pin != nullptr ? result.output.root_output_pin->get_name() : "<null>") << "\n";
  for (const auto& point : delay_power_points) {
    if (!point.is_pareto && !point.is_selected) {
      continue;
    }

    report << (point.is_selected ? "selected_solution" : "pareto_solution") << " pattern_id=" << point.entry->get_pattern_id().local_id
           << ", delay=" << point.entry->get_delay() << ", power=" << point.entry->get_power() << ", norm_delay=" << point.normalized_delay
           << ", norm_power=" << point.normalized_power << ", driven_cap_idx=" << point.entry->get_driven_cap_idx()
           << ", output_slew_idx=" << point.entry->get_output_slew_idx() << ", load_cap_idx=" << point.entry->get_load_cap_idx() << "\n";
  }
  for (std::size_t level_index = 0; level_index < result.output.levels.size(); ++level_index) {
    const auto& level = result.output.levels.at(level_index);
    report << "level[" << level_index << "] requested_dbu=" << level.requested_length_dbu << ", requested_um=" << level.requested_length_um
           << ", aligned_idx=" << level.aligned_length_idx << ", aligned_um=" << level.aligned_length_um
           << ", segment_pattern_id=" << level.segment_pattern_id.local_id << "\n";
  }
  report << "artifacts=cts.log, topology.svg, materialized_htree.svg, pareto_delay_power.svg\n";
  report << "output_dir=" << paths.output_dir.string() << "\n";
  return report.str();
}

}  // namespace icts_test::htree
