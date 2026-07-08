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
 * @file HTreeVisualizationRendering.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief SVG primitive and report rendering helpers for H-tree artifacts.
 */

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <compare>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Inst.hh"
#include "Net.hh"
#include "Pin.hh"
#include "Point.hh"
#include "Tree.hh"
#include "flow/synthesis/htree/HTreeSvgRenderer.hh"
#include "synthesis/htree/HTree.hh"
#include "visualization/core/SvgCommon.hh"

namespace icts_test::htree {

using icts::visualization::detail::ComputeBounds;
using icts::visualization::detail::MakeTransform;
using icts::visualization::detail::MapX;
using icts::visualization::detail::MapY;

constexpr double kBufferHalfSize = 6.0;
constexpr double kRootRingRadius = 10.0;

auto FormatSvgNumber(double value) -> std::string
{
  std::ostringstream stream;
  stream.setf(std::ostringstream::fixed, std::ostringstream::floatfield);
  stream << std::setprecision(2) << value;
  return stream.str();
}

auto EscapeXml(const std::string& text) -> std::string
{
  std::string escaped;
  escaped.reserve(text.size());
  for (const char value : text) {
    switch (value) {
      case '&':
        escaped += "&amp;";
        break;
      case '<':
        escaped += "&lt;";
        break;
      case '>':
        escaped += "&gt;";
        break;
      case '"':
        escaped += "&quot;";
        break;
      case '\'':
        escaped += "&apos;";
        break;
      default:
        escaped.push_back(value);
        break;
    }
  }
  return escaped;
}

namespace {

auto HasValidLocation(const icts::Point<int>& location) -> bool
{
  return location.get_x() >= 0 && location.get_y() >= 0;
}

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

auto FindRenderableLocation(const icts::Pin* pin) -> icts::Point<int>
{
  if (pin == nullptr) {
    return {-1, -1};
  }
  if (HasValidLocation(pin->get_location())) {
    return pin->get_location();
  }
  if (pin->get_inst() != nullptr) {
    return pin->get_inst()->get_location();
  }
  return {-1, -1};
}

auto CollectExtraPoints(const icts::htree::DiagnosticBuild& result) -> std::vector<icts::Point<int>>
{
  std::vector<icts::Point<int>> extra_points;
  extra_points.reserve(result.output.topology.get_size() + result.output.inserted_insts.size() + result.output.inserted_pins.size());

  for (std::size_t node_id = 0; node_id < result.output.topology.get_size(); ++node_id) {
    const auto* node = result.output.topology.get_node(node_id);
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

}  // namespace

auto WriteTooltip(std::ofstream& output_stream, const std::string& text) -> void
{
  output_stream << "<title>" << EscapeXml(text) << "</title>";
}

namespace {

auto IsOriginalLoadPin(const icts::Pin* pin, const std::unordered_set<const icts::Pin*>& original_loads) -> bool
{
  return pin != nullptr && original_loads.contains(pin);
}

auto ResolveNetStrokeColor(bool is_root_net, bool reaches_sink) -> std::string
{
  if (is_root_net) {
    return icts::visualization::detail::kSvgColorFlylineRootNet;
  }
  if (reaches_sink) {
    return icts::visualization::detail::kSvgColorRoutedSinkNet;
  }
  return icts::visualization::detail::kSvgColorDegradedInternalNet;
}

auto ResolveNetStrokeWidth(bool is_root_net, bool reaches_sink) -> double
{
  if (is_root_net) {
    return 2.4;
  }
  if (reaches_sink) {
    return 2.0;
  }
  return 1.6;
}

}  // namespace

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

namespace {

auto BuildBufferRenderStyles(const std::vector<BufferMasterSummary>& summaries) -> std::unordered_map<std::string, BufferRenderStyle>
{
  std::unordered_map<std::string, BufferRenderStyle> styles;
  styles.reserve(summaries.size());
  if (summaries.empty()) {
    return styles;
  }

  constexpr double min_half_size = kBufferHalfSize - 1.5;
  constexpr double max_half_size = kBufferHalfSize + 2.5;
  const std::size_t rank_count = summaries.size() > 1 ? summaries.size() - 1 : 1;

  for (std::size_t index = 0; index < summaries.size(); ++index) {
    const double rank_ratio = summaries.size() == 1 ? 0.5 : static_cast<double>(index) / static_cast<double>(rank_count);
    const double half_size = min_half_size + ((max_half_size - min_half_size) * rank_ratio);
    BufferRenderStyle style;
    style.fill_color
        = icts::visualization::detail::kSvgBufferFillPalette[index % icts::visualization::detail::kSvgBufferFillPalette.size()];
    style.stroke_color
        = icts::visualization::detail::kSvgBufferStrokePalette[index % icts::visualization::detail::kSvgBufferStrokePalette.size()];
    style.half_size = half_size;
    styles.emplace(summaries[index].cell_master, style);
  }

  return styles;
}

auto WriteStraightConnection(std::ofstream& output_stream, const std::string& source_x, const std::string& source_y,
                             const std::string& target_x, const std::string& target_y, const std::string& stroke_color, double stroke_width,
                             const std::string& dash_array, const std::string& tooltip) -> void
{
  output_stream << R"(<line x1=")" << source_x << R"(" y1=")" << source_y << R"(" x2=")" << target_x << R"(" y2=")" << target_y
                << R"(" stroke=")" << stroke_color << R"(" stroke-width=")" << stroke_width << R"(" stroke-linecap="round")";
  if (!dash_array.empty()) {
    output_stream << R"( stroke-dasharray=")" << dash_array << '"';
  }
  output_stream << '>';
  WriteTooltip(output_stream, tooltip);
  output_stream << R"(</line>
)";
}

auto WriteLegend(std::ofstream& output_stream, const icts::visualization::detail::SvgTransform& transform,
                 const std::vector<BufferMasterSummary>& buffer_summaries,
                 const std::unordered_map<std::string, BufferRenderStyle>& buffer_styles) -> void
{
  const double legend_x = icts::visualization::detail::kSvgLegendX;
  const double legend_row_height = icts::visualization::detail::kSvgLegendRowHeight;
  const std::size_t base_row_count = 5U;
  const std::size_t total_row_count = base_row_count + buffer_summaries.size();
  const double legend_height = 24.0 + (legend_row_height * static_cast<double>(total_row_count));
  const double legend_y = std::max(22.0, static_cast<double>(transform.height) - legend_height - 18.0);

  output_stream << R"(<g font-family="monospace" font-size="12" fill=")" << icts::visualization::detail::kSvgColorLegendText << R"(">)";
  output_stream << R"(<rect x=")" << FormatSvgNumber(legend_x - 8.0) << R"(" y=")" << FormatSvgNumber(legend_y - 16.0)
                << R"(" width="280" height=")" << FormatSvgNumber(legend_height) << R"(" rx="6" fill=")"
                << icts::visualization::detail::kSvgColorLegendFill << R"(" fill-opacity=")"
                << FormatSvgNumber(icts::visualization::detail::kSvgLegendFrameOpacity) << R"(" stroke=")"
                << icts::visualization::detail::kSvgColorLegendStroke << R"(" />)";
  output_stream << R"(<text x=")" << FormatSvgNumber(legend_x) << R"(" y=")" << FormatSvgNumber(legend_y) << R"(">Legend</text>)";

  const double row_1_y = legend_y + legend_row_height;
  const double row_2_y = legend_y + (2.0 * legend_row_height);
  const double row_3_y = legend_y + (3.0 * legend_row_height);
  const double row_4_y = legend_y + (4.0 * legend_row_height);
  const double row_5_y = legend_y + (5.0 * legend_row_height);

  output_stream << R"(<circle cx=")" << FormatSvgNumber(legend_x + 6.0) << R"(" cy=")" << FormatSvgNumber(row_1_y - 4.0)
                << R"(" r="4" fill=")" << icts::visualization::detail::kSvgColorSinkLoad << R"(" fill-opacity="0.75" />)";
  output_stream << R"(<text x=")" << FormatSvgNumber(legend_x + 18.0) << R"(" y=")" << FormatSvgNumber(row_1_y) << R"(">sink load</text>)";

  output_stream << R"(<line x1=")" << FormatSvgNumber(legend_x) << R"(" y1=")" << FormatSvgNumber(row_2_y - 4.0) << R"(" x2=")"
                << FormatSvgNumber(legend_x + 12.0) << R"(" y2=")" << FormatSvgNumber(row_2_y - 4.0) << R"(" stroke=")"
                << icts::visualization::detail::kSvgColorTopologyEdge << R"(" stroke-width="1.4" stroke-dasharray="6,4" />)";
  output_stream << R"(<text x=")" << FormatSvgNumber(legend_x + 18.0) << R"(" y=")" << FormatSvgNumber(row_2_y)
                << R"(">H-tree topology</text>)";

  output_stream << R"(<line x1=")" << FormatSvgNumber(legend_x) << R"(" y1=")" << FormatSvgNumber(row_3_y - 4.0) << R"(" x2=")"
                << FormatSvgNumber(legend_x + 12.0) << R"(" y2=")" << FormatSvgNumber(row_3_y - 4.0) << R"(" stroke=")"
                << icts::visualization::detail::kSvgColorRoutedSinkNet << R"(" stroke-width="2.0" />)";
  output_stream << R"(<text x=")" << FormatSvgNumber(legend_x + 18.0) << R"(" y=")" << FormatSvgNumber(row_3_y)
                << R"(">sink-reaching net</text>)";

  output_stream << R"(<line x1=")" << FormatSvgNumber(legend_x) << R"(" y1=")" << FormatSvgNumber(row_4_y - 4.0) << R"(" x2=")"
                << FormatSvgNumber(legend_x + 12.0) << R"(" y2=")" << FormatSvgNumber(row_4_y - 4.0) << R"(" stroke=")"
                << icts::visualization::detail::kSvgColorDegradedInternalNet << R"(" stroke-width="1.6" />)";
  output_stream << R"(<text x=")" << FormatSvgNumber(legend_x + 18.0) << R"(" y=")" << FormatSvgNumber(row_4_y)
                << R"(">internal fanout net</text>)";

  output_stream << R"(<line x1=")" << FormatSvgNumber(legend_x) << R"(" y1=")" << FormatSvgNumber(row_5_y - 4.0) << R"(" x2=")"
                << FormatSvgNumber(legend_x + 12.0) << R"(" y2=")" << FormatSvgNumber(row_5_y - 4.0) << R"(" stroke=")"
                << icts::visualization::detail::kSvgColorFlylineRootNet << R"(" stroke-width="2.4" />)";
  output_stream << R"(<text x=")" << FormatSvgNumber(legend_x + 18.0) << R"(" y=")" << FormatSvgNumber(row_5_y) << R"(">root net</text>)";

  for (std::size_t index = 0; index < buffer_summaries.size(); ++index) {
    const auto& summary = buffer_summaries[index];
    const auto style_itr = buffer_styles.find(summary.cell_master);
    const BufferRenderStyle style = style_itr != buffer_styles.end() ? style_itr->second : BufferRenderStyle{};
    const double row_y = legend_y + (legend_row_height * static_cast<double>(base_row_count + index + 1));
    const double symbol_size = 2.0 * std::min(style.half_size, 6.0);

    output_stream << R"(<rect x=")" << FormatSvgNumber(legend_x + 1.0) << R"(" y=")" << FormatSvgNumber(row_y - 11.0) << R"(" width=")"
                  << FormatSvgNumber(symbol_size) << R"(" height=")" << FormatSvgNumber(symbol_size) << R"(" fill=")" << style.fill_color
                  << R"(" stroke=")" << style.stroke_color << R"(" stroke-width="1.2" />)";

    std::ostringstream label;
    label << summary.cell_master;
    label << " x" << summary.count;
    output_stream << R"(<text x=")" << FormatSvgNumber(legend_x + 18.0) << R"(" y=")" << FormatSvgNumber(row_y) << R"(">)"
                  << EscapeXml(label.str()) << R"(</text>)";
  }
  output_stream << R"(</g>
)";
}

auto WriteTopologyOverlay(std::ofstream& output_stream, const icts::visualization::detail::SvgTransform& transform,
                          const icts::Tree& topology) -> void
{
  for (std::size_t node_id = 0; node_id < topology.get_size(); ++node_id) {
    const auto* node = topology.get_node(node_id);
    if (node == nullptr || node->get_parent() == icts::visualization::detail::kInvalidNodeId || !HasValidLocation(node->get_position())) {
      continue;
    }

    const auto* parent = topology.get_node(node->get_parent());
    if (parent == nullptr || !HasValidLocation(parent->get_position())) {
      continue;
    }

    const auto source_x = FormatSvgNumber(MapX(transform, parent->get_position().get_x()));
    const auto source_y = FormatSvgNumber(MapY(transform, parent->get_position().get_y()));
    const auto target_x = FormatSvgNumber(MapX(transform, node->get_position().get_x()));
    const auto target_y = FormatSvgNumber(MapY(transform, node->get_position().get_y()));
    const std::string tooltip = "tree edge " + std::to_string(parent->get_id()) + " -> " + std::to_string(node->get_id());
    WriteStraightConnection(output_stream, source_x, source_y, target_x, target_y, icts::visualization::detail::kSvgColorTopologyEdge, 1.4,
                            "6,4", tooltip);
  }
}

auto WriteMaterializedNets(std::ofstream& output_stream, const icts::visualization::detail::SvgTransform& transform,
                           const std::unordered_set<const icts::Pin*>& original_loads, const icts::htree::DiagnosticBuild& result) -> void
{
  for (const auto& net_owner : result.output.inserted_nets) {
    const auto* net = net_owner.get();
    if (net == nullptr || net->get_driver() == nullptr) {
      continue;
    }

    const auto driver_location = FindRenderableLocation(net->get_driver());
    if (!HasValidLocation(driver_location)) {
      continue;
    }

    const bool is_root_net = net->get_driver() == result.output.root_output_pin;
    const bool reaches_sink = std::ranges::any_of(
        net->get_loads(), [&original_loads](const icts::Pin* pin) -> bool { return IsOriginalLoadPin(pin, original_loads); });

    const std::string stroke_color = ResolveNetStrokeColor(is_root_net, reaches_sink);
    const double stroke_width = ResolveNetStrokeWidth(is_root_net, reaches_sink);

    const auto source_x = FormatSvgNumber(MapX(transform, driver_location.get_x()));
    const auto source_y = FormatSvgNumber(MapY(transform, driver_location.get_y()));
    for (const auto* load : net->get_loads()) {
      const auto target_location = FindRenderableLocation(load);
      if (!HasValidLocation(target_location)) {
        continue;
      }

      const auto target_x = FormatSvgNumber(MapX(transform, target_location.get_x()));
      const auto target_y = FormatSvgNumber(MapY(transform, target_location.get_y()));
      const std::string tooltip = "net " + net->get_name() + ": "
                                  + (net->get_driver() != nullptr ? net->get_driver()->get_name() : std::string("<null-driver>")) + " -> "
                                  + (load != nullptr ? load->get_name() : std::string("<null-load>"));
      WriteStraightConnection(output_stream, source_x, source_y, target_x, target_y, stroke_color, stroke_width, "", tooltip);
    }
  }
}

auto WriteTopologyNodes(std::ofstream& output_stream, const icts::visualization::detail::SvgTransform& transform,
                        const icts::Tree& topology) -> void
{
  for (std::size_t node_id = 0; node_id < topology.get_size(); ++node_id) {
    const auto* node = topology.get_node(node_id);
    if (node == nullptr || !HasValidLocation(node->get_position())) {
      continue;
    }

    const bool is_root = node->get_parent() == icts::visualization::detail::kInvalidNodeId;
    const auto position_x = FormatSvgNumber(MapX(transform, node->get_position().get_x()));
    const auto position_y = FormatSvgNumber(MapY(transform, node->get_position().get_y()));

    const double radius = is_root ? 6.0 : 4.0;
    const std::string fill_color
        = is_root ? icts::visualization::detail::kSvgColorDriverRoot : icts::visualization::detail::kSvgColorTopologyNode;
    output_stream << R"(<circle cx=")" << position_x << R"(" cy=")" << position_y << R"(" r=")" << radius << R"(" fill=")" << fill_color
                  << R"(" fill-opacity="0.92" stroke=")" << icts::visualization::detail::kSvgColorNodeStroke << R"(" stroke-width="1">)";
    WriteTooltip(output_stream, "tree node " + std::to_string(node->get_id()));
    output_stream << "</circle>\n";
  }
}

auto WriteBuffers(std::ofstream& output_stream, const icts::visualization::detail::SvgTransform& transform,
                  const std::vector<std::unique_ptr<icts::Inst>>& inserted_insts,
                  const std::unordered_map<std::string, BufferRenderStyle>& buffer_styles) -> void
{
  for (const auto& inst_owner : inserted_insts) {
    const auto* inst = inst_owner.get();
    if (inst == nullptr || !HasValidLocation(inst->get_location())) {
      continue;
    }

    const auto style_itr = buffer_styles.find(inst->get_cell_master());
    const BufferRenderStyle style = style_itr != buffer_styles.end() ? style_itr->second : BufferRenderStyle{};
    const double center_x = MapX(transform, inst->get_location().get_x());
    const double center_y = MapY(transform, inst->get_location().get_y());
    output_stream << R"(<rect x=")" << FormatSvgNumber(center_x - style.half_size) << R"(" y=")"
                  << FormatSvgNumber(center_y - style.half_size) << R"(" width=")" << FormatSvgNumber(2.0 * style.half_size)
                  << R"(" height=")" << FormatSvgNumber(2.0 * style.half_size) << R"(" fill=")" << style.fill_color << R"(" stroke=")"
                  << style.stroke_color << R"(" stroke-width="1.2">)";
    WriteTooltip(output_stream, "buffer " + inst->get_name() + " [" + inst->get_cell_master() + "]");
    output_stream << "</rect>\n";
  }
}

auto WriteLoads(std::ofstream& output_stream, const icts::visualization::detail::SvgTransform& transform,
                const std::vector<icts::Pin*>& loads) -> void
{
  for (const auto* load : loads) {
    if (load == nullptr || !HasValidLocation(load->get_location())) {
      continue;
    }

    const auto location_x = FormatSvgNumber(MapX(transform, load->get_location().get_x()));
    const auto location_y = FormatSvgNumber(MapY(transform, load->get_location().get_y()));
    const std::string load_label
        = load->get_name() + (load->get_inst() != nullptr ? " [" + load->get_inst()->get_name() + "]" : std::string{});

    output_stream << R"(<circle cx=")" << location_x << R"(" cy=")" << location_y << R"(" r="4.5" fill=")"
                  << icts::visualization::detail::kSvgColorSinkLoad << R"(" fill-opacity="0.75" stroke=")"
                  << icts::visualization::detail::kSvgColorLoadStroke << R"(" stroke-width="0.8">)";
    WriteTooltip(output_stream, load_label);
    output_stream << "</circle>\n";
  }
}

auto WriteRootMarker(std::ofstream& output_stream, const icts::visualization::detail::SvgTransform& transform,
                     const icts::htree::DiagnosticBuild& result) -> void
{
  const auto root_location = FindRenderableLocation(result.output.root_output_pin);
  if (!HasValidLocation(root_location)) {
    return;
  }

  const auto center_x = FormatSvgNumber(MapX(transform, root_location.get_x()));
  const auto center_y = FormatSvgNumber(MapY(transform, root_location.get_y()));

  output_stream << R"(<circle cx=")" << center_x << R"(" cy=")" << center_y << R"(" r=")" << FormatSvgNumber(kRootRingRadius)
                << R"(" fill="none" stroke=")" << icts::visualization::detail::kSvgColorFlylineRootNet << R"(" stroke-width="2.0">)";
  WriteTooltip(output_stream, "root output pin");
  output_stream << "</circle>\n";
}

}  // namespace

auto WriteMaterializedSvg(const std::filesystem::path& path, const std::vector<icts::Pin*>& loads,
                          const icts::htree::DiagnosticBuild& result) -> bool
{
  const auto extra_points = CollectExtraPoints(result);
  const auto bounds = ComputeBounds(loads, extra_points);
  const auto transform = MakeTransform(bounds);
  const auto buffer_summaries = CollectBufferMasterSummaries(result.output.inserted_insts);
  const auto buffer_styles = BuildBufferRenderStyles(buffer_summaries);

  std::ofstream output_stream(path);
  if (!output_stream.is_open()) {
    return false;
  }

  output_stream << icts::visualization::detail::kSvgOpenTagPrefix << FormatSvgNumber(transform.width)
                << icts::visualization::detail::kSvgHeightTag << FormatSvgNumber(transform.height)
                << icts::visualization::detail::kSvgViewBoxPrefix << FormatSvgNumber(transform.width) << ' '
                << FormatSvgNumber(transform.height) << icts::visualization::detail::kSvgOpenTagSuffix;
  output_stream << icts::visualization::detail::kSvgBackgroundRect;

  const std::unordered_set<const icts::Pin*> original_loads(loads.begin(), loads.end());
  WriteTopologyOverlay(output_stream, transform, result.output.topology);
  WriteMaterializedNets(output_stream, transform, original_loads, result);
  WriteTopologyNodes(output_stream, transform, result.output.topology);
  WriteBuffers(output_stream, transform, result.output.inserted_insts, buffer_styles);
  WriteLoads(output_stream, transform, loads);
  WriteRootMarker(output_stream, transform, result);
  WriteLegend(output_stream, transform, buffer_summaries, buffer_styles);

  output_stream << icts::visualization::detail::kSvgClosingTag;
  return true;
}

}  // namespace icts_test::htree
