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
 * @file SvgVisualization.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief CTS clock-tree SVG visualization generation implementation.
 */

#include "report/visualization/svg/SvgVisualization.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include "Log.hh"
#include "config/Config.hh"
#include "design/ClockLayout.hh"
#include "logger/Schema.hh"
#include "report/visualization/drawing/Drawing.hh"
#include "spatial/Point.hh"
#include "visualization/core/SvgCommon.hh"

namespace icts::visualization {
namespace {

constexpr const char* kDesignSvgLabel = "cts_design.svg";
constexpr const char* kFlylineSvgLabel = "cts_flyline.svg";

enum class WireKind
{
  kRouted,
  kSourceToRoot,
  kFlyline,
  kDegraded
};

struct VisualizationReportStatus
{
  std::string label;
  std::filesystem::path path;
  std::string view_label;
  bool success = false;
  std::string reason;
};

auto ResolveVisualizationDir(const Config& config, const std::filesystem::path& visualization_dir) -> std::filesystem::path
{
  if (!visualization_dir.empty()) {
    return visualization_dir;
  }
  return std::filesystem::path(config.get_visualization_dir());
}

auto HasValidLocation(const Point<int>& point) -> bool
{
  return point.get_x() >= 0 && point.get_y() >= 0;
}

auto wireKindFromRole(LayoutNetRole role, bool flyline) -> WireKind
{
  if (flyline) {
    return WireKind::kFlyline;
  }
  if (role == LayoutNetRole::kSourceToRoot || role == LayoutNetRole::kClockSource) {
    return WireKind::kSourceToRoot;
  }
  return WireKind::kRouted;
}

auto EnsureParentDirectory(const std::filesystem::path& path, std::string& detail) -> bool
{
  const auto parent_path = path.parent_path();
  if (parent_path.empty()) {
    return true;
  }

  std::error_code error;
  std::filesystem::create_directories(parent_path, error);
  if (error) {
    detail = "failed to create report directory " + parent_path.string() + ": " + error.message();
    return false;
  }
  return true;
}

auto MakeFailure(std::string label, const std::filesystem::path& path, std::string view_label, std::string reason)
    -> VisualizationReportStatus
{
  LOG_WARNING << "CTS visualization report " << label << " failed: " << reason;
  return VisualizationReportStatus{
      .label = std::move(label), .path = path, .view_label = std::move(view_label), .success = false, .reason = std::move(reason)};
}

auto MakeSuccess(std::string label, const std::filesystem::path& path, std::string view_label, std::string reason)
    -> VisualizationReportStatus
{
  return VisualizationReportStatus{
      .label = std::move(label), .path = path, .view_label = std::move(view_label), .success = true, .reason = std::move(reason)};
}

auto CollectDrawablePoints(const std::vector<DrawingSegment>& segments, const std::vector<DrawingPin>& pins,
                           const std::vector<DrawingInst>& insts, const std::vector<DrawingLogicCell>& logic_cells)
    -> std::vector<Point<int>>
{
  std::vector<Point<int>> points;
  points.reserve((segments.size() * 2U) + pins.size() + insts.size() + logic_cells.size());
  for (const auto& segment : segments) {
    if (HasValidLocation(segment.begin)) {
      points.push_back(segment.begin);
    }
    if (HasValidLocation(segment.end)) {
      points.push_back(segment.end);
    }
  }
  for (const auto& pin : pins) {
    if (HasValidLocation(pin.location)) {
      points.push_back(pin.location);
    }
  }
  for (const auto& inst : insts) {
    if (HasValidLocation(inst.origin)) {
      points.push_back(inst.origin);
    }
  }
  for (const auto& logic_cell : logic_cells) {
    if (HasValidLocation(logic_cell.origin)) {
      points.push_back(logic_cell.origin);
    }
  }
  return points;
}

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
  for (const char ch : text) {
    switch (ch) {
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
        escaped += ch;
        break;
    }
  }
  return escaped;
}

auto ResolveSegmentStyle(WireKind kind, bool degraded, bool flyline_view) -> std::tuple<const char*, double, double, std::string>
{
  const char* stroke_color = visualization::detail::kSvgColorRoutedSinkNet;
  double stroke_width = 1.8;
  double stroke_opacity = 0.78;
  std::string dash_array;
  if (flyline_view || kind == WireKind::kFlyline) {
    stroke_color = visualization::detail::kSvgColorFlylineRootNet;
    stroke_width = 1.2;
    stroke_opacity = 0.62;
    dash_array = "8,5";
  } else if (degraded || kind == WireKind::kDegraded) {
    stroke_color = visualization::detail::kSvgColorDegradedInternalNet;
    stroke_width = 1.4;
    stroke_opacity = 0.74;
    dash_array = "5,3";
  } else if (kind == WireKind::kSourceToRoot) {
    stroke_color = visualization::detail::kSvgColorFlylineRootNet;
    stroke_width = 1.9;
    stroke_opacity = 0.82;
  }
  return {stroke_color, stroke_width, stroke_opacity, dash_array};
}

auto WriteSvgSegments(std::ofstream& output_stream, const visualization::detail::SvgTransform& transform,
                      const std::vector<DrawingSegment>& segments, bool flyline_view) -> void
{
  output_stream << R"(<g fill="none" stroke-linecap="round">
)";
  for (const auto& segment : segments) {
    const auto begin_x = FormatSvgNumber(visualization::detail::MapX(transform, segment.begin.get_x()));
    const auto begin_y = FormatSvgNumber(visualization::detail::MapY(transform, segment.begin.get_y()));
    const auto end_x = FormatSvgNumber(visualization::detail::MapX(transform, segment.end.get_x()));
    const auto end_y = FormatSvgNumber(visualization::detail::MapY(transform, segment.end.get_y()));
    const auto wire_kind = wireKindFromRole(segment.net_role, flyline_view);
    const auto [stroke_color, stroke_width, stroke_opacity, dash_array] = ResolveSegmentStyle(wire_kind, segment.degraded, flyline_view);
    output_stream << R"(<line x1=")" << begin_x << R"(" y1=")" << begin_y << R"(" x2=")" << end_x << R"(" y2=")" << end_y << R"(" stroke=")"
                  << stroke_color << R"(" stroke-width=")" << FormatSvgNumber(stroke_width) << R"(" stroke-opacity=")"
                  << FormatSvgNumber(stroke_opacity) << '"';
    if (!dash_array.empty()) {
      output_stream << R"( stroke-dasharray=")" << dash_array << '"';
    }
    output_stream << R"( />
)";
  }
  output_stream << R"(</g>
)";
}

auto WriteSvgPins(std::ofstream& output_stream, const visualization::detail::SvgTransform& transform, const std::vector<DrawingPin>& pins)
    -> void
{
  output_stream << R"(<g stroke-width="0.7">
)";
  for (const auto& pin : pins) {
    if (!HasValidLocation(pin.location)) {
      continue;
    }
    const auto location = pin.location;
    if (pin.driver) {
      output_stream << R"(<circle cx=")" << FormatSvgNumber(visualization::detail::MapX(transform, location.get_x())) << R"(" cy=")"
                    << FormatSvgNumber(visualization::detail::MapY(transform, location.get_y())) << R"(" r=")"
                    << FormatSvgNumber(visualization::detail::kReportDriverRadius) << R"(" fill=")"
                    << visualization::detail::kSvgColorDriverRoot << R"(" stroke=")" << visualization::detail::kSvgColorNodeStroke
                    << R"(" />
)";
    } else {
      output_stream << R"(<circle cx=")" << FormatSvgNumber(visualization::detail::MapX(transform, location.get_x())) << R"(" cy=")"
                    << FormatSvgNumber(visualization::detail::MapY(transform, location.get_y())) << R"(" r=")"
                    << FormatSvgNumber(visualization::detail::kReportSinkLoadRadius) << R"(" fill=")"
                    << visualization::detail::kSvgColorSinkLoad << R"(" fill-opacity="0.78" stroke=")"
                    << visualization::detail::kSvgColorLoadStroke << R"(" />
)";
    }
  }
  output_stream << R"(</g>
)";
}

auto WriteSvgInsts(std::ofstream& output_stream, const visualization::detail::SvgTransform& transform,
                   const std::vector<DrawingInst>& insts) -> void
{
  output_stream << R"(<g fill=")" << visualization::detail::kSvgColorBufferFillDefault << R"(" fill-opacity="0.86" stroke=")"
                << visualization::detail::kSvgColorBufferStrokeDefault << R"(" stroke-width="0.8">
)";
  for (const auto& inst : insts) {
    if (!HasValidLocation(inst.origin)) {
      continue;
    }
    const auto location = inst.origin;
    const auto center_x = visualization::detail::MapX(transform, location.get_x());
    const auto center_y = visualization::detail::MapY(transform, location.get_y());
    const double marker_size = visualization::detail::kReportCtsBufferSize;
    output_stream << R"(<rect x=")" << FormatSvgNumber(center_x - marker_size / 2.0) << R"(" y=")"
                  << FormatSvgNumber(center_y - marker_size / 2.0) << R"(" width=")" << FormatSvgNumber(marker_size) << R"(" height=")"
                  << FormatSvgNumber(marker_size) << R"(" />
)";
  }
  output_stream << R"(</g>
)";
}

auto WriteSvgLegend(std::ofstream& output_stream, const visualization::detail::SvgTransform& transform) -> void
{
  const double legend_x = visualization::detail::kSvgLegendX;
  const double legend_row_height = visualization::detail::kSvgLegendRowHeight;
  constexpr std::size_t row_count = 6U;
  const double legend_height = 24.0 + (legend_row_height * static_cast<double>(row_count));
  const double legend_y = std::max(22.0, static_cast<double>(transform.height) - legend_height - 18.0);
  auto row_y = [legend_y, legend_row_height](double row_index) -> double { return legend_y + (row_index * legend_row_height); };

  output_stream << R"(<g font-family="monospace" font-size="12" fill=")" << visualization::detail::kSvgColorLegendText << R"(">)";
  output_stream << R"(<rect x=")" << FormatSvgNumber(legend_x - 8.0) << R"(" y=")" << FormatSvgNumber(legend_y - 16.0)
                << R"(" width="260" height=")" << FormatSvgNumber(legend_height) << R"(" rx="6" fill=")"
                << visualization::detail::kSvgColorLegendFill << R"(" fill-opacity=")"
                << FormatSvgNumber(visualization::detail::kSvgLegendFrameOpacity) << R"(" stroke=")"
                << visualization::detail::kSvgColorLegendStroke << R"(" />)";
  output_stream << R"(<text x=")" << FormatSvgNumber(legend_x) << R"(" y=")" << FormatSvgNumber(legend_y) << R"(">Legend</text>)";

  output_stream << R"(<line x1=")" << FormatSvgNumber(legend_x) << R"(" y1=")" << FormatSvgNumber(row_y(1.0) - 4.0) << R"(" x2=")"
                << FormatSvgNumber(legend_x + 14.0) << R"(" y2=")" << FormatSvgNumber(row_y(1.0) - 4.0) << R"(" stroke=")"
                << visualization::detail::kSvgColorRoutedSinkNet << R"(" stroke-width="1.8" />)";
  output_stream << R"(<text x=")" << FormatSvgNumber(legend_x + 20.0) << R"(" y=")" << FormatSvgNumber(row_y(1.0))
                << R"(">routed segment</text>)";

  output_stream << R"(<line x1=")" << FormatSvgNumber(legend_x) << R"(" y1=")" << FormatSvgNumber(row_y(2.0) - 4.0) << R"(" x2=")"
                << FormatSvgNumber(legend_x + 14.0) << R"(" y2=")" << FormatSvgNumber(row_y(2.0) - 4.0) << R"(" stroke=")"
                << visualization::detail::kSvgColorFlylineRootNet << R"(" stroke-width="1.2" stroke-dasharray="8,5" />)";
  output_stream << R"(<text x=")" << FormatSvgNumber(legend_x + 20.0) << R"(" y=")" << FormatSvgNumber(row_y(2.0))
                << R"(">flyline segment</text>)";

  output_stream << R"(<line x1=")" << FormatSvgNumber(legend_x) << R"(" y1=")" << FormatSvgNumber(row_y(3.0) - 4.0) << R"(" x2=")"
                << FormatSvgNumber(legend_x + 14.0) << R"(" y2=")" << FormatSvgNumber(row_y(3.0) - 4.0) << R"(" stroke=")"
                << visualization::detail::kSvgColorDegradedInternalNet << R"(" stroke-width="1.4" stroke-dasharray="5,3" />)";
  output_stream << R"(<text x=")" << FormatSvgNumber(legend_x + 20.0) << R"(" y=")" << FormatSvgNumber(row_y(3.0))
                << R"(">degraded segment</text>)";

  output_stream << R"(<circle cx=")" << FormatSvgNumber(legend_x + 7.0) << R"(" cy=")" << FormatSvgNumber(row_y(4.0) - 4.0) << R"(" r=")"
                << FormatSvgNumber(visualization::detail::kReportDriverRadius) << R"(" fill=")"
                << visualization::detail::kSvgColorDriverRoot << R"(" />)";
  output_stream << R"(<text x=")" << FormatSvgNumber(legend_x + 20.0) << R"(" y=")" << FormatSvgNumber(row_y(4.0))
                << R"(">driver/root</text>)";

  output_stream << R"(<circle cx=")" << FormatSvgNumber(legend_x + 7.0) << R"(" cy=")" << FormatSvgNumber(row_y(5.0) - 4.0) << R"(" r=")"
                << FormatSvgNumber(visualization::detail::kReportSinkLoadRadius) << R"(" fill=")"
                << visualization::detail::kSvgColorSinkLoad << R"(" />)";
  output_stream << R"(<text x=")" << FormatSvgNumber(legend_x + 20.0) << R"(" y=")" << FormatSvgNumber(row_y(5.0))
                << R"(">sink load</text>)";

  output_stream << R"(<rect x=")" << FormatSvgNumber(legend_x + 4.0) << R"(" y=")" << FormatSvgNumber(row_y(6.0) - 8.0) << R"(" width=")"
                << FormatSvgNumber(visualization::detail::kReportCtsBufferSize) << R"(" height=")"
                << FormatSvgNumber(visualization::detail::kReportCtsBufferSize) << R"(" fill=")"
                << visualization::detail::kSvgColorBufferFillDefault << R"(" stroke=")"
                << visualization::detail::kSvgColorBufferStrokeDefault << R"(" stroke-width="1.0" />)";
  output_stream << R"(<text x=")" << FormatSvgNumber(legend_x + 20.0) << R"(" y=")" << FormatSvgNumber(row_y(6.0))
                << R"(">CTS buffer</text>)";
  output_stream << R"(</g>
)";
}

auto WriteSvgFile(const std::filesystem::path& path, const std::string& label, const std::string& view_label, const Drawing& model,
                  const std::vector<DrawingSegment>& segments, bool flyline_view) -> VisualizationReportStatus
{
  if (segments.empty()) {
    return MakeFailure(label, path, view_label, "no drawable CTS net segments are available");
  }

  const auto points = CollectDrawablePoints(segments, model.pins, model.insts, model.logic_cells);
  const auto bounds = visualization::detail::ComputeBounds({}, points);
  if (!bounds.valid) {
    return MakeFailure(label, path, view_label, "no valid CTS locations are available for SVG bounds");
  }
  const auto transform = visualization::detail::MakeTransform(bounds);

  std::string detail;
  if (!EnsureParentDirectory(path, detail)) {
    return MakeFailure(label, path, view_label, detail);
  }

  std::ofstream output_stream(path);
  if (!output_stream.is_open()) {
    return MakeFailure(label, path, view_label, "failed to open report file for writing");
  }

  output_stream << visualization::detail::kSvgOpenTagPrefix << FormatSvgNumber(transform.width) << visualization::detail::kSvgHeightTag
                << FormatSvgNumber(transform.height) << visualization::detail::kSvgViewBoxPrefix << FormatSvgNumber(transform.width) << ' '
                << FormatSvgNumber(transform.height) << visualization::detail::kSvgOpenTagSuffix;
  output_stream << visualization::detail::kSvgBackgroundRect;
  output_stream << R"(<title>)" << EscapeXml(label) << R"(</title>
)";
  WriteSvgSegments(output_stream, transform, segments, flyline_view);
  WriteSvgInsts(output_stream, transform, model.insts);
  WriteSvgPins(output_stream, transform, model.pins);
  WriteSvgLegend(output_stream, transform);
  output_stream << visualization::detail::kSvgClosingTag;
  output_stream.close();
  if (!output_stream) {
    return MakeFailure(label, path, view_label, "failed while flushing report file to disk");
  }
  return MakeSuccess(label, path, view_label, "generated");
}

auto BuildUnavailableStatuses(const std::filesystem::path& output_dir, const std::string& reason) -> std::vector<VisualizationReportStatus>
{
  return {
      MakeFailure(kDesignSvgLabel, output_dir / kDesignSvgLabel, "svg/design", reason),
      MakeFailure(kFlylineSvgLabel, output_dir / kFlylineSvgLabel, "svg/flyline", reason),
  };
}

auto EmitReportStatusTable(SchemaWriter& reporter, const std::vector<VisualizationReportStatus>& statuses) -> void
{
  TableRows rows;
  rows.reserve(statuses.size());
  for (const auto& status : statuses) {
    rows.push_back({
        status.label,
        status.path.string(),
        status.view_label,
        status.success ? "generated" : "failed",
        status.reason,
    });
    if (!status.success) {
      EmitDiagnostic(reporter, DiagnosticLevel::kWarning, "CTS Report Visualization", "visualization report generation failed",
                     {{"report", status.label}, {"path", status.path.string()}, {"view", status.view_label}, {"reason", status.reason}});
    }
  }
  EmitTable(reporter, "CTS Visualization Reports", {"Report", "Path", "View", "Status", "Detail"}, rows);
}

}  // namespace

auto EmitSvgVisualizations(const SvgVisualizationInput& input) -> SvgVisualizationSummary
{
  LOG_FATAL_IF(input.config == nullptr) << "SVG visualization requires config.";
  LOG_FATAL_IF(input.design == nullptr) << "SVG visualization requires design.";
  LOG_FATAL_IF(input.wrapper == nullptr) << "SVG visualization requires wrapper.";
  LOG_FATAL_IF(input.reporter == nullptr) << "SVG visualization requires reporter.";
  LOG_FATAL_IF(input.clock_layout == nullptr) << "SVG visualization requires clock layout.";
  const auto output_dir = ResolveVisualizationDir(*input.config, input.visualization_dir) / "svg";
  const auto model = DrawingBuilder::build(DrawingInput{
      .design = input.design,
      .wrapper = input.wrapper,
      .clock_layout = input.clock_layout,
  });

  std::vector<VisualizationReportStatus> statuses;
  if (!model.has_clocks) {
    statuses = BuildUnavailableStatuses(output_dir, "CTS design contains no clocks; run CTS or initialize clock data before report");
    EmitReportStatusTable(*input.reporter, statuses);
    return SvgVisualizationSummary{.success = false};
  }
  if (model.design_segments.empty() && model.flyline_segments.empty()) {
    statuses = BuildUnavailableStatuses(output_dir, "CTS design contains no clock nets to visualize");
    EmitReportStatusTable(*input.reporter, statuses);
    return SvgVisualizationSummary{.success = false};
  }

  statuses.push_back(WriteSvgFile(output_dir / kDesignSvgLabel, kDesignSvgLabel, "svg/design", model, model.design_segments, false));
  statuses.push_back(WriteSvgFile(output_dir / kFlylineSvgLabel, kFlylineSvgLabel, "svg/flyline", model, model.flyline_segments, true));

  EmitReportStatusTable(*input.reporter, statuses);
  const bool success = std::ranges::all_of(statuses, [](const auto& status) -> bool { return status.success; });
  return SvgVisualizationSummary{.success = success};
}

}  // namespace icts::visualization
