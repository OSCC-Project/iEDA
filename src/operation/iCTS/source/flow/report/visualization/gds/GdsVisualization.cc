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
 * @file GdsVisualization.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief CTS clock-tree GDSII visualization generation implementation.
 */

#include "report/visualization/gds/GdsVisualization.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "Log.hh"
#include "Point.hh"
#include "config/Config.hh"
#include "design/ClockLayout.hh"
#include "logger/Schema.hh"
#include "report/visualization/drawing/Drawing.hh"
#include "report/visualization/gds/layer/LayerPolicy.hh"
#include "report/visualization/gds/writer/GdsStream.hh"

namespace icts::visualization {
namespace {

constexpr const char* kDesignGdsLabel = "cts_design.gds";
constexpr const char* kFlylineGdsLabel = "cts_flyline.gds";
constexpr const char* kDesignLayerPropertiesLabel = "cts_design.lyp";
constexpr const char* kFlylineLayerPropertiesLabel = "cts_flyline.lyp";

auto resolveVisualizationDir(const Config& config, const std::filesystem::path& visualization_dir) -> std::filesystem::path
{
  if (!visualization_dir.empty()) {
    return visualization_dir;
  }
  return std::filesystem::path(config.get_visualization_dir());
}

auto isValidLocation(const Point<int>& point) -> bool
{
  return point.get_x() >= 0 && point.get_y() >= 0;
}

auto makeBoundary(const GdsLayerKey& key, const Point<int>& origin, int32_t width, int32_t height) -> GdsBoundary
{
  const int32_t safe_width = std::max(width, int32_t{1});
  const int32_t safe_height = std::max(height, int32_t{1});
  const int32_t lx = origin.get_x();
  const int32_t ly = origin.get_y();
  const int32_t ux = lx + safe_width;
  const int32_t uy = ly + safe_height;
  return GdsBoundary{
      .key = key,
      .points = {
          Point<int>(lx, ly),
          Point<int>(ux, ly),
          Point<int>(ux, uy),
          Point<int>(lx, uy),
          Point<int>(lx, ly),
      },
  };
}

auto resolvePathWidthDbu(const Config& config, int32_t dbu_per_um) -> int32_t
{
  if (config.get_wire_width() > 0.0) {
    return std::max(static_cast<int32_t>(config.get_wire_width() * static_cast<double>(std::max(dbu_per_um, int32_t{1}))), int32_t{1});
  }
  return std::max(std::max(dbu_per_um, int32_t{1}) / 20, int32_t{1});
}

auto appendLogicCellBoundaries(GdsLibrary& library, LayerPolicy& layer_policy, const Drawing& model) -> void
{
  for (const auto& logic_cell : model.logic_cells) {
    if (!isValidLocation(logic_cell.origin)) {
      continue;
    }
    const auto key = layer_policy.logicCellLayer();
    library.boundaries.push_back(makeBoundary(key, logic_cell.origin, logic_cell.width_dbu, logic_cell.height_dbu));
  }
}

auto appendClockInstBoundaries(GdsLibrary& library, LayerPolicy& layer_policy, const Drawing& model) -> void
{
  for (const auto& inst : model.insts) {
    const auto layer_key = layer_policy.instLayer(inst);
    if (!layer_key.has_value() || !isValidLocation(inst.origin)) {
      continue;
    }
    library.boundaries.push_back(makeBoundary(*layer_key, inst.origin, inst.width_dbu, inst.height_dbu));
  }
}

auto appendInstBoundaries(GdsLibrary& library, LayerPolicy& layer_policy, const Drawing& model) -> void
{
  appendLogicCellBoundaries(library, layer_policy, model);
  appendClockInstBoundaries(library, layer_policy, model);
}

auto appendSegments(const Config& config, GdsLibrary& library, LayerPolicy& layer_policy, const Drawing& model, ClockLayoutMode view)
    -> void
{
  const bool flyline = view == ClockLayoutMode::kFlyline;
  const int32_t path_width = resolvePathWidthDbu(config, model.dbu_per_um);
  const auto& segments = flyline ? model.flyline_segments : model.design_segments;
  for (const auto& segment : segments) {
    if (!isValidLocation(segment.begin) || !isValidLocation(segment.end)) {
      continue;
    }
    const auto key = layer_policy.segmentLayer(segment);
    library.paths.push_back(GdsPath{
        .key = key,
        .width_dbu = flyline ? std::max(path_width / 2, int32_t{1}) : path_width,
        .points = {segment.begin, segment.end},
    });
  }
}

auto buildLibrary(const Config& config, const std::string& library_name, const std::string& structure_name, const Drawing& model,
                  ClockLayoutMode view, LayerPolicy& layer_policy) -> GdsLibrary
{
  GdsLibrary library{
      .library_name = library_name,
      .structure_name = structure_name,
      .dbu_per_um = model.dbu_per_um,
      .paths = {},
      .boundaries = {},
      .texts = {},
  };
  appendInstBoundaries(library, layer_policy, model);
  appendSegments(config, library, layer_policy, model, view);
  return library;
}

auto emitReportTable(SchemaWriter& reporter, const std::filesystem::path& design_gds_path, bool design_success,
                     const std::filesystem::path& design_lyp_path, bool design_lyp_success, const std::filesystem::path& flyline_gds_path,
                     bool flyline_success, const std::filesystem::path& flyline_lyp_path, bool flyline_lyp_success) -> void
{
  TableRows rows = {
      {kDesignGdsLabel, design_gds_path.string(), "gds/design", design_success ? "generated" : "failed"},
      {kDesignLayerPropertiesLabel, design_lyp_path.string(), "lyp/design", design_lyp_success ? "generated" : "failed"},
      {kFlylineGdsLabel, flyline_gds_path.string(), "gds/flyline", flyline_success ? "generated" : "failed"},
      {kFlylineLayerPropertiesLabel, flyline_lyp_path.string(), "lyp/flyline", flyline_lyp_success ? "generated" : "failed"},
  };
  EmitTable(reporter, "CTS GDS Reports", {"Report", "Path", "View", "Status"}, rows);
}

}  // namespace

auto EmitGdsVisualizations(const GdsVisualizationInput& input) -> GdsVisualizationSummary
{
  LOG_FATAL_IF(input.config == nullptr) << "GDS visualization requires config.";
  LOG_FATAL_IF(input.design == nullptr) << "GDS visualization requires design.";
  LOG_FATAL_IF(input.wrapper == nullptr) << "GDS visualization requires wrapper.";
  LOG_FATAL_IF(input.reporter == nullptr) << "GDS visualization requires reporter.";
  LOG_FATAL_IF(input.clock_layout == nullptr) << "GDS visualization requires clock layout.";
  const auto output_dir = resolveVisualizationDir(*input.config, input.visualization_dir) / "gds";
  const auto design_gds_path = output_dir / kDesignGdsLabel;
  const auto flyline_gds_path = output_dir / kFlylineGdsLabel;
  const auto design_lyp_path = output_dir / kDesignLayerPropertiesLabel;
  const auto flyline_lyp_path = output_dir / kFlylineLayerPropertiesLabel;

  const auto model = DrawingBuilder::build(DrawingInput{
      .design = input.design,
      .wrapper = input.wrapper,
      .clock_layout = input.clock_layout,
  });
  if (!model.has_clocks || (model.design_segments.empty() && model.flyline_segments.empty())) {
    LOG_WARNING << "CTS GDS visualization skipped because no clock-tree view data is available.";
    emitReportTable(*input.reporter, design_gds_path, false, design_lyp_path, false, flyline_gds_path, false, flyline_lyp_path, false);
    return GdsVisualizationSummary{.success = false};
  }

  LayerPolicy design_layer_policy;
  auto design_library = buildLibrary(*input.config, "CTS_DESIGN", "CTS_DESIGN", model, ClockLayoutMode::kDesign, design_layer_policy);
  LayerPolicy flyline_layer_policy;
  auto flyline_library = buildLibrary(*input.config, "CTS_FLYLINE", "CTS_FLYLINE", model, ClockLayoutMode::kFlyline, flyline_layer_policy);

  const bool design_success = GdsStream::writeBinary(design_gds_path, design_library);
  const bool flyline_success = GdsStream::writeBinary(flyline_gds_path, flyline_library);
  const bool design_lyp_success = GdsStream::writeLayerProperties(design_lyp_path, design_layer_policy.getLayerProperties());
  const bool flyline_lyp_success = GdsStream::writeLayerProperties(flyline_lyp_path, flyline_layer_policy.getLayerProperties());
  emitReportTable(*input.reporter, design_gds_path, design_success, design_lyp_path, design_lyp_success, flyline_gds_path, flyline_success,
                  flyline_lyp_path, flyline_lyp_success);
  return GdsVisualizationSummary{.success = design_success && flyline_success && design_lyp_success && flyline_lyp_success};
}

}  // namespace icts::visualization
