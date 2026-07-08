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
 * @file LayerPolicy.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-29
 * @brief CTS visualization semantic layer and palette policy implementation.
 */

#include "report/visualization/gds/layer/LayerPolicy.hh"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <ostream>

#include "Log.hh"
#include "report/visualization/drawing/Drawing.hh"

namespace icts::visualization {
namespace {

auto palette() -> const std::vector<std::string>&
{
  static const std::vector<std::string> colors = {
      "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#C44E52", "#000000", "#F0E442",
  };
  return colors;
}

auto normalizeTopologyLevel(int topology_level, int topology_depth) -> int
{
  if (topology_level >= 0) {
    return topology_level;
  }
  if (topology_depth >= 0) {
    return topology_depth;
  }
  return 0;
}

auto colorFor(std::size_t clock_index, int level, std::size_t offset) -> std::string
{
  const auto& colors = palette();
  const auto safe_level = static_cast<std::size_t>(std::max(level, 0));
  return colors.at((clock_index * 3U + safe_level + offset) % colors.size());
}

auto layerKeysEqual(const LayerPolicy::LayerKey& lhs, const LayerPolicy::LayerKey& rhs) -> bool
{
  return lhs.kind == rhs.kind && lhs.view == rhs.view && lhs.synthesis_phase == rhs.synthesis_phase && lhs.net_role == rhs.net_role
         && lhs.sink_domain == rhs.sink_domain && lhs.clock_index == rhs.clock_index && lhs.topology_level == rhs.topology_level;
}

auto routeSemanticLabel(const DrawingSegment& segment) -> std::string
{
  switch (segment.net_role) {
    case LayoutNetRole::kDownstream:
      if (segment.sink_domain == SinkDomainKind::kRegular) {
        return "regular_downstream_root";
      }
      if (segment.sink_domain == SinkDomainKind::kHardMacro) {
        return "macro_downstream_root";
      }
      return "downstream_root";
    case LayoutNetRole::kSinkTree:
      return "htree_buffer";
    case LayoutNetRole::kSourceToRoot:
      return "source_to_root";
    case LayoutNetRole::kClockSource:
      return "clock_source";
    case LayoutNetRole::kUnknown:
      return "unknown";
  }
  return "unknown";
}

auto routeUsesTopologyLevel(const DrawingSegment& segment) -> bool
{
  if (segment.net_role == LayoutNetRole::kSinkTree) {
    return true;
  }
  return segment.net_role == LayoutNetRole::kSourceToRoot
         && (segment.synthesis_phase == ClockLayoutPhase::kSourceToRootHTree
             || segment.synthesis_phase == ClockLayoutPhase::kSourceToRootSegment);
}

}  // namespace

auto LayerPolicy::getLayer(const LayerDescriptor& descriptor) -> GdsLayerKey
{
  auto iter = std::ranges::find_if(
      _layers, [&descriptor](const RegisteredLayer& layer) -> bool { return layerKeysEqual(layer.key, descriptor.key); });
  if (iter != _layers.end()) {
    return iter->property.key;
  }

  if (_next_layer >= std::numeric_limits<int16_t>::max()) {
    LOG_WARNING << "CTS GDS report exceeded available semantic layer ids; reusing the last layer id.";
    return GdsLayerKey{.layer = std::numeric_limits<int16_t>::max(), .datatype = 0};
  }

  GdsLayerProperty property{
      .key = GdsLayerKey{.layer = _next_layer++, .datatype = 0},
      .name = descriptor.display_name,
      .color = descriptor.color,
  };
  _layers.push_back(RegisteredLayer{
      .key = descriptor.key,
      .property = property,
  });
  return property.key;
}

auto LayerPolicy::logicCellLayer() -> GdsLayerKey
{
  return getLayer(LayerDescriptor{
      .key = LayerKey{.kind = LayerKind::kLogicCell},
      .display_name = "logic cells",
      .color = "#AEB4BC",
  });
}

auto LayerPolicy::instLayer(const DrawingInst& inst) -> std::optional<GdsLayerKey>
{
  switch (inst.role) {
    case LayoutInstRole::kClockLoad:
      if (inst.sink_domain == SinkDomainKind::kRegular) {
        return getLayer(LayerDescriptor{
            .key = LayerKey{.kind = LayerKind::kRegularSink, .clock_index = inst.clock_index},
            .display_name = inst.clock_name + " regular sinks",
            .color = colorFor(inst.clock_index, 0, 0U),
        });
      }
      if (inst.sink_domain == SinkDomainKind::kHardMacro) {
        return getLayer(LayerDescriptor{
            .key = LayerKey{.kind = LayerKind::kMacroSink, .clock_index = inst.clock_index},
            .display_name = inst.clock_name + " macro sinks",
            .color = colorFor(inst.clock_index, 0, 1U),
        });
      }
      return std::nullopt;
    case LayoutInstRole::kRootBuffer:
      return getLayer(LayerDescriptor{
          .key = LayerKey{.kind = LayerKind::kRootBuffer, .clock_index = inst.clock_index},
          .display_name = inst.clock_name + " root buffers",
          .color = colorFor(inst.clock_index, 0, 2U),
      });
    case LayoutInstRole::kHTreeBuffer:
    case LayoutInstRole::kSourceRootBuffer: {
      const int level = normalizeTopologyLevel(inst.topology_level, inst.topology_depth);
      return getLayer(LayerDescriptor{
          .key = LayerKey{.kind = LayerKind::kHTreeBuffer, .clock_index = inst.clock_index, .topology_level = level},
          .display_name = inst.clock_name + " htree_buffer level " + std::to_string(level),
          .color = colorFor(inst.clock_index, level, 3U),
      });
    }
    case LayoutInstRole::kLogicCell:
    case LayoutInstRole::kClockSource:
    case LayoutInstRole::kUnknown:
      return std::nullopt;
  }
  return std::nullopt;
}

auto LayerPolicy::segmentLayer(const DrawingSegment& segment) -> GdsLayerKey
{
  const int level = normalizeTopologyLevel(segment.topology_level, segment.topology_depth);
  const bool has_meaningful_level = routeUsesTopologyLevel(segment);
  const bool flyline = segment.view == ClockLayoutMode::kFlyline;
  const auto level_suffix = has_meaningful_level ? " level " + std::to_string(level) : std::string{};
  return getLayer(LayerDescriptor{
      .key = LayerKey{
          .kind = LayerKind::kRouteSegment,
          .view = segment.view,
          .synthesis_phase = segment.synthesis_phase,
          .net_role = segment.net_role,
          .sink_domain = segment.sink_domain,
          .clock_index = segment.clock_index,
          .topology_level = has_meaningful_level ? level : 0,
      },
      .display_name = segment.clock_name + (flyline ? " flyline " : " routing ") + ToString(segment.synthesis_phase) + " "
                      + routeSemanticLabel(segment) + level_suffix,
      .color = colorFor(segment.clock_index, level, flyline ? 6U : 4U),
  });
}

auto LayerPolicy::getLayerProperties() const -> std::vector<GdsLayerProperty>
{
  std::vector<GdsLayerProperty> properties;
  properties.reserve(_layers.size());
  for (const auto& layer : _layers) {
    properties.push_back(layer.property);
  }
  return properties;
}

}  // namespace icts::visualization
