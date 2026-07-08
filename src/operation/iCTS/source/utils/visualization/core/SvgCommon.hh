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
 * @file SvgCommon.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Shared SVG rendering helpers for iCTS visualization utilities.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "Point.hh"
#include "design/Pin.hh"
#include "spatial/Tree.hh"

namespace icts::visualization::detail {

constexpr int kCanvasMax = 1000;
constexpr int kSvgMargin = 20;
constexpr int kClusterCenterRadius = 8;
constexpr int kRootRadius = 8;
constexpr int kNodeRadius = 5;
constexpr int kCenterCrossHalfSize = 8;
constexpr int kLoadRadius = 4;
constexpr int kRootSpokeWidth = 2;
constexpr int kNeighborCountForAdjacency = 3;
constexpr int kSameColorPenalty = 1000;
constexpr int kColorDistancePenaltyBase = 9;
constexpr std::size_t kInvalidNodeId = std::numeric_limits<std::size_t>::max();
constexpr double kSvgLegendX = 18.0;
constexpr double kSvgLegendRowHeight = 18.0;
constexpr double kSvgLegendFrameOpacity = 0.88;
constexpr double kReportSinkLoadRadius = 2.0;
constexpr double kReportDriverRadius = 3.0;
constexpr double kReportCtsBufferSize = 4.5;
constexpr const char* kSvgOpenTagPrefix = R"(<svg xmlns="http://www.w3.org/2000/svg" width=")";
constexpr const char* kSvgHeightTag = R"(" height=")";
constexpr const char* kSvgViewBoxPrefix = R"(" viewBox="0 0 )";
constexpr const char* kSvgOpenTagSuffix = R"(">
)";
constexpr const char* kSvgBackgroundRect = R"(<rect width="100%" height="100%" fill="#ffffff" />
)";
constexpr const char* kSvgClosingTag = R"(</svg>
)";
constexpr const char* kSvgColorSinkLoad = "#1f77b4";
constexpr const char* kSvgColorDriverRoot = "#d62728";
constexpr const char* kSvgColorRoutedSinkNet = "#2ca25f";
constexpr const char* kSvgColorFlylineRootNet = "#6a3d9a";
constexpr const char* kSvgColorDegradedInternalNet = "#ff8c42";
constexpr const char* kSvgColorSinkLevelNet = "#0f766e";
constexpr const char* kSvgColorTopologyEdge = "#9aa0a6";
constexpr const char* kSvgColorTopologyNode = "#3d3d3d";
constexpr const char* kSvgColorLoadStroke = "#0c4068";
constexpr const char* kSvgColorNodeStroke = "#ffffff";
constexpr const char* kSvgColorLegendText = "#222222";
constexpr const char* kSvgColorLegendFill = "#ffffff";
constexpr const char* kSvgColorLegendStroke = "#d0d0d0";
constexpr const char* kSvgColorBufferFillDefault = "#ffbf69";
constexpr const char* kSvgColorBufferStrokeDefault = "#8c4f00";
inline constexpr std::array<const char*, 6> kSvgBufferFillPalette = {
    "#ffe0b2", "#ffbf69", "#f4a261", "#e76f51", "#d1495b", "#9c6644",
};
inline constexpr std::array<const char*, 6> kSvgBufferStrokePalette = {
    "#b36a00", "#8c4f00", "#9a4d1f", "#91361f", "#7a1f30", "#5d4037",
};

struct Bounds
{
  int min_x = std::numeric_limits<int>::max();
  int min_y = std::numeric_limits<int>::max();
  int max_x = std::numeric_limits<int>::min();
  int max_y = std::numeric_limits<int>::min();
  bool valid = false;
};

struct SvgTransform
{
  int min_x = 0;
  int max_y = 0;
  double scale = 1.0;
  int margin = kSvgMargin;
  int width = 0;
  int height = 0;
};

inline const std::array<const char*, 16> kPalette
    = {"#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
       "#bcbd22", "#17becf", "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"};

[[nodiscard]] inline auto MapX(const SvgTransform& transform, int x_coord) -> double
{
  return ((x_coord - transform.min_x) * transform.scale) + transform.margin;
}

[[nodiscard]] inline auto MapY(const SvgTransform& transform, int y_coord) -> double
{
  return ((transform.max_y - y_coord) * transform.scale) + transform.margin;
}

inline auto ComputeLoadCentroid(const std::vector<icts::Pin*>& loads) -> icts::Point<int>
{
  if (loads.empty()) {
    return {-1, -1};
  }
  long long sum_x = 0;
  long long sum_y = 0;
  for (const auto* pin : loads) {
    if (pin == nullptr) {
      continue;
    }
    const auto& location = pin->get_location();
    sum_x += location.get_x();
    sum_y += location.get_y();
  }
  return {static_cast<int>(sum_x / static_cast<long long>(loads.size())), static_cast<int>(sum_y / static_cast<long long>(loads.size()))};
}

inline auto ComputeBounds(const std::vector<icts::Pin*>& loads, const std::vector<icts::Point<int>>& extras) -> Bounds
{
  Bounds bounds;
  for (const auto* pin : loads) {
    if (pin == nullptr) {
      continue;
    }
    const auto& location = pin->get_location();
    bounds.min_x = std::min(bounds.min_x, location.get_x());
    bounds.min_y = std::min(bounds.min_y, location.get_y());
    bounds.max_x = std::max(bounds.max_x, location.get_x());
    bounds.max_y = std::max(bounds.max_y, location.get_y());
    bounds.valid = true;
  }
  for (const auto& point : extras) {
    bounds.min_x = std::min(bounds.min_x, point.get_x());
    bounds.min_y = std::min(bounds.min_y, point.get_y());
    bounds.max_x = std::max(bounds.max_x, point.get_x());
    bounds.max_y = std::max(bounds.max_y, point.get_y());
    bounds.valid = true;
  }
  return bounds;
}

inline auto ComputeBounds(const std::vector<icts::Pin*>& loads, const icts::Tree& tree) -> Bounds
{
  Bounds bounds;
  for (const auto* pin : loads) {
    if (pin == nullptr) {
      continue;
    }
    const auto& location = pin->get_location();
    bounds.min_x = std::min(bounds.min_x, location.get_x());
    bounds.min_y = std::min(bounds.min_y, location.get_y());
    bounds.max_x = std::max(bounds.max_x, location.get_x());
    bounds.max_y = std::max(bounds.max_y, location.get_y());
    bounds.valid = true;
  }
  for (std::size_t id = 0; id < tree.get_size(); ++id) {
    const auto* node = tree.get_node(id);
    if (node == nullptr) {
      continue;
    }
    const auto& position = node->get_position();
    if (position.get_x() < 0 || position.get_y() < 0) {
      continue;
    }
    bounds.min_x = std::min(bounds.min_x, position.get_x());
    bounds.min_y = std::min(bounds.min_y, position.get_y());
    bounds.max_x = std::max(bounds.max_x, position.get_x());
    bounds.max_y = std::max(bounds.max_y, position.get_y());
    bounds.valid = true;
  }
  return bounds;
}

inline auto MakeTransform(const Bounds& bounds) -> SvgTransform
{
  SvgTransform transform;
  if (!bounds.valid) {
    transform.width = 2 * transform.margin;
    transform.height = 2 * transform.margin;
    return transform;
  }

  const int width = std::max(1, bounds.max_x - bounds.min_x);
  const int height = std::max(1, bounds.max_y - bounds.min_y);
  const double scale = static_cast<double>(kCanvasMax) / static_cast<double>(std::max(width, height));

  transform.min_x = bounds.min_x;
  transform.max_y = bounds.max_y;
  transform.scale = scale;
  transform.width = static_cast<int>(std::lround(width * scale)) + (2 * transform.margin);
  transform.height = static_cast<int>(std::lround(height * scale)) + (2 * transform.margin);
  return transform;
}

inline auto AddUniqueNeighbor(std::vector<std::size_t>& neighbors, std::size_t neighbor_id) -> void
{
  if (std::ranges::find(neighbors, neighbor_id) == neighbors.end()) {
    neighbors.push_back(neighbor_id);
  }
}

inline auto BuildAdjacency(const std::vector<icts::Point<int>>& centers) -> std::vector<std::vector<std::size_t>>
{
  std::vector<std::vector<std::size_t>> adjacency(centers.size());
  if (centers.size() <= 1) {
    return adjacency;
  }

  for (std::size_t i = 0; i < centers.size(); ++i) {
    std::vector<std::pair<double, std::size_t>> nearest;
    nearest.reserve(centers.size() - 1);
    for (std::size_t j = 0; j < centers.size(); ++j) {
      if (i == j) {
        continue;
      }
      const auto delta_x = static_cast<double>(centers.at(i).get_x() - centers.at(j).get_x());
      const auto delta_y = static_cast<double>(centers.at(i).get_y() - centers.at(j).get_y());
      nearest.emplace_back((delta_x * delta_x) + (delta_y * delta_y), j);
    }
    std::ranges::sort(nearest);

    const std::size_t link_count = std::min<std::size_t>(kNeighborCountForAdjacency, nearest.size());
    for (std::size_t link = 0; link < link_count; ++link) {
      const auto neighbor_id = nearest.at(link).second;
      AddUniqueNeighbor(adjacency.at(i), neighbor_id);
      AddUniqueNeighbor(adjacency.at(neighbor_id), i);
    }
  }
  return adjacency;
}

inline auto ChoosePaletteIndex(const std::vector<std::vector<std::size_t>>& adjacency, const std::vector<int>& assigned,
                               std::size_t cluster_id) -> int
{
  const int palette_size = static_cast<int>(kPalette.size());
  int best_index = 0;
  int best_penalty = std::numeric_limits<int>::max();

  for (int color_index = 0; color_index < palette_size; ++color_index) {
    int penalty = 0;
    for (const auto neighbor_id : adjacency.at(cluster_id)) {
      const int neighbor_color = assigned.at(neighbor_id);
      if (neighbor_color < 0) {
        continue;
      }
      const int diff = std::abs(color_index - neighbor_color);
      const int cyclic_diff = std::min(diff, palette_size - diff);
      if (cyclic_diff == 0) {
        penalty += kSameColorPenalty;
      } else {
        penalty += std::max(0, kColorDistancePenaltyBase - cyclic_diff);
      }
    }
    if (penalty < best_penalty) {
      best_penalty = penalty;
      best_index = color_index;
    }
  }
  return best_index;
}

inline auto BuildClusterColors(std::size_t cluster_count, const std::vector<icts::Point<int>>& centers) -> std::vector<std::string>
{
  std::vector<std::string> color_table(cluster_count, kPalette.at(0));
  if (cluster_count == 0) {
    return color_table;
  }

  const std::size_t center_count = std::min(cluster_count, centers.size());
  if (center_count > 0) {
    const auto center_prefix = std::vector<icts::Point<int>>(centers.begin(), centers.begin() + static_cast<std::ptrdiff_t>(center_count));
    const auto adjacency = BuildAdjacency(center_prefix);

    std::vector<std::size_t> order(center_count);
    std::iota(order.begin(), order.end(), 0);
    std::ranges::sort(order, [&adjacency](std::size_t lhs, std::size_t rhs) -> bool {
      if (adjacency.at(lhs).size() != adjacency.at(rhs).size()) {
        return adjacency.at(lhs).size() > adjacency.at(rhs).size();
      }
      return lhs < rhs;
    });

    std::vector<int> assigned(center_count, -1);
    for (const auto cluster_id : order) {
      assigned.at(cluster_id) = ChoosePaletteIndex(adjacency, assigned, cluster_id);
    }
    for (std::size_t cluster_id = 0; cluster_id < center_count; ++cluster_id) {
      color_table.at(cluster_id) = kPalette.at(static_cast<std::size_t>(assigned.at(cluster_id) % static_cast<int>(kPalette.size())));
    }
  }

  constexpr std::size_t color_stride = 5;
  for (std::size_t cluster_id = center_count; cluster_id < cluster_count; ++cluster_id) {
    color_table.at(cluster_id) = kPalette.at((cluster_id * color_stride) % kPalette.size());
  }
  return color_table;
}

}  // namespace icts::visualization::detail
