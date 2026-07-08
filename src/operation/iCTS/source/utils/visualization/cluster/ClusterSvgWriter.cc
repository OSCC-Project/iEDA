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
 * @file ClusterSvgWriter.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Cluster-specific SVG writer for iCTS visualization utilities.
 */

#include "visualization/cluster/ClusterSvgWriter.hh"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "design/Pin.hh"
#include "spatial/Point.hh"
#include "visualization/core/SvgCommon.hh"

namespace icts::visualization::cluster {
namespace {

using detail::BuildClusterColors;
using detail::ComputeBounds;
using detail::ComputeLoadCentroid;
using detail::MakeTransform;
using detail::MapX;
using detail::MapY;

constexpr int kSvgPrecision = 2;
constexpr int kComparisonOuterMargin = 24;
constexpr int kComparisonHeaderHeight = 58;
constexpr int kComparisonPanelGap = 44;

struct ClusterRenderData
{
  std::size_t cluster_count = 0;
  std::vector<std::vector<icts::Pin*>> cluster_loads;
  std::vector<icts::Point<int>> cluster_roots;
  std::vector<std::string> colors;
};

auto FormatSvgNumber(double value) -> std::string
{
  std::ostringstream stream;
  stream.setf(std::ostringstream::fixed, std::ostringstream::floatfield);
  stream << std::setprecision(kSvgPrecision) << value;
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

auto CalcClusterCount(const std::unordered_map<const icts::Pin*, std::size_t>& cluster_map, const std::vector<icts::Point<int>>& centers)
    -> std::size_t
{
  std::size_t cluster_count = 1;
  for (const auto& [pin, cluster_id] : cluster_map) {
    (void) pin;
    cluster_count = std::max(cluster_count, cluster_id + 1U);
  }
  return std::max(cluster_count, centers.size());
}

auto BuildClusterLoads(std::size_t cluster_count, const std::vector<icts::Pin*>& loads,
                       const std::unordered_map<const icts::Pin*, std::size_t>& cluster_map) -> std::vector<std::vector<icts::Pin*>>
{
  std::vector<std::vector<icts::Pin*>> cluster_loads(cluster_count);
  for (auto* pin : loads) {
    const auto cluster_itr = cluster_map.find(pin);
    const std::size_t cluster_id = cluster_itr == cluster_map.end() ? 0 : cluster_itr->second;
    if (cluster_id >= cluster_loads.size()) {
      cluster_loads.resize(cluster_id + 1);
    }
    cluster_loads.at(cluster_id).push_back(pin);
  }
  return cluster_loads;
}

auto BuildClusterRoots(std::size_t cluster_count, const std::vector<icts::Point<int>>& centers,
                       const std::vector<std::vector<icts::Pin*>>& cluster_loads, const std::vector<icts::Pin*>& loads)
    -> std::vector<icts::Point<int>>
{
  std::vector<icts::Point<int>> cluster_roots(cluster_count, ComputeLoadCentroid(loads));
  for (std::size_t cluster_id = 0; cluster_id < cluster_count; ++cluster_id) {
    if (cluster_id < centers.size()) {
      cluster_roots.at(cluster_id) = centers.at(cluster_id);
      continue;
    }
    if (!cluster_loads.at(cluster_id).empty()) {
      cluster_roots.at(cluster_id) = ComputeLoadCentroid(cluster_loads.at(cluster_id));
    }
  }
  return cluster_roots;
}

auto BuildClusterRenderData(const std::vector<icts::Pin*>& loads, const std::unordered_map<const icts::Pin*, std::size_t>& cluster_map,
                            const std::vector<icts::Point<int>>& centers) -> ClusterRenderData
{
  ClusterRenderData render_data;
  render_data.cluster_count = CalcClusterCount(cluster_map, centers);
  render_data.cluster_loads = BuildClusterLoads(render_data.cluster_count, loads, cluster_map);
  render_data.cluster_roots = BuildClusterRoots(render_data.cluster_count, centers, render_data.cluster_loads, loads);
  render_data.colors = BuildClusterColors(render_data.cluster_count, render_data.cluster_roots);
  return render_data;
}

auto WriteClusterSpokes(std::ofstream& output_stream, const detail::SvgTransform& transform,
                        const std::vector<icts::Point<int>>& cluster_roots, const std::vector<std::vector<icts::Pin*>>& cluster_loads,
                        const std::vector<std::string>& colors) -> void
{
  for (std::size_t cluster_id = 0; cluster_id < cluster_roots.size(); ++cluster_id) {
    const auto& root = cluster_roots.at(cluster_id);
    const double root_x = MapX(transform, root.get_x());
    const double root_y = MapY(transform, root.get_y());
    const auto& cluster_color = colors.at(cluster_id % colors.size());
    for (const auto* pin : cluster_loads.at(cluster_id)) {
      if (pin == nullptr) {
        continue;
      }
      const auto& location = pin->get_location();
      const double location_x = MapX(transform, location.get_x());
      const double location_y = MapY(transform, location.get_y());
      output_stream << "<line x1=\"" << FormatSvgNumber(root_x) << "\" y1=\"" << FormatSvgNumber(root_y) << "\" x2=\""
                    << FormatSvgNumber(location_x) << "\" y2=\"" << FormatSvgNumber(location_y) << R"(" stroke=")" << cluster_color
                    << R"(" stroke-width="1.4" stroke-dasharray="7,4" stroke-opacity="0.65" />
)";
    }
  }
}

auto WriteClusterLoadMarkers(std::ofstream& output_stream, const detail::SvgTransform& transform, const std::vector<icts::Pin*>& loads,
                             const std::unordered_map<const icts::Pin*, std::size_t>& cluster_map, const std::vector<std::string>& colors)
    -> void
{
  for (const auto* pin : loads) {
    if (pin == nullptr) {
      continue;
    }
    const auto cluster_itr = cluster_map.find(pin);
    const std::size_t cluster_id = cluster_itr == cluster_map.end() ? 0 : cluster_itr->second;
    const auto& location = pin->get_location();
    const double location_x = MapX(transform, location.get_x());
    const double location_y = MapY(transform, location.get_y());
    output_stream << "<circle cx=\"" << FormatSvgNumber(location_x) << "\" cy=\"" << FormatSvgNumber(location_y) << "\" r=\""
                  << detail::kLoadRadius << "\" fill=\"" << colors.at(cluster_id % colors.size()) << "\" fill-opacity=\"0.82\" />\n";
  }
}

auto WriteClusterCenters(std::ofstream& output_stream, const detail::SvgTransform& transform, const std::vector<icts::Point<int>>& centers,
                         const std::vector<std::string>& colors) -> void
{
  for (std::size_t index = 0; index < centers.size(); ++index) {
    const auto& center = centers.at(index);
    const double center_x = MapX(transform, center.get_x());
    const double center_y = MapY(transform, center.get_y());
    output_stream << R"(<circle cx=")" << FormatSvgNumber(center_x) << R"(" cy=")" << FormatSvgNumber(center_y) << R"(" r=")"
                  << detail::kClusterCenterRadius << R"(" fill="none" stroke=")" << colors.at(index % colors.size())
                  << R"(" stroke-width="2.4" />
)";
    output_stream << "<line x1=\"" << FormatSvgNumber(center_x - detail::kCenterCrossHalfSize) << "\" y1=\"" << FormatSvgNumber(center_y)
                  << "\" x2=\"" << FormatSvgNumber(center_x + detail::kCenterCrossHalfSize) << "\" y2=\"" << FormatSvgNumber(center_y)
                  << R"(" stroke="#222222" stroke-width="1.4" />
)";
    output_stream << "<line x1=\"" << FormatSvgNumber(center_x) << "\" y1=\"" << FormatSvgNumber(center_y - detail::kCenterCrossHalfSize)
                  << "\" x2=\"" << FormatSvgNumber(center_x) << "\" y2=\"" << FormatSvgNumber(center_y + detail::kCenterCrossHalfSize)
                  << R"(" stroke="#222222" stroke-width="1.4" />
)";
  }
}

auto WriteClusterRootOutlines(std::ofstream& output_stream, const detail::SvgTransform& transform,
                              const std::vector<icts::Point<int>>& cluster_roots, const std::vector<std::string>& colors) -> void
{
  const auto root_outline_radius = static_cast<double>(detail::kRootRadius + 2);
  for (std::size_t cluster_id = 0; cluster_id < cluster_roots.size(); ++cluster_id) {
    const auto& root = cluster_roots.at(cluster_id);
    const double root_x = MapX(transform, root.get_x());
    const double root_y = MapY(transform, root.get_y());
    output_stream << "<circle cx=\"" << FormatSvgNumber(root_x) << "\" cy=\"" << FormatSvgNumber(root_y) << "\" r=\"" << root_outline_radius
                  << R"(" fill="none" stroke=")" << colors.at(cluster_id % colors.size()) << R"(" stroke-width="2.0" stroke-opacity="0.9" />
)";
  }
}

auto WriteClusterDrawing(std::ofstream& output_stream, const detail::SvgTransform& transform, const std::vector<icts::Pin*>& loads,
                         const std::unordered_map<const icts::Pin*, std::size_t>& cluster_map, const std::vector<icts::Point<int>>& centers,
                         const ClusterRenderData& render_data) -> void
{
  WriteClusterSpokes(output_stream, transform, render_data.cluster_roots, render_data.cluster_loads, render_data.colors);
  WriteClusterLoadMarkers(output_stream, transform, loads, cluster_map, render_data.colors);
  WriteClusterCenters(output_stream, transform, centers, render_data.colors);
  WriteClusterRootOutlines(output_stream, transform, render_data.cluster_roots, render_data.colors);
}

auto WriteClusterPanel(std::ofstream& output_stream, const detail::SvgTransform& transform, double origin_x, double origin_y,
                       const std::string& title, const std::vector<icts::Pin*>& loads,
                       const std::unordered_map<const icts::Pin*, std::size_t>& cluster_map, const std::vector<icts::Point<int>>& centers,
                       const ClusterRenderData& render_data) -> void
{
  output_stream << R"(<text x=")" << FormatSvgNumber(origin_x) << R"(" y=")" << FormatSvgNumber(origin_y - 26.0)
                << R"(" font-size="16" font-weight="700">)" << EscapeXml(title) << R"(</text>
)";
  output_stream << R"(<text x=")" << FormatSvgNumber(origin_x) << R"(" y=")" << FormatSvgNumber(origin_y - 8.0)
                << R"(" font-size="12" fill="#4a5568">loads=)" << loads.size() << ", clusters=" << render_data.cluster_count << R"(</text>
)";
  output_stream << "<g transform=\"translate(" << FormatSvgNumber(origin_x) << " " << FormatSvgNumber(origin_y) << ")\">\n";
  output_stream << R"(<rect x="0" y="0" width=")" << FormatSvgNumber(transform.width) << R"(" height=")"
                << FormatSvgNumber(transform.height) << R"(" fill="#ffffff" stroke="#d0d7de" />
)";
  WriteClusterDrawing(output_stream, transform, loads, cluster_map, centers, render_data);
  output_stream << R"(</g>
)";
}

}  // namespace

auto WriteClusterSvg(const std::string& path, const std::vector<icts::Pin*>& loads,
                     const std::unordered_map<const icts::Pin*, std::size_t>& cluster_map, const std::vector<icts::Point<int>>& centers)
    -> bool
{
  if (loads.empty()) {
    return false;
  }

  const auto render_data = BuildClusterRenderData(loads, cluster_map, centers);
  const auto bounds = ComputeBounds(loads, render_data.cluster_roots);
  const auto transform = MakeTransform(bounds);

  std::ofstream output_stream(path);
  if (!output_stream.is_open()) {
    return false;
  }

  const auto width = FormatSvgNumber(transform.width);
  const auto height = FormatSvgNumber(transform.height);
  output_stream << detail::kSvgOpenTagPrefix << width << detail::kSvgHeightTag << height << detail::kSvgViewBoxPrefix << width << ' '
                << height << detail::kSvgOpenTagSuffix;
  output_stream << detail::kSvgBackgroundRect;

  WriteClusterDrawing(output_stream, transform, loads, cluster_map, centers, render_data);

  output_stream << detail::kSvgClosingTag;
  return true;
}

auto WriteClusterComparisonSvg(const std::string& path, const std::vector<icts::Pin*>& loads, const std::string& left_title,
                               const std::unordered_map<const icts::Pin*, std::size_t>& left_cluster_map,
                               const std::vector<icts::Point<int>>& left_centers, const std::string& right_title,
                               const std::unordered_map<const icts::Pin*, std::size_t>& right_cluster_map,
                               const std::vector<icts::Point<int>>& right_centers) -> bool
{
  if (loads.empty()) {
    return false;
  }

  const auto left_render_data = BuildClusterRenderData(loads, left_cluster_map, left_centers);
  const auto right_render_data = BuildClusterRenderData(loads, right_cluster_map, right_centers);
  auto bounds_extras = left_render_data.cluster_roots;
  bounds_extras.insert(bounds_extras.end(), right_render_data.cluster_roots.begin(), right_render_data.cluster_roots.end());
  const auto transform = MakeTransform(ComputeBounds(loads, bounds_extras));

  std::ofstream output_stream(path);
  if (!output_stream.is_open()) {
    return false;
  }

  const auto canvas_width = (2 * transform.width) + kComparisonPanelGap + (2 * kComparisonOuterMargin);
  const auto canvas_height = transform.height + kComparisonHeaderHeight + kComparisonOuterMargin;
  const auto width = FormatSvgNumber(canvas_width);
  const auto height = FormatSvgNumber(canvas_height);
  output_stream << detail::kSvgOpenTagPrefix << width << detail::kSvgHeightTag << height << detail::kSvgViewBoxPrefix << width << ' '
                << height << detail::kSvgOpenTagSuffix;
  output_stream << detail::kSvgBackgroundRect;
  output_stream << R"(<g font-family="monospace" fill="#1f2933">
)";

  const double left_x = kComparisonOuterMargin;
  const double right_x = left_x + static_cast<double>(transform.width) + kComparisonPanelGap;
  constexpr double panel_y = kComparisonHeaderHeight;
  WriteClusterPanel(output_stream, transform, left_x, panel_y, left_title, loads, left_cluster_map, left_centers, left_render_data);
  WriteClusterPanel(output_stream, transform, right_x, panel_y, right_title, loads, right_cluster_map, right_centers, right_render_data);

  output_stream << R"(</g>
)";
  output_stream << detail::kSvgClosingTag;
  return true;
}

}  // namespace icts::visualization::cluster
