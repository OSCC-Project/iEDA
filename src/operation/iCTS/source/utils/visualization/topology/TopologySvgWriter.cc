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
 * @file TopologySvgWriter.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Topology-specific SVG writer for iCTS visualization utilities.
 */

#include "visualization/topology/TopologySvgWriter.hh"

#include <cstddef>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "design/Pin.hh"
#include "spatial/Point.hh"
#include "spatial/Tree.hh"
#include "visualization/core/SvgCommon.hh"

namespace icts::visualization::topology {
namespace {

using detail::ComputeBounds;
using detail::MakeTransform;
using detail::MapX;
using detail::MapY;

auto FormatSvgNumber(double value) -> std::string
{
  std::ostringstream stream;
  stream.setf(std::ostringstream::fixed, std::ostringstream::floatfield);
  stream << std::setprecision(2) << value;
  return stream.str();
}

auto WriteTopologyEdges(std::ofstream& output_stream, const detail::SvgTransform& transform, const icts::Tree& tree) -> void
{
  for (std::size_t node_id = 0; node_id < tree.get_size(); ++node_id) {
    const auto* node = tree.get_node(node_id);
    if (node == nullptr || node->get_parent() == detail::kInvalidNodeId) {
      continue;
    }

    const auto* parent = tree.get_node(node->get_parent());
    if (parent == nullptr) {
      continue;
    }

    const auto& source = node->get_position();
    const auto& target = parent->get_position();
    if (source.get_x() < 0 || source.get_y() < 0 || target.get_x() < 0 || target.get_y() < 0) {
      continue;
    }

    const bool is_root_spoke = parent->get_parent() == detail::kInvalidNodeId;
    const auto source_x = FormatSvgNumber(MapX(transform, source.get_x()));
    const auto source_y = FormatSvgNumber(MapY(transform, source.get_y()));
    const auto target_x = FormatSvgNumber(MapX(transform, target.get_x()));
    const auto target_y = FormatSvgNumber(MapY(transform, target.get_y()));
    output_stream << "<line x1=\"" << source_x << "\" y1=\"" << source_y << "\" x2=\"" << target_x << "\" y2=\"" << target_y
                  << "\" stroke=\"" << (is_root_spoke ? "#8a8f99" : "#666666") << "\" stroke-width=\""
                  << (is_root_spoke ? detail::kRootSpokeWidth : 1);
    if (is_root_spoke) {
      output_stream << R"(" stroke-dasharray="7,4" />;
)";
    } else {
      output_stream << R"(" />
)";
    }
  }
}

auto WriteTopologyNodes(std::ofstream& output_stream, const detail::SvgTransform& transform, const icts::Tree& tree) -> void
{
  for (std::size_t node_id = 0; node_id < tree.get_size(); ++node_id) {
    const auto* node = tree.get_node(node_id);
    if (node == nullptr) {
      continue;
    }
    const auto& position = node->get_position();
    if (position.get_x() < 0 || position.get_y() < 0) {
      continue;
    }
    const bool is_root = node->get_parent() == detail::kInvalidNodeId;
    const auto position_x = FormatSvgNumber(MapX(transform, position.get_x()));
    const auto position_y = FormatSvgNumber(MapY(transform, position.get_y()));
    output_stream << "<circle cx=\"" << position_x << "\" cy=\"" << position_y << "\" r=\""
                  << (is_root ? detail::kRootRadius : detail::kNodeRadius) << "\" fill=\"" << (is_root ? "#d62728" : "#444444") << R"(" />
)";
  }
}

auto WriteTopologyLoads(std::ofstream& output_stream, const detail::SvgTransform& transform, const std::vector<icts::Pin*>& loads) -> void
{
  for (const auto* pin : loads) {
    if (pin == nullptr) {
      continue;
    }
    const auto& location = pin->get_location();
    const auto location_x = FormatSvgNumber(MapX(transform, location.get_x()));
    const auto location_y = FormatSvgNumber(MapY(transform, location.get_y()));
    output_stream << "<circle cx=\"" << location_x << "\" cy=\"" << location_y << "\" r=\"" << detail::kLoadRadius
                  << R"(" fill="#1f77b4" fill-opacity="0.6" />
)";
  }
}

}  // namespace

auto WriteTopologySvg(const std::string& path, const icts::Tree& tree, const std::vector<icts::Pin*>& loads) -> bool
{
  if (tree.get_size() == 0) {
    return false;
  }
  const auto bounds = ComputeBounds(loads, tree);
  const auto transform = MakeTransform(bounds);

  std::ofstream output_stream(path);
  if (!output_stream.is_open()) {
    return false;
  }

  output_stream << detail::kSvgOpenTagPrefix << FormatSvgNumber(transform.width) << detail::kSvgHeightTag
                << FormatSvgNumber(transform.height) << detail::kSvgViewBoxPrefix << FormatSvgNumber(transform.width) << ' '
                << FormatSvgNumber(transform.height) << detail::kSvgOpenTagSuffix;
  output_stream << detail::kSvgBackgroundRect;

  WriteTopologyEdges(output_stream, transform, tree);
  WriteTopologyNodes(output_stream, transform, tree);
  WriteTopologyLoads(output_stream, transform, loads);

  output_stream << detail::kSvgClosingTag;
  return true;
}

}  // namespace icts::visualization::topology
