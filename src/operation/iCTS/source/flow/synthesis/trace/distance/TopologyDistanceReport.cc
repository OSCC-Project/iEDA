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
 * @file TopologyDistanceReport.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief Implements Topology-specific structured report sections.
 */

#include "synthesis/trace/distance/TopologyDistanceReport.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Log.hh"
#include "Net.hh"
#include "Pin.hh"
#include "Point.hh"
#include "logger/Schema.hh"
#include "synthesis/topology/buffer/BufferInsertion.hh"

namespace icts::topology {
namespace {

auto calcManhattanDistance(const Point<int>& lhs, const Point<int>& rhs) -> int
{
  const int delta_x = std::abs(lhs.get_x() - rhs.get_x());
  const int delta_y = std::abs(lhs.get_y() - rhs.get_y());
  return delta_x + delta_y;
}

auto formatDecimal(double value) -> std::string
{
  std::ostringstream stream;
  stream.setf(std::ostringstream::fixed, std::ostringstream::floatfield);
  stream.precision(2);
  stream << value;
  return stream.str();
}

auto resolveDbuPerUm(std::int32_t dbu_per_um) -> std::optional<double>
{
  if (dbu_per_um <= 0) {
    LOG_WARNING << "Topology distance report skipped: DBU-per-micron is unavailable.";
    return std::nullopt;
  }
  return static_cast<double>(dbu_per_um);
}

auto dbuToUm(double value_dbu, double dbu_per_um) -> double
{
  return value_dbu / dbu_per_um;
}

auto resolveDistanceReportPath(const std::string& log_file, SchemaWriter& reporter) -> std::filesystem::path
{
  if (reporter.isOpen()) {
    return reporter.getActivePath();
  }

  return log_file.empty() ? std::filesystem::path{} : std::filesystem::path(log_file);
}

}  // namespace

auto EmitClusterLeafDistanceTables(const ClusterLeafDistanceReportInput& input) -> std::optional<Topology::ClusterLeafDistanceSummary>
{
  LOG_FATAL_IF(input.reporter == nullptr) << "Topology distance report requires reporter.";
  LOG_FATAL_IF(input.build == nullptr) << "Topology distance report requires topology build.";
  auto& reporter = *input.reporter;
  const auto& build = *input.build;
  if (build.output.cluster_buffers.empty()) {
    return std::nullopt;
  }

  const auto report_path = resolveDistanceReportPath(input.log_file, reporter);
  if (report_path.empty()) {
    return std::nullopt;
  }

  constexpr const char* run_title = "iCTS Report";
  constexpr const char* summary_title = "Cluster Center vs H-Tree Leaf Distance Overview";
  const auto dbu_per_um = resolveDbuPerUm(input.dbu_per_um);
  if (!dbu_per_um.has_value()) {
    const KeyValueFields summary_fields = {
        {"count", "0"},
        {"status", "dbu_per_micron_unavailable"},
    };
    SchemaWriter::appendStandaloneKeyValueTable(report_path, run_title, summary_title, summary_fields);
    return std::nullopt;
  }

  std::vector<double> distances;
  distances.reserve(build.output.cluster_buffers.size());

  double total_distance = 0.0;
  for (const auto& cluster_buffer : build.output.cluster_buffers) {
    if (!HasValidLocation(cluster_buffer.location) || cluster_buffer.input_pin == nullptr) {
      continue;
    }

    const auto* leaf_net = cluster_buffer.input_pin->get_net();
    if (leaf_net == nullptr || leaf_net->get_driver() == nullptr) {
      continue;
    }
    const auto leaf_location = FindRenderableLocation(leaf_net->get_driver());
    if (!HasValidLocation(leaf_location)) {
      continue;
    }

    const int distance_dbu = calcManhattanDistance(cluster_buffer.location, leaf_location);
    const double distance_um = dbuToUm(distance_dbu, *dbu_per_um);
    distances.push_back(distance_um);
    total_distance += distance_um;
  }

  const bool emit_to_active_writer = reporter.isOpen() && reporter.getActivePath() == report_path;
  if (distances.empty()) {
    const KeyValueFields summary_fields = {
        {"count", "0"},
        {"status", "no_renderable_htree_leaf_locations"},
    };
    if (emit_to_active_writer) {
      reporter.emitSection("### Cluster Distance Overview");
      reporter.emitKeyValueTable(summary_title, summary_fields);
    } else {
      SchemaWriter::appendStandaloneKeyValueTable(report_path, run_title, summary_title, summary_fields);
    }
    return std::nullopt;
  }

  std::ranges::sort(distances);
  const std::size_t median_index = distances.size() / 2U;
  const double median_distance
      = (distances.size() % 2U) == 0U
            ? (static_cast<double>(distances.at(median_index - 1U)) + static_cast<double>(distances.at(median_index))) / 2.0
            : static_cast<double>(distances.at(median_index));
  Topology::ClusterLeafDistanceSummary summary{
      .count = distances.size(),
      .min_distance_um = distances.front(),
      .max_distance_um = distances.back(),
      .mean_distance_um = total_distance / static_cast<double>(distances.size()),
      .median_distance_um = median_distance,
  };
  const KeyValueFields summary_fields = {
      {"count", std::to_string(summary.count)},
      {"min_distance", formatDecimal(summary.min_distance_um) + " um"},
      {"max_distance", formatDecimal(summary.max_distance_um) + " um"},
      {"mean_distance", formatDecimal(summary.mean_distance_um) + " um"},
      {"median_distance", formatDecimal(summary.median_distance_um) + " um"},
  };

  if (emit_to_active_writer) {
    reporter.emitSection("### Cluster Distance Overview");
    reporter.emitKeyValueTable(summary_title, summary_fields);
    return summary;
  }

  SchemaWriter::appendStandaloneKeyValueTable(report_path, run_title, summary_title, summary_fields);
  return summary;
}

}  // namespace icts::topology
