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
 * @file BinaryTopology.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief Bound-skew tree binary topology builder implementation (stage A).
 */

#include "bound_skew_tree/algorithm/BinaryTopology.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <numbers>
#include <numeric>
#include <optional>
#include <ostream>
#include <random>
#include <ranges>
#include <string_view>
#include <utility>
#include <vector>

#include "Log.hh"
#include "bound_skew_tree/algorithm/BoundSkewTreeImpl.hh"
#include "bound_skew_tree/component/Components.hh"

namespace icts::bst::detail {
namespace {

struct TreeBuildFrame
{
  std::vector<Area*> areas;
  std::optional<size_t> parent_index;
  size_t side = 0;
  bool expanded = false;
  Area* left_result = nullptr;
  Area* right_result = nullptr;
};

auto copyAreaRange(const std::vector<Area*>& areas, const size_t begin_index, const size_t end_index) -> std::vector<Area*>
{
  LOG_FATAL_IF(begin_index > end_index || end_index > areas.size()) << "Area range is invalid.";

  std::vector<Area*> area_range;
  area_range.reserve(end_index - begin_index);
  for (size_t area_index = begin_index; area_index < end_index; ++area_index) {
    area_range.push_back(areas.at(area_index));
  }
  return area_range;
}

auto assignTreeBuildResult(std::vector<TreeBuildFrame>& frames, const TreeBuildFrame& frame, Area* result, Area*& root) -> void
{
  if (frame.parent_index.has_value()) {
    auto& parent_frame = frames.at(*frame.parent_index);
    if (frame.side == kLeft) {
      parent_frame.left_result = result;
    } else {
      parent_frame.right_result = result;
    }
    return;
  }
  root = result;
}

template <typename SplitFunc, typename MergeFunc, typename CenterFunc>
auto buildBinaryTreeIteratively(const std::vector<Area*>& areas, const SplitFunc& split_func, const MergeFunc& merge_func,
                                const CenterFunc& center_func, std::string_view tree_name) -> Area*
{
  LOG_FATAL_IF(areas.empty()) << tree_name << " areas are empty.";

  std::vector<TreeBuildFrame> frames;
  frames.push_back(TreeBuildFrame{.areas = areas, .parent_index = std::nullopt});
  Area* root = nullptr;

  while (!frames.empty()) {
    const auto current_index = frames.size() - 1;
    auto& frame = frames.back();
    const auto area_count = frame.areas.size();

    if (area_count == 1) {
      assignTreeBuildResult(frames, frame, frame.areas.front(), root);
      frames.pop_back();
      continue;
    }

    if (!frame.expanded && area_count > 2) {
      auto [left_areas, right_areas] = split_func(frame.areas);
      frame.expanded = true;
      frames.push_back(TreeBuildFrame{.areas = right_areas, .parent_index = current_index, .side = kRight});
      frames.push_back(TreeBuildFrame{.areas = left_areas, .parent_index = current_index, .side = kLeft});
      continue;
    }

    Area* result = nullptr;
    if (area_count == 2) {
      result = merge_func(frame.areas.front(), frame.areas.back());
    } else {
      LOG_FATAL_IF(frame.left_result == nullptr || frame.right_result == nullptr) << tree_name << " child result is null.";
      result = merge_func(frame.left_result, frame.right_result);
    }
    result->set_location(center_func(frame.areas));
    assignTreeBuildResult(frames, frame, result, root);
    frames.pop_back();
  }

  LOG_FATAL_IF(root == nullptr) << tree_name << " root is null.";
  return root;
}

auto chooseInitialCenter(const std::vector<Area*>& areas, std::vector<Point>& center_points, std::mt19937& generator) -> void
{
  std::uniform_int_distribution<size_t> distribution(0, areas.size() - 1);
  center_points.push_back(areas.at(distribution(generator))->get_location());
}

auto calcSquaredDistancesToCenters(const std::vector<Area*>& areas, const std::vector<Point>& center_points) -> std::vector<double>
{
  std::vector<double> squared_distances(areas.size(), std::numeric_limits<double>::max());
  for (size_t area_index = 0; area_index < areas.size(); ++area_index) {
    double min_distance = std::numeric_limits<double>::max();
    for (const auto& center_point : center_points) {
      min_distance = std::min(min_distance, Geom::distance(areas.at(area_index)->get_location(), center_point));
    }
    squared_distances.at(area_index) = min_distance * min_distance;
  }
  return squared_distances;
}

auto expandCentersByKMeansPlus(const std::vector<Area*>& areas, const size_t cluster_count, std::vector<Point>& center_points,
                               std::mt19937& generator) -> void
{
  while (center_points.size() < cluster_count) {
    auto squared_distances = calcSquaredDistancesToCenters(areas, center_points);
    std::discrete_distribution<> distribution(squared_distances.begin(), squared_distances.end());
    const auto selected_index = static_cast<std::size_t>(distribution(generator));
    center_points.push_back(areas.at(selected_index)->get_location());
  }
}

auto assignAreasToCenters(const std::vector<Area*>& areas, const std::vector<Point>& center_points, std::vector<size_t>& center_assignments)
    -> void
{
  for (size_t area_index = 0; area_index < areas.size(); ++area_index) {
    double min_distance = std::numeric_limits<double>::max();
    size_t nearest_center_index = 0;
    for (size_t center_index = 0; center_index < center_points.size(); ++center_index) {
      const auto distance = Geom::distance(areas.at(area_index)->get_location(), center_points.at(center_index));
      if (distance < min_distance) {
        min_distance = distance;
        nearest_center_index = center_index;
      }
    }
    center_assignments.at(area_index) = nearest_center_index;
  }
}

auto calcUpdatedCenters(const std::vector<Area*>& areas, const std::vector<size_t>& center_assignments, const size_t cluster_count)
    -> std::vector<Point>
{
  std::vector<Point> center_points(cluster_count, Point(0, 0));
  std::vector<size_t> center_counts(cluster_count, 0);
  for (size_t area_index = 0; area_index < areas.size(); ++area_index) {
    const auto center_index = center_assignments.at(area_index);
    center_points.at(center_index) += areas.at(area_index)->get_location();
    ++center_counts.at(center_index);
  }
  for (size_t center_index = 0; center_index < cluster_count; ++center_index) {
    if (center_counts.at(center_index) > 0) {
      center_points.at(center_index) /= static_cast<double>(center_counts.at(center_index));
    }
  }
  return center_points;
}

auto calcWithinClusterDistance(const std::vector<Area*>& areas, const std::vector<Point>& center_points,
                               const std::vector<size_t>& center_assignments) -> double
{
  double total_distance = 0.0;
  for (size_t area_index = 0; area_index < areas.size(); ++area_index) {
    total_distance += Geom::distance(areas.at(area_index)->get_location(), center_points.at(center_assignments.at(area_index)));
  }
  return total_distance;
}

auto collectClusters(const std::vector<Area*>& areas, const std::vector<size_t>& center_assignments, const size_t cluster_count)
    -> std::vector<std::vector<Area*>>
{
  std::vector<std::vector<Area*>> clusters(cluster_count);
  for (size_t area_index = 0; area_index < areas.size(); ++area_index) {
    clusters.at(center_assignments.at(area_index)).push_back(areas.at(area_index));
  }
  auto [remove_begin, remove_end]
      = std::ranges::remove_if(clusters, [](const std::vector<Area*>& cluster) -> bool { return cluster.empty(); });
  clusters.erase(remove_begin, remove_end);
  return clusters;
}

}  // namespace

auto BinaryTopology::biPartition() -> void
{
  LOG_FATAL_IF(_impl._unmerged_nodes.size() < 2) << "unmerged nodes size is less than 2";
  _impl._root = buildBiPartitionTree(_impl._unmerged_nodes);
  _impl.areaReset();
}

auto BinaryTopology::buildBiPartitionTree(const std::vector<Area*>& areas) -> Area*
{
  return buildBinaryTreeIteratively(
      areas,
      [&](std::vector<Area*>& split_areas) -> std::pair<std::vector<Area*>, std::vector<Area*>> { return octagonDivide(split_areas); },
      [&](Area* left_area, Area* right_area) -> Area* { return _impl.merge(left_area, right_area); },
      [&](const std::vector<Area*>& center_areas) -> Point { return calcAreasCenter(center_areas); }, "Bi-partition");
}

auto BinaryTopology::octagonDivide(std::vector<Area*>& areas) -> std::pair<std::vector<Area*>, std::vector<Area*>>
{
  auto cap_sum
      = std::accumulate(areas.begin(), areas.end(), 0.0, [](double sum, Area* area) -> double { return sum + area->get_cap_load(); });
  auto half_cap = 1.0 * cap_sum / 2;

  auto octagon = calcOctagon(areas);
  auto bound_areas = areaOnOctagonBound(areas, octagon);
  const auto bound_area_count = bound_areas.size();
  const auto half_bound_area_count = bound_area_count / 2;

  const auto initial_bound_areas = bound_areas;
  for (size_t bound_area_index = 0; bound_area_index < half_bound_area_count; ++bound_area_index) {
    bound_areas.push_back(initial_bound_areas.at(bound_area_index));
  }

  auto calc_diameter = [](Area* area, const std::vector<Area*>& refs) -> double {
    auto min_dist = std::numeric_limits<double>::max();
    auto max_dist = std::numeric_limits<double>::lowest();
    std::ranges::for_each(refs, [&area, &min_dist, &max_dist](const Area* ref) -> void {
      auto dist = Geom::distance(area->get_location(), ref->get_location());
      min_dist = std::min(min_dist, dist);
      max_dist = std::max(max_dist, dist);
    });
    return max_dist + min_dist;
  };

  auto bound_diameter = [&](const std::vector<Area*>& ref) -> double {
    auto oct = calcOctagon(ref);
    auto bound = areaOnOctagonBound(ref, oct);
    double max_dist = std::numeric_limits<double>::lowest();
    for (auto left_iter = bound.begin(); left_iter != bound.end(); ++left_iter) {
      for (auto right_iter = std::next(left_iter); right_iter != bound.end(); ++right_iter) {
        max_dist = std::max(max_dist, Geom::distance((*left_iter)->get_location(), (*right_iter)->get_location()));
      }
    }
    return max_dist;
  };

  std::vector<Area*> left_set;
  std::vector<Area*> right_set;
  auto min_cost = std::numeric_limits<double>::max();

  for (size_t window_begin_index = 0; window_begin_index < bound_area_count; ++window_begin_index) {
    auto ref_set = copyAreaRange(bound_areas, window_begin_index, window_begin_index + half_bound_area_count);
    std::ranges::for_each(areas, [&ref_set, &calc_diameter](Area* area) -> void {
      auto point = area->get_location();
      point.val = calc_diameter(area, ref_set);
      area->set_location(point);
    });
    std::ranges::sort(areas, [](Area* left, Area* right) -> bool { return left->get_location().val < right->get_location().val; });

    size_t left_area_count = 0;
    double accumulated_capacitance = 0.0;
    double min_cap_difference = std::numeric_limits<double>::max();
    for (size_t area_index = 0; area_index + 1 < areas.size(); ++area_index) {
      accumulated_capacitance += areas.at(area_index)->get_cap_load();
      const auto current_difference = std::abs(accumulated_capacitance - half_cap);
      if (current_difference < min_cap_difference) {
        min_cap_difference = current_difference;
        left_area_count = area_index + 1;
      }
    }
    auto left = copyAreaRange(areas, 0, left_area_count);
    auto right = copyAreaRange(areas, left_area_count, areas.size());
    auto cost = bound_diameter(left) + bound_diameter(right);
    if (cost < min_cost) {
      min_cost = cost;
      left_set = left;
      right_set = right;
    }
  }
  return {left_set, right_set};
}

auto BinaryTopology::calcOctagon(const std::vector<Area*>& areas) -> std::vector<Point>
{
  auto x_p = std::numeric_limits<double>::lowest();
  auto y_p = std::numeric_limits<double>::lowest();
  auto ymx_p = std::numeric_limits<double>::lowest();
  auto ypx_p = std::numeric_limits<double>::lowest();
  auto x_m = std::numeric_limits<double>::max();
  auto y_m = std::numeric_limits<double>::max();
  auto ymx_m = std::numeric_limits<double>::max();
  auto ypx_m = std::numeric_limits<double>::max();
  std::ranges::for_each(areas, [&](const Area* area) -> void {
    const auto location = area->get_location();
    const auto x_coord = location.x;
    const auto y_coord = location.y;
    x_p = std::max(x_coord, x_p);
    x_m = std::min(x_coord, x_m);
    y_p = std::max(y_coord, y_p);
    y_m = std::min(y_coord, y_m);
    ymx_p = std::max(y_coord - x_coord, ymx_p);
    ymx_m = std::min(y_coord - x_coord, ymx_m);
    ypx_p = std::max(y_coord + x_coord, ypx_p);
    ypx_m = std::min(y_coord + x_coord, ypx_m);
  });

  std::vector<Point> octagon{Point(y_p - ymx_p, y_p), Point(ypx_p - y_p, y_p), Point(x_p, ypx_p - x_p), Point(x_p, x_p + ymx_m),
                             Point(y_m - ymx_m, y_m), Point(ypx_m - y_m, y_m), Point(x_m, ypx_m - x_m), Point(x_m, x_m + ymx_p)};
  Geom::convexHull(octagon);
  return octagon;
}

auto BinaryTopology::areaOnOctagonBound(const std::vector<Area*>& areas, const std::vector<Point>& octagon) -> std::vector<Area*>
{
  std::vector<Area*> result;
  std::ranges::for_each(areas, [&result, &octagon](Area* area) -> void {
    for (size_t i = 0; i < octagon.size(); ++i) {
      auto line = Side<Point>{octagon.at(i), octagon.at((i + 1) % octagon.size())};
      auto point = area->get_location();
      if (Geom::onLine(point, line)) {
        result.push_back(area);
        break;
      }
    }
  });
  auto center = Geom::centerPoint(octagon);
  std::ranges::for_each(areas, [&center](Area* area) -> void {
    auto point = area->get_location();
    auto arc_tan2 = std::atan2(point.y - center.y, point.x - center.x);
    if (arc_tan2 < 0) {
      arc_tan2 += 2 * std::numbers::pi;
    }
    point.val = arc_tan2;
    area->set_location(point);
  });
  std::ranges::sort(result, [](Area* left, Area* right) -> bool { return left->get_location().val < right->get_location().val; });
  return result;
}

auto BinaryTopology::biCluster() -> void
{
  LOG_FATAL_IF(_impl._unmerged_nodes.size() < 2) << "unmerged nodes size is less than 2";
  _impl._root = buildBiClusterTree(_impl._unmerged_nodes);
  _impl.areaReset();
}

auto BinaryTopology::buildBiClusterTree(const std::vector<Area*>& areas) -> Area*
{
  return buildBinaryTreeIteratively(
      areas,
      [&](const std::vector<Area*>& split_areas) -> std::pair<std::vector<Area*>, std::vector<Area*>> {
        auto clusters = kMeansPlus(split_areas, KMeansConfig{.cluster_count = 2});
        LOG_FATAL_IF(clusters.size() != 2) << "Bi-cluster requires exactly two non-empty clusters.";
        return std::pair<std::vector<Area*>, std::vector<Area*>>{clusters.front(), clusters.back()};
      },
      [&](Area* left_area, Area* right_area) -> Area* { return _impl.merge(left_area, right_area); },
      [&](const std::vector<Area*>& center_areas) -> Point { return calcAreasCenter(center_areas); }, "Bi-cluster");
}

auto BinaryTopology::calcAreasCenter(const std::vector<Area*>& areas) -> Point
{
  std::vector<Point> points;
  points.reserve(areas.size());
  std::ranges::for_each(areas, [&](Area* area) -> void { points.push_back(area->get_location()); });
  return Geom::centerPoint(points);
}

auto BinaryTopology::kMeansPlus(const std::vector<Area*>& areas, const KMeansConfig& config) -> std::vector<std::vector<Area*>>
{
  std::vector<std::vector<Area*>> best_clusters(config.cluster_count);
  std::vector<Point> centers;
  std::vector<size_t> assignments(areas.size(), 0);
  std::mt19937 generator(config.seed);

  chooseInitialCenter(areas, centers, generator);
  expandCentersByKMeansPlus(areas, config.cluster_count, centers, generator);

  double min_total_distance = std::numeric_limits<double>::max();
  for (size_t iteration = 0; iteration < config.max_iter; ++iteration) {
    assignAreasToCenters(areas, centers, assignments);
    auto new_centers = calcUpdatedCenters(areas, assignments, config.cluster_count);
    auto total_distance = calcWithinClusterDistance(areas, new_centers, assignments);
    if (total_distance < min_total_distance) {
      min_total_distance = total_distance;
      best_clusters = collectClusters(areas, assignments, config.cluster_count);
    }
    centers = std::move(new_centers);
  }

  return best_clusters;
}

}  // namespace icts::bst::detail
