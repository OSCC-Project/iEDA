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
 * @file vec_wire_pattern.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-12-08
 * @brief wire pattern for vectorization
 */

#include "vec_wire_pattern.hh"

#include <algorithm>
#include <fstream>
#include <limits>
#include <numeric>

#include "log/Log.hh"
#include "vec_net_graph_gen.hh"
namespace ivec {
void VecWirePatternGenerator::genPatterns()
{
  auto gen = VecNetGraphGenerator();
  auto graphs = gen.buildGraphs();
  auto build_pattern = [&](auto& edge, auto& graph) {
    auto path = graph[edge].path;
    std::vector<Point> points;
    for (auto& [start, end] : path) {
      auto start_point = Point{bg::get<0>(start), bg::get<1>(start), bg::get<2>(start)};
      points.push_back(start_point);
    }
    auto end = path.back().second;
    points.push_back(Point{bg::get<0>(end), bg::get<1>(end), bg::get<2>(end)});
    auto pattern = calcPattern(points);
    if (_pattern_count.contains(pattern.name)) {
      _pattern_count[pattern.name] += 1;
    } else {
      _patterns[pattern.name] = pattern;
      _pattern_count[pattern.name] = 1;
    }
  };
  std::ranges::for_each(graphs, [&](auto& graph) {
    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
      build_pattern(e, graph);
    }
  });
}
void VecWirePatternGenerator::addPattern(VecNetWire& wire)
{
  auto points = getPointList(wire);
  auto pattern = calcPattern(points);
  if (_pattern_count.contains(pattern.name)) {
    _pattern_count[pattern.name] += 1;
  } else {
    _patterns[pattern.name] = pattern;
    _pattern_count[pattern.name] = 1;
  }
}

void VecWirePatternGenerator::patternSummary(const std::string& csv_path)
{
  std::ofstream csv_file(csv_path);
  csv_file << "Pattern,Count\n";
  for (auto& [pattern, count] : _pattern_count) {
    csv_file << pattern << "," << count << "\n";
  }
}

std::vector<Point> VecWirePatternGenerator::getPointList(VecNetWire& wire)
{
  auto& path = wire.get_paths();
  std::vector<Point> points;
  for (auto& [start, end] : path) {
    auto start_point = Point{start->get_x(), start->get_y(), start->get_layer_id()};
    points.push_back(start_point);
  }
  auto end = path.back().second;
  points.push_back(Point{end->get_x(), end->get_y(), end->get_layer_id()});
  return points;
}

VecWirePatternSequence VecWirePatternGenerator::calcPattern(const std::vector<Point>& points)
{
  // make front in the top-right corner, reverse if needed
  auto sorted_points = points;
  if (sorted_points.front().x > sorted_points.back().x) {
    std::ranges::reverse(sorted_points);
  } else if (sorted_points.front().x == sorted_points.back().x && sorted_points.front().y > sorted_points.back().y) {
    std::ranges::reverse(sorted_points);
  }
  // init pattern
  VecWirePatternSequence pattern;
  for (size_t i = 0; i < sorted_points.size() - 1; ++i) {
    auto& start = sorted_points[i];
    auto& end = sorted_points[i + 1];
    VecWirePatternUnit unit;
    bool x_same = start.x == end.x;
    bool y_same = start.y == end.y;
    bool z_same = start.z == end.z;
    if (x_same && y_same && z_same) {
      continue;
    }
    unit.direction = (x_same && y_same) ? kVIA : (x_same ? (start.y < end.y ? kTOP : kBOTTOM) : (start.x < end.x ? kRIGHT : kLEFT));
    if (unit.direction == kVIA) {
      unit.length = 1;
    } else {
      unit.length = std::abs(start.x - end.x) + std::abs(start.y - end.y);
    }
    pattern.units.push_back(unit);
  }

  if (pattern.units.empty()) {
    return pattern;
  }

  // post-process
  // 1. find the max common factor by gcd
  auto max_common_factor = pattern.units.front().length;
  std::ranges::for_each(pattern.units,
                        [&max_common_factor](const auto& unit) { max_common_factor = std::gcd(max_common_factor, unit.length); });
  // 2. normalize the pattern
  std::ranges::for_each(pattern.units, [&max_common_factor](auto& unit) { unit.length /= max_common_factor; });
  // 3. generate the pattern name
  std::string pattern_name;
  std::ranges::for_each(pattern.units, [&pattern_name](const auto& unit) {
    switch (unit.direction) {
      case kTOP:
        pattern_name += "T";
        break;
      case kBOTTOM:
        pattern_name += "B";
        break;
      case kLEFT:
        pattern_name += "L";
        break;
      case kRIGHT:
        pattern_name += "R";
        break;
      case kVIA:
        pattern_name += "V";
        break;
    }
    pattern_name += std::to_string(unit.length / 100 + 1);
  });
  pattern.name = pattern_name;
  return pattern;
}

}  // namespace ivec