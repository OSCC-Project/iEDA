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
 * @file TimingEngine.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-03-16
 * @brief Pure RCTree timing propagation engine for the timing module.
 */

#include "TimingEngine.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <ostream>
#include <ranges>
#include <vector>

#include "Log.hh"
#include "RCTree.hh"

namespace icts {
namespace {

constexpr double kPiRcElmoreFactor = 2.0;
constexpr double kLogNine = 9.0;

auto BuildPreOrder(const RCTree& rc_tree) -> std::vector<std::size_t>
{
  std::vector<std::size_t> order;
  if (rc_tree.vertex_count() == 0) {
    return order;
  }

  std::vector<std::size_t> stack{rc_tree.get_root()};
  while (!stack.empty()) {
    auto vertex_id = stack.back();
    stack.pop_back();
    order.push_back(vertex_id);

    const auto* vertex = rc_tree.get_vertex(vertex_id);
    LOG_FATAL_IF(vertex == nullptr) << "RCTree vertex is null during preorder traversal.";

    for (const auto child_arc_id : std::ranges::reverse_view(vertex->child_arc_ids)) {
      const auto* arc = rc_tree.get_arc(child_arc_id);
      LOG_FATAL_IF(arc == nullptr) << "RCTree arc is null during preorder traversal.";
      stack.push_back(arc->sink_vertex_id);
    }
  }

  return order;
}

}  // namespace

auto TimingEngine::update(RCTree& rc_tree) -> TimingEngine::Metrics
{
  LOG_FATAL_IF(!rc_tree.validate()) << "RCTree is invalid before timing update.";
  if (rc_tree.vertex_count() == 0) {
    return {};
  }

  rc_tree.clearTimingCache();
  updateDownstreamCap(rc_tree);
  updateIncreaseDelay(rc_tree);
  updateArrival(rc_tree);
  updateSlew(rc_tree);
  updateDownstreamDelay(rc_tree);
  return evaluate(rc_tree);
}

auto TimingEngine::updateDownstreamCap(RCTree& rc_tree) -> void
{
  const auto preorder = BuildPreOrder(rc_tree);
  for (const auto vertex_id : std::ranges::reverse_view(preorder)) {
    auto* vertex = rc_tree.get_vertex(vertex_id);
    LOG_FATAL_IF(vertex == nullptr) << "RCTree vertex is null during downstream capacitance update.";

    double downstream_cap = vertex->lumped_cap;
    for (auto arc_id : vertex->child_arc_ids) {
      const auto* arc = rc_tree.get_arc(arc_id);
      LOG_FATAL_IF(arc == nullptr) << "RCTree arc is null during downstream capacitance update.";

      const auto* child = rc_tree.get_vertex(arc->sink_vertex_id);
      LOG_FATAL_IF(child == nullptr) << "RCTree child vertex is null during downstream capacitance update.";
      downstream_cap += child->downstream_cap + arc->capacitance;
    }

    vertex->downstream_cap = downstream_cap;
  }
}

auto TimingEngine::updateIncreaseDelay(RCTree& rc_tree) -> void
{
  for (auto& arc : rc_tree.get_arcs()) {
    const auto* child = rc_tree.get_vertex(arc.sink_vertex_id);
    LOG_FATAL_IF(child == nullptr) << "RCTree child vertex is null during arc delay update.";
    arc.increase_delay = calcArcDelay(child->downstream_cap, arc.resistance, arc.capacitance);
  }
}

auto TimingEngine::updateArrival(RCTree& rc_tree) -> void
{
  auto preorder = BuildPreOrder(rc_tree);
  if (preorder.empty()) {
    return;
  }

  auto* root = rc_tree.get_vertex(preorder.front());
  LOG_FATAL_IF(root == nullptr) << "RCTree root vertex is null during arrival update.";
  root->arrival = 0.0;

  for (auto vertex_id : preorder) {
    const auto* vertex = rc_tree.get_vertex(vertex_id);
    LOG_FATAL_IF(vertex == nullptr) << "RCTree vertex is null during arrival update.";

    for (auto arc_id : vertex->child_arc_ids) {
      const auto* arc = rc_tree.get_arc(arc_id);
      LOG_FATAL_IF(arc == nullptr) << "RCTree arc is null during arrival update.";

      auto* child = rc_tree.get_vertex(arc->sink_vertex_id);
      LOG_FATAL_IF(child == nullptr) << "RCTree child vertex is null during arrival update.";
      child->arrival = vertex->arrival + arc->increase_delay;
    }
  }
}

auto TimingEngine::updateSlew(RCTree& rc_tree) -> void
{
  auto preorder = BuildPreOrder(rc_tree);
  if (preorder.empty()) {
    return;
  }

  auto* root = rc_tree.get_vertex(preorder.front());
  LOG_FATAL_IF(root == nullptr) << "RCTree root vertex is null during slew update.";
  root->slew = 0.0;

  for (auto vertex_id : preorder) {
    const auto* vertex = rc_tree.get_vertex(vertex_id);
    LOG_FATAL_IF(vertex == nullptr) << "RCTree vertex is null during slew update.";

    for (auto arc_id : vertex->child_arc_ids) {
      const auto* arc = rc_tree.get_arc(arc_id);
      LOG_FATAL_IF(arc == nullptr) << "RCTree arc is null during slew update.";

      auto* child = rc_tree.get_vertex(arc->sink_vertex_id);
      LOG_FATAL_IF(child == nullptr) << "RCTree child vertex is null during slew update.";
      auto ideal_slew = calcIdealSlew(arc->increase_delay);
      child->slew = std::sqrt((vertex->slew * vertex->slew) + (ideal_slew * ideal_slew));
    }
  }
}

auto TimingEngine::updateDownstreamDelay(RCTree& rc_tree) -> void
{
  const auto preorder = BuildPreOrder(rc_tree);
  for (const auto vertex_id : std::ranges::reverse_view(preorder)) {
    auto* vertex = rc_tree.get_vertex(vertex_id);
    LOG_FATAL_IF(vertex == nullptr) << "RCTree vertex is null during downstream delay update.";

    if (vertex->child_arc_ids.empty()) {
      vertex->min_downstream_delay = 0.0;
      vertex->max_downstream_delay = 0.0;
      continue;
    }

    double min_delay = std::numeric_limits<double>::max();
    double max_delay = std::numeric_limits<double>::lowest();
    for (auto arc_id : vertex->child_arc_ids) {
      const auto* arc = rc_tree.get_arc(arc_id);
      LOG_FATAL_IF(arc == nullptr) << "RCTree arc is null during downstream delay update.";

      const auto* child = rc_tree.get_vertex(arc->sink_vertex_id);
      LOG_FATAL_IF(child == nullptr) << "RCTree child vertex is null during downstream delay update.";
      min_delay = std::min(min_delay, arc->increase_delay + child->min_downstream_delay);
      max_delay = std::max(max_delay, arc->increase_delay + child->max_downstream_delay);
    }

    vertex->min_downstream_delay = min_delay == std::numeric_limits<double>::max() ? 0.0 : min_delay;
    vertex->max_downstream_delay = max_delay == std::numeric_limits<double>::lowest() ? 0.0 : max_delay;
  }
}

auto TimingEngine::evaluate(const RCTree& rc_tree) -> TimingEngine::Metrics
{
  if (rc_tree.vertex_count() == 0) {
    return {};
  }

  const auto* root = rc_tree.get_vertex(rc_tree.get_root());
  LOG_FATAL_IF(root == nullptr) << "RCTree root vertex is null during timing evaluation.";

  double max_slew = 0.0;
  for (const auto& tree_vertex : rc_tree.get_vertices()) {
    max_slew = std::max(max_slew, tree_vertex.slew);
  }

  return Metrics{.skew = root->max_downstream_delay - root->min_downstream_delay,
                 .min_delay = root->arrival + root->min_downstream_delay,
                 .max_delay = root->arrival + root->max_downstream_delay,
                 .max_slew = max_slew,
                 .total_cap = root->downstream_cap};
}

auto TimingEngine::calcSkew(const RCTree& rc_tree) -> double
{
  if (rc_tree.vertex_count() == 0) {
    return 0.0;
  }

  const auto* root = rc_tree.get_vertex(rc_tree.get_root());
  LOG_FATAL_IF(root == nullptr) << "RCTree root vertex is null during skew calculation.";
  return root->max_downstream_delay - root->min_downstream_delay;
}

auto TimingEngine::calcArcDelay(double downstream_cap, double resistance, double capacitance) -> double
{
  return resistance * ((capacitance / kPiRcElmoreFactor) + downstream_cap);
}

auto TimingEngine::calcIdealSlew(double arc_delay) -> double
{
  return std::log(kLogNine) * arc_delay;
}

}  // namespace icts
