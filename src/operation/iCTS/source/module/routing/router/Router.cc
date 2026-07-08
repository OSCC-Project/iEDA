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
 * @file Router.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-03-08
 * @brief Unified routing dispatch facade implementation
 */

#include "Router.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "ClockRouteSegmentRc.hh"
#include "Inst.hh"
#include "Log.hh"
#include "Net.hh"
#include "Pin.hh"
#include "PinLocationHelper.hh"
#include "Point.hh"
#include "RoutingTerminal.hh"
#include "SteinerTree.hh"
#include "bound_skew_tree/BSTRouter.hh"
#include "concurrent_bst_salt/CBSRouter.hh"
#include "flute/FLUTERouter.hh"
#include "geometry/Geometry.hh"
#include "module/routing/local_legalization/LocalLegalization.hh"
#include "salt/SALTRouter.hh"

namespace icts {
namespace {

auto ValidateClockRouteSegmentRc(const ClockRouteSegmentRc& route_segment_rc) -> void
{
  LOG_FATAL_IF(route_segment_rc.dbu_per_um <= 0) << "Router: DBU-per-micron is unavailable for RC-tree construction.";
  LOG_FATAL_IF(!std::isfinite(route_segment_rc.resistance_per_um_ohm) || route_segment_rc.resistance_per_um_ohm <= 0.0)
      << "Router: clock route segment resistance must be positive for RC-tree construction.";
  LOG_FATAL_IF(!std::isfinite(route_segment_rc.capacitance_per_um_pf) || route_segment_rc.capacitance_per_um_pf <= 0.0)
      << "Router: clock route segment capacitance must be positive for RC-tree construction.";
}

auto QueryArcParasitics(int route_distance_dbu, const ClockRouteSegmentRc& route_segment_rc) -> std::pair<double, double>
{
  const auto route_distance_um = static_cast<double>(std::max(route_distance_dbu, 0)) / static_cast<double>(route_segment_rc.dbu_per_um);
  if (route_distance_um <= 0.0) {
    return {0.0, 0.0};
  }

  const auto resistance = route_distance_um * route_segment_rc.resistance_per_um_ohm;
  const auto capacitance = route_distance_um * route_segment_rc.capacitance_per_um_pf;
  return {resistance, capacitance};
}

auto ResolveVertexName(const Router::ClockSteinerTreeType::NodeType& node, const Router::RCTreeType& rc_tree) -> std::string
{
  if (!node.name.empty() && !rc_tree.hasVertex(node.name)) {
    return node.name;
  }

  const auto base_name = node.is_terminal ? std::string("terminal_") : std::string("steiner_");
  auto name = base_name + std::to_string(node.id);
  std::size_t suffix = 0;
  while (rc_tree.hasVertex(name)) {
    ++suffix;
    name = base_name + std::to_string(node.id) + "_" + std::to_string(suffix);
  }
  return name;
}

auto GetWireDistance(const Router::ClockSteinerTreeType::EdgeType& edge) -> int
{
  return std::max(edge.distance, edge.routed_distance);
}

auto WriteBackPinLocations(const std::vector<Pin*>& pins, const std::vector<Point<int>>& points) -> void
{
  const auto count = std::min(pins.size(), points.size());
  for (std::size_t i = 0; i < count; ++i) {
    auto* pin = pins.at(i);
    if (pin == nullptr) {
      continue;
    }
    pin->set_location(points.at(i));
    auto* inst = pin->get_inst();
    if (inst != nullptr) {
      inst->set_location(points.at(i));
    }
  }
}

auto MakeTerminal(Pin* pin) -> Router::ClockTerminal
{
  Router::ClockTerminal terminal;
  if (pin == nullptr) {
    return terminal;
  }
  auto* inst = pin->get_inst();
  terminal.name = inst != nullptr ? (inst->get_name() + "/" + pin->get_name()) : pin->get_name();
  terminal.location = pin->get_location();
  terminal.pin_cap = 0.0;
  terminal.insertion_delay = 0.0;
  return terminal;
}

auto HasOverlappingTerminalLocation(const Router::ClockTerminal& driver_terminal, const std::vector<Router::ClockTerminal>& load_terminals)
    -> bool
{
  std::vector<Point<int>> terminal_locations;
  terminal_locations.reserve(load_terminals.size() + 1U);
  terminal_locations.push_back(driver_terminal.location);
  for (const auto& load_terminal : load_terminals) {
    terminal_locations.push_back(load_terminal.location);
  }

  std::ranges::sort(terminal_locations, [](const Point<int>& lhs, const Point<int>& rhs) -> bool {
    if (lhs.get_x() != rhs.get_x()) {
      return lhs.get_x() < rhs.get_x();
    }
    return lhs.get_y() < rhs.get_y();
  });

  for (std::size_t i = 1U; i < terminal_locations.size(); ++i) {
    if (terminal_locations.at(i - 1U) == terminal_locations.at(i)) {
      return true;
    }
  }
  return false;
}

auto LegalizeFluteLoadTerminals(const Router::ClockTerminal& driver_terminal, const std::vector<Router::ClockTerminal>& load_terminals)
    -> std::vector<Router::ClockTerminal>
{
  if (!HasOverlappingTerminalLocation(driver_terminal, load_terminals)) {
    return load_terminals;
  }

  std::vector<Router::ClockTerminal> legalized_terminals = load_terminals;
  std::vector<Point<int>> movable_points;
  movable_points.reserve(legalized_terminals.size());
  for (const auto& load_terminal : legalized_terminals) {
    movable_points.push_back(load_terminal.location);
  }

  LocalLegalization::Config legalization_config;
  legalization_config.failure_policy = LocalLegalization::FailurePolicy::kKeepOriginal;
  auto result = LocalLegalization::legalize(movable_points, std::vector<Point<int>>{driver_terminal.location},
                                            LocalLegalization::RegionType{}, LocalLegalization::RegionType{}, legalization_config);
  if (result.legalized_points.size() != legalized_terminals.size()) {
    LOG_WARNING << "Router: terminal legalization before FLUTE returned an unexpected point count; continuing with original locations.";
    return load_terminals;
  }

  for (std::size_t i = 0; i < legalized_terminals.size(); ++i) {
    legalized_terminals.at(i).location = result.legalized_points.at(i);
  }

  if (!result.success || HasOverlappingTerminalLocation(driver_terminal, legalized_terminals)) {
    LOG_WARNING << "Router: terminal legalization before FLUTE did not fully resolve overlaps; continuing with original locations.";
    return load_terminals;
  }

  return legalized_terminals;
}

auto BuildClockStarTree(const Router::ClockTerminal& driver_terminal, const std::vector<Router::ClockTerminal>& load_terminals)
    -> Router::ClockSteinerTreeType
{
  Router::ClockSteinerTreeType route_tree;
  const auto root_id
      = route_tree.addNode(driver_terminal.name, driver_terminal.location, true, driver_terminal.pin_cap, driver_terminal.insertion_delay);
  if (root_id == Router::ClockSteinerTreeType::kInvalidId) {
    return {};
  }
  route_tree.setRoot(root_id);

  for (const auto& load_terminal : load_terminals) {
    const auto load_id
        = route_tree.addNode(load_terminal.name, load_terminal.location, true, load_terminal.pin_cap, load_terminal.insertion_delay);
    if (load_id == Router::ClockSteinerTreeType::kInvalidId) {
      return {};
    }

    const auto distance = geometry::Manhattan(driver_terminal.location, load_terminal.location);
    const auto edge_id = route_tree.addEdge(root_id, load_id, distance, distance);
    if (edge_id == Router::ClockSteinerTreeType::kInvalidId) {
      return {};
    }
  }

  return route_tree.validate() ? route_tree : Router::ClockSteinerTreeType{};
}

auto BuildClockRCTree(const Router::ClockSteinerTreeType& tree, const ClockRouteSegmentRc& route_segment_rc) -> Router::RCTreeType
{
  Router::RCTreeType rc_tree;
  if (tree.node_count() == 0) {
    return rc_tree;
  }

  ValidateClockRouteSegmentRc(route_segment_rc);
  LOG_FATAL_IF(!tree.validate()) << "Routing tree is invalid before RCTree conversion.";

  rc_tree.reserveVertices(tree.node_count());
  rc_tree.reserveArcs(tree.edge_count());

  std::vector<std::size_t> node_to_vertex_id(tree.node_count(), Router::RCTreeType::kInvalidId);
  for (const auto& node : tree.get_nodes()) {
    auto vertex_name = ResolveVertexName(node, rc_tree);
    auto vertex_id = rc_tree.addVertex(vertex_name, node.is_terminal, node.pin_cap);
    LOG_FATAL_IF(vertex_id == Router::RCTreeType::kInvalidId) << "Failed to add RCTree vertex for routing node: " << vertex_name;
    node_to_vertex_id.at(node.id) = vertex_id;
  }

  rc_tree.setRoot(node_to_vertex_id.at(tree.get_root()));

  for (const auto& edge : tree.get_edges()) {
    auto source_vertex_id = node_to_vertex_id.at(edge.source_node_id);
    auto sink_vertex_id = node_to_vertex_id.at(edge.target_node_id);
    LOG_FATAL_IF(source_vertex_id == Router::RCTreeType::kInvalidId || sink_vertex_id == Router::RCTreeType::kInvalidId)
        << "Routing edge endpoint is missing during RCTree conversion.";

    const auto [arc_resistance, arc_capacitance] = QueryArcParasitics(GetWireDistance(edge), route_segment_rc);
    auto arc_id = rc_tree.addArc(source_vertex_id, sink_vertex_id, arc_resistance, arc_capacitance);
    LOG_FATAL_IF(arc_id == Router::RCTreeType::kInvalidId) << "Failed to add RCTree arc when converting routing tree edge " << edge.id;
  }

  LOG_FATAL_IF(!rc_tree.validate()) << "Constructed RCTree is invalid after routing conversion.";
  return rc_tree;
}

}  // namespace

auto Router::buildFluteTree(const ClockTerminal& driver_terminal, const std::vector<ClockTerminal>& load_terminals)
    -> Router::ClockSteinerTreeType
{
  return FLUTERouter::buildTree(driver_terminal, LegalizeFluteLoadTerminals(driver_terminal, load_terminals));
}

auto Router::buildSaltTree(const ClockTerminal& driver_terminal, const std::vector<ClockTerminal>& load_terminals)
    -> Router::ClockSteinerTreeType
{
  return SALTRouter::buildTree(driver_terminal, load_terminals);
}

auto Router::buildBstTree(const std::vector<ClockTerminal>& load_terminals, const BSTRoutingConfig& parameters)
    -> Router::ClockSteinerTreeType
{
  return BSTRouter::buildTree(load_terminals, parameters);
}

auto Router::buildCbsTree(const std::vector<ClockTerminal>& load_terminals, const BSTRoutingConfig& parameters)
    -> Router::ClockSteinerTreeType
{
  return CBSRouter::buildTree(load_terminals, parameters);
}

auto Router::buildClockNetTree(const Net& net) -> Router::ClockSteinerTreeType
{
  if (net.get_driver() == nullptr || net.get_loads().empty()) {
    return {};
  }

  std::vector<Router::ClockTerminal> load_terminals;
  load_terminals.reserve(net.get_loads().size());
  for (auto* load : net.get_loads()) {
    if (load == nullptr) {
      continue;
    }
    load_terminals.push_back(MakeTerminal(load));
  }
  if (load_terminals.empty()) {
    return {};
  }

  const auto driver_terminal = MakeTerminal(net.get_driver());
  if (load_terminals.size() == 1U) {
    return BuildClockStarTree(driver_terminal, load_terminals);
  }

  return buildFluteTree(driver_terminal, load_terminals);
}

auto Router::legalizePins(std::vector<Pin*>& movable_pins, const std::vector<Pin*>& fixed_pins, const LegalizationRegion& feasible_region,
                          const LegalizationRegion& block_region) -> Router::LegalizationOutput
{
  return legalizePins(movable_pins, fixed_pins, feasible_region, block_region, LegalizationConfig{});
}

auto Router::legalizePins(std::vector<Pin*>& movable_pins, const std::vector<Pin*>& fixed_pins, const LegalizationRegion& feasible_region,
                          const LegalizationRegion& block_region, const LegalizationConfig& config) -> Router::LegalizationOutput
{
  auto movable_points = CollectPinLocations(movable_pins);
  const auto fixed_points = CollectPinLocations(fixed_pins);
  auto result = LocalLegalization::legalize(movable_points, fixed_points, feasible_region, block_region, config);
  if (!result.success) {
    LOG_WARNING << "Router::legalizePins did not produce a successful legalization result.";
  }
  if (result.success || config.failure_policy == LocalLegalization::FailurePolicy::kKeepOriginal) {
    WriteBackPinLocations(movable_pins, result.legalized_points);
  }
  return result;
}

auto Router::buildRCTree(const ClockSteinerTreeType& clock_tree, const ClockRouteSegmentRc& route_segment_rc) -> Router::RCTreeType
{
  return BuildClockRCTree(clock_tree, route_segment_rc);
}

}  // namespace icts
