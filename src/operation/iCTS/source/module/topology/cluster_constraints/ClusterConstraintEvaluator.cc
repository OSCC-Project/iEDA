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
 * @file ClusterConstraintEvaluator.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Shared cluster legality and electrical evaluator.
 */

#include "ClusterConstraintEvaluator.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ClockRouteSegmentRc.hh"
#include "Log.hh"
#include "Pin.hh"
#include "PinLocationHelper.hh"
#include "Point.hh"
#include "RCTree.hh"
#include "SteinerTree.hh"
#include "TimingEngine.hh"
#include "TopologyConfig.hh"
#include "bound_skew_tree/BSTRouter.hh"
#include "local_legalization/LocalLegalization.hh"
#include "router/Router.hh"

namespace icts {
namespace {

auto BuildBstParameters(const ClusterConfig& config, const Point<int>& routing_root) -> BSTRoutingConfig
{
  BSTRoutingConfig parameters;
  parameters.dbu_per_um = config.clock_route_segment_rc.dbu_per_um;
  LOG_FATAL_IF(parameters.dbu_per_um <= 0) << "ClusterConstraintEvaluator: DBU-per-micron is unavailable.";
  parameters.skew_bound = 0.0;
  parameters.rc_pattern = BSTRoutingRCPattern::kHV;
  parameters.topology_mode = BSTRoutingTopologyMode::kGreedyDistance;
  parameters.root_guide = routing_root;

  LOG_FATAL_IF(config.clock_route_segment_rc.capacitance_per_um_pf <= 0.0)
      << "ClusterConstraintEvaluator: clock route segment capacitance is unavailable.";
  LOG_FATAL_IF(config.clock_route_segment_rc.resistance_per_um_ohm <= 0.0)
      << "ClusterConstraintEvaluator: clock route segment resistance is unavailable.";
  parameters.unit_h_cap = config.clock_route_segment_rc.capacitance_per_um_pf;
  parameters.unit_v_cap = parameters.unit_h_cap;
  parameters.unit_h_res = config.clock_route_segment_rc.resistance_per_um_ohm;
  parameters.unit_v_res = parameters.unit_h_res;
  return parameters;
}

auto IsPointOverlappingAnyLoad(const Point<int>& point, const std::vector<Point<int>>& load_locations) -> bool
{
  return std::ranges::any_of(load_locations, [&](const auto& location) -> bool { return location == point; });
}

auto LegalizeRoutingRoot(const Point<int>& raw_synthetic_root, const std::vector<Point<int>>& load_locations, Point<int>& legalized_root)
    -> bool
{
  legalized_root = raw_synthetic_root;
  if (!IsPointOverlappingAnyLoad(raw_synthetic_root, load_locations)) {
    return true;
  }

  std::vector<Point<int>> movable_points{raw_synthetic_root};
  LocalLegalization::Config legalization_config;
  legalization_config.failure_policy = LocalLegalization::FailurePolicy::kKeepOriginal;

  const auto result = LocalLegalization::legalize(movable_points, load_locations, LocalLegalization::RegionType{},
                                                  LocalLegalization::RegionType{}, legalization_config);

  if (result.legalized_points.empty()) {
    LOG_WARNING << "Cluster constraint exact-cap root legalization failed: legalization returned empty points, synthetic root "
                << raw_synthetic_root << ".";
    return false;
  }

  legalized_root = result.legalized_points.front();
  if (IsPointOverlappingAnyLoad(legalized_root, load_locations)) {
    LOG_WARNING << "Cluster constraint exact-cap root legalization failed: legalized root still overlaps load location, synthetic root "
                << raw_synthetic_root << ", legalized root " << legalized_root << ".";
    return false;
  }

  return true;
}

auto CalcTreeWirelength(const Router::ClockSteinerTreeType& tree) -> double
{
  double total_wirelength = 0.0;
  for (const auto& edge : tree.get_edges()) {
    total_wirelength += static_cast<double>(std::max(edge.distance, edge.routed_distance));
  }
  return total_wirelength;
}

auto UpdateEstimateFromRcTree(ElectricalEstimate& estimate, RCTree& rc_tree) -> bool
{
  if (!rc_tree.validate()) {
    LOG_WARNING << "Cluster constraint electrical estimate got invalid RCTree.";
    return false;
  }
  const auto metrics = TimingEngine::update(rc_tree);
  estimate.total_cap = std::max(metrics.total_cap, estimate.pin_cap);
  estimate.wire_cap = std::max(0.0, estimate.total_cap - estimate.pin_cap);
  estimate.route_success = true;
  return true;
}

}  // namespace

auto ClusterConstraintEvaluator::evaluateLoads(const std::vector<Pin*>& loads, const Point<int>& routing_root, const ClusterConfig& config,
                                               bool need_exact_cap) -> ConstraintEvaluation
{
  std::vector<Pin*> active_loads;
  active_loads.reserve(loads.size());
  for (auto* pin : loads) {
    if (pin != nullptr) {
      active_loads.push_back(pin);
    }
  }

  if (active_loads.empty()) {
    return ConstraintEvaluation{
        .legal = false,
        .violation = ConstraintViolation::kEmptyCluster,
        .metrics = {},
    };
  }

  int min_x = active_loads.front()->get_location().get_x();
  int min_y = active_loads.front()->get_location().get_y();
  int max_x = min_x;
  int max_y = min_y;
  for (const auto* pin : active_loads) {
    const auto& location = pin->get_location();
    min_x = std::min(min_x, location.get_x());
    min_y = std::min(min_y, location.get_y());
    max_x = std::max(max_x, location.get_x());
    max_y = std::max(max_y, location.get_y());
  }

  const int diameter = (max_x - min_x) + (max_y - min_y);
  return evaluatePinnedLoads(active_loads, active_loads.size(), diameter, routing_root, config, need_exact_cap);
}

auto ClusterConstraintEvaluator::evaluatePinnedLoads(const std::vector<Pin*>& loads, std::size_t fanout, int diameter,
                                                     const Point<int>& routing_root, const ClusterConfig& config, bool need_exact_cap)
    -> ConstraintEvaluation
{
  ConstraintEvaluation evaluation;
  evaluation.legal = false;
  evaluation.violation = ConstraintViolation::kEmptyCluster;
  evaluation.metrics.fanout = fanout;
  if (fanout == 0U) {
    return evaluation;
  }

  evaluation.metrics.diameter = diameter;
  evaluation.metrics.electrical.synthetic_root = routing_root;
  evaluation.metrics.electrical.legalized_root = routing_root;
  evaluation.metrics.electrical.routed_root = routing_root;

  const auto has_cap_limit = IsFiniteCapLimit(config.max_cap);
  const auto need_exact_eval = need_exact_cap && (has_cap_limit || config.always_build_exact_cap);

  if (has_cap_limit || need_exact_eval) {
    evaluation.metrics.cap_lower_bound = estimatePinCap(loads, config);
    evaluation.metrics.total_cap = evaluation.metrics.cap_lower_bound;
    evaluation.metrics.electrical.pin_cap = evaluation.metrics.cap_lower_bound;
    evaluation.metrics.electrical.total_cap = evaluation.metrics.cap_lower_bound;
  } else {
    evaluation.metrics.cap_lower_bound = 0.0;
    evaluation.metrics.total_cap = 0.0;
    evaluation.metrics.electrical.pin_cap = 0.0;
    evaluation.metrics.electrical.total_cap = 0.0;
  }

  if (config.max_fanout > 0 && fanout > config.max_fanout) {
    evaluation.violation = ConstraintViolation::kFanout;
    return evaluation;
  }

  if (config.max_diameter > 0 && diameter > config.max_diameter) {
    evaluation.violation = ConstraintViolation::kDiameter;
    return evaluation;
  }

  if (has_cap_limit && evaluation.metrics.cap_lower_bound > config.max_cap) {
    evaluation.violation = ConstraintViolation::kCapacitance;
    return evaluation;
  }

  if (need_exact_eval) {
    auto exact_electrical = estimateExactCap(loads, routing_root, config);
    evaluation.metrics.electrical = exact_electrical;
    evaluation.metrics.total_cap = exact_electrical.total_cap;
    evaluation.metrics.wirelength = exact_electrical.wirelength;
    if (!exact_electrical.route_success) {
      evaluation.violation = ConstraintViolation::kRoutingFailed;
      return evaluation;
    }
    if (has_cap_limit && exact_electrical.total_cap > config.max_cap) {
      evaluation.violation = ConstraintViolation::kCapacitance;
      return evaluation;
    }
  }

  evaluation.violation = ConstraintViolation::kNone;
  evaluation.legal = true;
  return evaluation;
}

auto ClusterConstraintEvaluator::estimatePinCap(const std::vector<Pin*>& loads, const ClusterConfig& config) -> double
{
  double total_pin_cap = 0.0;
  for (const auto* pin : loads) {
    if (pin == nullptr) {
      continue;
    }
    total_pin_cap += queryPinCap(pin, config);
  }
  return total_pin_cap;
}

auto ClusterConstraintEvaluator::estimateExactCap(const std::vector<Pin*>& loads, const Point<int>& synthetic_root,
                                                  const ClusterConfig& config) -> ElectricalEstimate
{
  ElectricalEstimate estimate;
  estimate.exact = true;
  estimate.synthetic_root = synthetic_root;
  estimate.legalized_root = synthetic_root;
  estimate.routed_root = synthetic_root;

  std::vector<Pin*> active_loads;
  active_loads.reserve(loads.size());
  for (auto* pin : loads) {
    if (pin != nullptr) {
      active_loads.push_back(pin);
    }
  }

  estimate.pin_cap = estimatePinCap(active_loads, config);
  estimate.total_cap = estimate.pin_cap;

  if (active_loads.empty()) {
    estimate.route_success = true;
    return estimate;
  }

  const auto load_locations = CollectPinLocations(active_loads);
  if (!LegalizeRoutingRoot(estimate.synthetic_root, load_locations, estimate.legalized_root)) {
    estimate.route_success = false;
    estimate.routed_root = estimate.legalized_root;
    return estimate;
  }

  estimate.routed_root = estimate.legalized_root;
  std::vector<Router::ClockTerminal> clock_terminals;
  clock_terminals.reserve(active_loads.size());
  for (std::size_t i = 0; i < active_loads.size(); ++i) {
    auto* pin = active_loads.at(i);
    if (pin == nullptr) {
      continue;
    }
    Router::ClockTerminal terminal;
    terminal.name = std::string("sink_") + std::to_string(i);
    terminal.location = pin->get_location();
    terminal.pin_cap = queryPinCap(pin, config);
    terminal.insertion_delay = 0.0;
    clock_terminals.push_back(std::move(terminal));
  }

  Router::ClockSteinerTreeType clock_tree;
  if (!clock_terminals.empty()) {
    Router::ClockTerminal driver_terminal;
    driver_terminal.name = "legalized_root";
    driver_terminal.location = estimate.legalized_root;
    driver_terminal.pin_cap = 0.0;
    driver_terminal.insertion_delay = 0.0;

    if (config.router_kind == ClusterRouterKind::kFlute) {
      clock_tree = Router::buildFluteTree(driver_terminal, clock_terminals);
    } else if (config.router_kind == ClusterRouterKind::kSalt) {
      clock_tree = Router::buildSaltTree(driver_terminal, clock_terminals);
    } else {
      const auto bst_parameters = BuildBstParameters(config, estimate.legalized_root);
      clock_tree = (config.router_kind == ClusterRouterKind::kBst) ? Router::buildBstTree(clock_terminals, bst_parameters)
                                                                   : Router::buildCbsTree(clock_terminals, bst_parameters);
    }
  }

  if (!clock_terminals.empty() && (clock_tree.node_count() == 0 || !clock_tree.validate())) {
    LOG_WARNING << "Cluster constraint clock-aware routing returned an empty or invalid tree.";
    return estimate;
  }

  const auto* routed_root = clock_tree.get_node(clock_tree.get_root());
  if (routed_root != nullptr) {
    estimate.routed_root = routed_root->location;
  }
  estimate.wirelength = CalcTreeWirelength(clock_tree);
  auto rc_tree = Router::buildRCTree(clock_tree, config.clock_route_segment_rc);
  UpdateEstimateFromRcTree(estimate, rc_tree);
  return estimate;
}

auto ClusterConstraintEvaluator::queryPinCap(const Pin* pin, const ClusterConfig& config) -> double
{
  if (pin == nullptr) {
    return 0.0;
  }

  const auto iter = config.sink_pin_cap_pf_by_pin.find(pin);
  if (iter != config.sink_pin_cap_pf_by_pin.end()) {
    LOG_FATAL_IF(!std::isfinite(iter->second)) << "ClusterConstraintEvaluator: load pin capacitance must be finite for " << pin->get_name()
                                               << ".";
    return std::max(0.0, iter->second);
  }

  if (pin->get_inst() == nullptr || pin->get_name().empty()) {
    return 0.0;
  }

  LOG_FATAL << "ClusterConstraintEvaluator: load pin capacitance is missing for " << pin->get_name() << ".";
  return 0.0;
}

}  // namespace icts
