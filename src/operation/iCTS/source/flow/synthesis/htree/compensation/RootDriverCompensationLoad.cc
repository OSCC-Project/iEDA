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
 * @file RootDriverCompensationLoad.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Root-closure load estimation helpers for H-tree root-driver compensation.
 */

#include <glog/logging.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <ostream>
#include <ratio>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "BufferingPattern.hh"
#include "HTreeTopologyPattern.hh"
#include "Log.hh"
#include "PatternId.hh"
#include "Pin.hh"
#include "Point.hh"
#include "SteinerTree.hh"
#include "Tree.hh"
#include "ValueLattice.hh"
#include "design/Design.hh"
#include "io/Wrapper.hh"
#include "router/Router.hh"
#include "synthesis/htree/compensation/RootDriverCompensation.hh"
#include "synthesis/htree/compensation/RootDriverCompensationState.hh"
#include "synthesis/htree/segment_pruning/SegmentPatternLibrary.hh"
#include "synthesis/htree/segment_pruning/TopologyPatternLibrary.hh"

namespace icts::htree {
namespace {

constexpr const char* kRootClosureFluteEstimator = "flute_clock_steiner_tree";
constexpr const char* kRootClosureSingleLoadEstimator = "single_load_manhattan";
constexpr const char* kRootClosureHpwlEstimate = "hpwl_bbox_estimate";

auto HashCombine(std::size_t seed, std::size_t value) -> std::size_t
{
  return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U));
}

auto ResolveRoutingLayer(const RootDriverCompensationInput& input) -> int
{
  LOG_FATAL_IF(input.routing_layer <= 0) << "HTree: routing layer is not configured for root-driver compensation.";
  return input.routing_layer;
}

auto ResolveWireWidth(const RootDriverCompensationInput& input) -> std::optional<double>
{
  return input.wire_width_um.has_value() && *input.wire_width_um > 0.0 ? input.wire_width_um : std::nullopt;
}

auto DbuToUm(const RootDriverCompensationInput& input, int64_t dbu) -> double
{
  LOG_FATAL_IF(input.dbu_per_um <= 0) << "HTree: DBU-per-micron is unavailable for root-driver compensation.";
  return static_cast<double>(std::max<int64_t>(dbu, 0)) / static_cast<double>(input.dbu_per_um);
}

auto QueryWireCapForDbuLength(const RootDriverCompensationInput& input, int64_t length_dbu) -> double
{
  const double length_um = DbuToUm(input, length_dbu);
  if (length_um <= 0.0) {
    return 0.0;
  }

  LOG_FATAL_IF(input.wrapper == nullptr) << "HTree: Wrapper is unavailable for root-driver compensation.";
  const double wire_cap_pf = input.wrapper->queryRequiredWireCapacitance(ResolveRoutingLayer(input), length_um, ResolveWireWidth(input));
  if (!std::isfinite(wire_cap_pf) || wire_cap_pf < 0.0) {
    LOG_WARNING << "HTree: root-closure wire-cap query returned an invalid value for length " << length_um << " um.";
    return 0.0;
  }
  return wire_cap_pf;
}

auto CalcManhattanDbu(const Point<int>& lhs, const Point<int>& rhs) -> int64_t
{
  return static_cast<int64_t>(std::abs(lhs.get_x() - rhs.get_x())) + static_cast<int64_t>(std::abs(lhs.get_y() - rhs.get_y()));
}

auto InterpolateRootClosurePoint(const Point<int>& source, const Point<int>& sink, double normalized_position) -> Point<int>
{
  const double clamped_position = std::clamp(normalized_position, 0.0, 1.0);
  const int dx = sink.get_x() - source.get_x();
  const int dy = sink.get_y() - source.get_y();
  const int total_distance = std::abs(dx) + std::abs(dy);
  if (total_distance == 0) {
    return source;
  }

  const int target_distance = static_cast<int>(std::lround(clamped_position * static_cast<double>(total_distance)));
  const int x_step = std::min(std::abs(dx), target_distance);
  const int y_step = std::max(0, target_distance - x_step);
  const int x = source.get_x() + ((dx >= 0) ? x_step : -x_step);
  const int y = source.get_y() + ((dy >= 0) ? y_step : -y_step);
  return Point<int>(x, y);
}

auto EstimateHpwlWire(const RootDriverCompensationInput& input, const Point<int>& root_location,
                      const std::vector<RootClosureTerminal>& terminals, RootDriverCompensationStats& stats) -> RootClosureWireEstimate
{
  int min_x = root_location.get_x();
  int max_x = root_location.get_x();
  int min_y = root_location.get_y();
  int max_y = root_location.get_y();
  for (const auto& terminal : terminals) {
    min_x = std::min(min_x, terminal.location.get_x());
    max_x = std::max(max_x, terminal.location.get_x());
    min_y = std::min(min_y, terminal.location.get_y());
    max_y = std::max(max_y, terminal.location.get_y());
  }

  const auto hpwl_dbu = static_cast<int64_t>(max_x - min_x) + static_cast<int64_t>(max_y - min_y);
  ++stats.hpwl_route_estimate_count;
  return RootClosureWireEstimate{
      .route_estimator = kRootClosureHpwlEstimate,
      .wire_cap_pf = QueryWireCapForDbuLength(input, hpwl_dbu),
      .routed_wirelength_um = DbuToUm(input, hpwl_dbu),
  };
}

auto EstimateRootClosureWire(const RootDriverCompensationInput& input, const Point<int>& root_location,
                             const std::vector<RootClosureTerminal>& terminals, RootDriverCompensationStats& stats)
    -> RootClosureWireEstimate
{
  if (terminals.empty()) {
    return RootClosureWireEstimate{.route_estimator = "none"};
  }

  if (terminals.size() == 1U) {
    const auto length_dbu = CalcManhattanDbu(root_location, terminals.front().location);
    ++stats.hpwl_route_estimate_count;
    return RootClosureWireEstimate{
        .route_estimator = kRootClosureSingleLoadEstimator,
        .wire_cap_pf = QueryWireCapForDbuLength(input, length_dbu),
        .routed_wirelength_um = DbuToUm(input, length_dbu),
    };
  }

  Router::ClockTerminal driver_terminal;
  driver_terminal.name = "root_driver_output";
  driver_terminal.location = root_location;

  std::vector<Router::ClockTerminal> load_terminals;
  load_terminals.reserve(terminals.size());
  for (std::size_t terminal_index = 0; terminal_index < terminals.size(); ++terminal_index) {
    const auto& terminal = terminals.at(terminal_index);
    Router::ClockTerminal load_terminal;
    load_terminal.name = terminal.name.empty() ? "root_closure_terminal_" + std::to_string(terminal_index) : terminal.name;
    load_terminal.location = terminal.location;
    load_terminal.pin_cap = terminal.pin_cap_pf;
    load_terminals.push_back(std::move(load_terminal));
  }

  auto route_tree = Router::buildFluteTree(driver_terminal, load_terminals);
  if (route_tree.validate() && route_tree.edge_count() > 0U) {
    double wire_cap_pf = 0.0;
    double routed_wirelength_um = 0.0;
    for (const auto& edge : route_tree.get_edges()) {
      const auto length_dbu = static_cast<int64_t>(std::max(edge.distance, edge.routed_distance));
      wire_cap_pf += QueryWireCapForDbuLength(input, length_dbu);
      routed_wirelength_um += DbuToUm(input, length_dbu);
    }
    ++stats.flute_route_estimate_count;
    return RootClosureWireEstimate{
        .route_estimator = kRootClosureFluteEstimator,
        .wire_cap_pf = wire_cap_pf,
        .routed_wirelength_um = routed_wirelength_um,
    };
  }

  LOG_WARNING << "HTree: root-closure FLUTE estimate was unavailable; using HPWL estimate.";
  return EstimateHpwlWire(input, root_location, terminals, stats);
}

auto MakeBufferRootClosureTerminal(Wrapper& wrapper, const TreeNode& parent_node, const TreeNode& child_node,
                                   const BufferingPattern& segment_pattern, std::size_t terminal_index) -> RootClosureTerminal
{
  const auto& positions = segment_pattern.get_buffer_positions();
  const auto& cell_masters = segment_pattern.get_cell_masters();
  if (positions.empty() || cell_masters.empty()) {
    return {};
  }

  const std::string& first_buffer_master = cell_masters.front();
  return RootClosureTerminal{
      .name = "root_closure_buffer_input_" + std::to_string(terminal_index),
      .location = InterpolateRootClosurePoint(parent_node.get_position(), child_node.get_position(), positions.front()),
      .pin_cap_pf = wrapper.queryCharInputPinCap(first_buffer_master),
  };
}

auto SegmentHasRealBuffer(const BufferingPattern& segment_pattern) -> bool
{
  return !segment_pattern.get_buffer_positions().empty() && !segment_pattern.get_cell_masters().empty();
}

auto BuildRootClosureLoadSignature(PatternId topology_pattern_id, const TopologyPatternLibrary& topology_library,
                                   const BufferPatternLibrary& segment_pattern_library) -> RootClosureLoadSignature
{
  RootClosureLoadSignature signature;
  const auto topology_pattern = topology_library.materialize(topology_pattern_id);
  const auto& level_segment_pattern_ids = topology_pattern.get_level_segment_pattern_ids();
  signature.root_prefix_segment_pattern_ids.reserve(level_segment_pattern_ids.size());
  for (const auto segment_pattern_id : level_segment_pattern_ids) {
    const auto* segment_pattern = segment_pattern_library.find(segment_pattern_id);
    LOG_FATAL_IF(segment_pattern == nullptr) << "HTree: candidate segment pattern metadata is missing during root-load signature build.";

    signature.root_prefix_segment_pattern_ids.push_back(segment_pattern_id);
    if (SegmentHasRealBuffer(*segment_pattern)) {
      signature.ends_at_real_buffer = true;
      break;
    }
  }
  return signature;
}

auto CollectExternalLoadTerminals(Wrapper& wrapper, const std::vector<std::size_t>& boundary_node_ids, const Tree& topology)
    -> std::vector<RootClosureTerminal>
{
  std::vector<RootClosureTerminal> terminals;
  std::unordered_set<const Pin*> seen_pins;
  for (const auto node_id : boundary_node_ids) {
    const auto* node = topology.get_node(node_id);
    if (node == nullptr) {
      continue;
    }
    for (auto* load : node->get_loads()) {
      if (load == nullptr || !seen_pins.insert(load).second) {
        continue;
      }
      terminals.push_back(RootClosureTerminal{
          .name = Design::getPinFullName(load),
          .location = load->get_location(),
          .pin_cap_pf = wrapper.queryPinCapacitance(load),
      });
    }
  }
  return terminals;
}

auto CountLoadedRootBranches(const TreeNode& root_node, const Tree& topology) -> std::size_t
{
  std::size_t branch_count = 0U;
  for (const auto child_id : root_node.get_children()) {
    if (child_id == std::numeric_limits<std::size_t>::max()) {
      continue;
    }
    const auto* child_node = topology.get_node(child_id);
    if (child_node != nullptr && !child_node->get_loads().empty()) {
      ++branch_count;
    }
  }
  return branch_count;
}

auto SetSourceBoundaryLoadEstimate(RootClosureLoadEstimate& estimate, double source_boundary_load_cap_pf,
                                   std::size_t source_boundary_branch_count, const UniformValueLattice& cap_lattice) -> void
{
  estimate.source_boundary_load_cap_pf = source_boundary_load_cap_pf;
  estimate.source_boundary_branch_count = source_boundary_branch_count;
  estimate.source_boundary_bucket_idx
      = source_boundary_load_cap_pf > 0.0 && cap_lattice.isValid() ? cap_lattice.coveringIndex(source_boundary_load_cap_pf) : 0U;
}

auto MakeRootClosureLoadEstimate(const RootDriverCompensationInput& input, const Point<int>& root_location,
                                 const std::vector<RootClosureTerminal>& terminals, const UniformValueLattice& cap_lattice,
                                 RootDriverCompensationStats& stats) -> RootClosureLoadEstimate
{
  RootClosureLoadEstimate estimate;
  estimate.source = kRootDriverCompensationLoadSource;
  estimate.terminal_count = terminals.size();
  if (terminals.empty()) {
    estimate.route_estimator = "none";
    return estimate;
  }

  for (const auto& terminal : terminals) {
    estimate.terminal_pin_cap_pf += std::max(0.0, terminal.pin_cap_pf);
  }
  const auto wire_estimate = EstimateRootClosureWire(input, root_location, terminals, stats);
  estimate.route_estimator = wire_estimate.route_estimator;
  estimate.wire_cap_pf = wire_estimate.wire_cap_pf;
  estimate.routed_wirelength_um = wire_estimate.routed_wirelength_um;
  estimate.total_load_cap_pf = estimate.terminal_pin_cap_pf + estimate.wire_cap_pf;
  estimate.bucket_idx = cap_lattice.isValid() ? cap_lattice.coveringIndex(estimate.total_load_cap_pf) : 0U;
  estimate.valid = estimate.total_load_cap_pf > 0.0;
  SetSourceBoundaryLoadEstimate(estimate, estimate.total_load_cap_pf, estimate.terminal_count > 0U ? 1U : 0U, cap_lattice);
  return estimate;
}

auto ResolveRootClosureLoadEstimate(PatternId topology_pattern_id, const TopologyPatternLibrary& topology_library,
                                    const BufferPatternLibrary& segment_pattern_library, const Tree& topology,
                                    const RootDriverCompensationInput& input, RootDriverCompensationStats& stats) -> RootClosureLoadEstimate
{
  ++stats.load_resolution_count;
  const auto* root_node = topology.get_node(topology.get_root());
  if (root_node == nullptr) {
    LOG_WARNING << "HTree: root-driver compensation load resolution failed because topology root is missing.";
    ++stats.load_resolution_failure_count;
    return {};
  }

  const auto topology_pattern = topology_library.materialize(topology_pattern_id);
  const auto& level_segment_pattern_ids = topology_pattern.get_level_segment_pattern_ids();
  std::vector<std::size_t> active_node_ids{topology.get_root()};
  const std::size_t loaded_root_branch_count = CountLoadedRootBranches(*root_node, topology);

  for (const auto segment_pattern_id : level_segment_pattern_ids) {
    const auto* segment_pattern = segment_pattern_library.find(segment_pattern_id);
    LOG_FATAL_IF(segment_pattern == nullptr) << "HTree: candidate segment pattern metadata is missing during root-load resolution.";

    const bool segment_has_real_buffer = SegmentHasRealBuffer(*segment_pattern);
    std::vector<RootClosureTerminal> buffer_input_terminals;
    std::vector<std::size_t> next_active_node_ids;

    for (const auto parent_id : active_node_ids) {
      const auto* parent_node = topology.get_node(parent_id);
      if (parent_node == nullptr) {
        continue;
      }
      for (const auto child_id : parent_node->get_children()) {
        if (child_id == std::numeric_limits<std::size_t>::max()) {
          continue;
        }
        const auto* child_node = topology.get_node(child_id);
        if (child_node == nullptr || child_node->get_loads().empty()) {
          continue;
        }
        if (segment_has_real_buffer) {
          LOG_FATAL_IF(input.wrapper == nullptr) << "HTree: Wrapper is unavailable for root-driver load resolution.";
          buffer_input_terminals.push_back(
              MakeBufferRootClosureTerminal(*input.wrapper, *parent_node, *child_node, *segment_pattern, buffer_input_terminals.size()));
        } else {
          next_active_node_ids.push_back(child_id);
        }
      }
    }

    if (segment_has_real_buffer) {
      auto estimate = MakeRootClosureLoadEstimate(input, root_node->get_position(), buffer_input_terminals, input.cap_lattice, stats);
      if (loaded_root_branch_count > 0U) {
        SetSourceBoundaryLoadEstimate(estimate, estimate.total_load_cap_pf / static_cast<double>(loaded_root_branch_count),
                                      loaded_root_branch_count, input.cap_lattice);
      }
      if (!estimate.valid) {
        ++stats.load_resolution_failure_count;
      }
      return estimate;
    }

    active_node_ids = std::move(next_active_node_ids);
    if (active_node_ids.empty()) {
      break;
    }
  }

  LOG_FATAL_IF(input.wrapper == nullptr) << "HTree: Wrapper is unavailable for root-driver load resolution.";
  auto estimate = MakeRootClosureLoadEstimate(
      input, root_node->get_position(), CollectExternalLoadTerminals(*input.wrapper, active_node_ids, topology), input.cap_lattice, stats);
  if (loaded_root_branch_count > 0U) {
    SetSourceBoundaryLoadEstimate(estimate, estimate.total_load_cap_pf / static_cast<double>(loaded_root_branch_count),
                                  loaded_root_branch_count, input.cap_lattice);
  }
  if (!estimate.valid) {
    ++stats.load_resolution_failure_count;
  }
  return estimate;
}

}  // namespace

auto RootClosureLoadSignatureHash::operator()(const RootClosureLoadSignature& signature) const noexcept -> std::size_t
{
  std::size_t seed = std::hash<bool>{}(signature.ends_at_real_buffer);
  for (const auto segment_pattern_id : signature.root_prefix_segment_pattern_ids) {
    seed = HashCombine(seed, std::hash<PatternId>{}(segment_pattern_id));
  }
  return seed;
}

auto QueryRootClosureLoadEstimate(PatternId pattern_id, const TopologyPatternLibrary& topology_library,
                                  const BufferPatternLibrary& segment_pattern_library, const Tree& topology,
                                  RootDriverCompensationState& state) -> RootClosureLoadEstimate
{
  auto signature = BuildRootClosureLoadSignature(pattern_id, topology_library, segment_pattern_library);
  auto cache_it = state.root_load_by_signature.find(signature);
  if (cache_it != state.root_load_by_signature.end()) {
    ++state.stats.load_resolution_cache_hit_count;
    return cache_it->second;
  }

  const auto resolution_start = std::chrono::steady_clock::now();
  auto estimate = ResolveRootClosureLoadEstimate(pattern_id, topology_library, segment_pattern_library, topology, state.input, state.stats);
  state.stats.load_resolution_runtime_ms
      += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - resolution_start).count();
  return state.root_load_by_signature.emplace(std::move(signature), std::move(estimate)).first->second;
}

}  // namespace icts::htree
