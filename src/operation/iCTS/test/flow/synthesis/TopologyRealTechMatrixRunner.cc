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
 * @file TopologyRealTechMatrixRunner.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief BP placement real-tech matrix runner for Topology smoke tests.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "HTreeTopologyChar.hh"
#include "Inst.hh"
#include "Pin.hh"
#include "SteinerTree.hh"
#include "TopologyRealTechScenario.hh"
#include "Tree.hh"
#include "common/realtech/setup/RealTechDesignSetup.hh"
#include "database/config/Config.hh"
#include "database/design/Net.hh"
#include "database/io/Wrapper.hh"
#include "flow/synthesis/topology/Topology.hh"
#include "geometry/Geometry.hh"
#include "module/characterization/fixture/CharacterizationRealTechFixture.hh"
#include "routing/router/Router.hh"
#include "synthesis/htree/HTree.hh"
#include "timing/TimingEngine.hh"

namespace icts_test::synthesis_realtech_smoke {
namespace {

namespace common_realtech = common::realtech;
namespace realtech_fixture = characterization::realtech;

auto MakeSkipResult(const std::string& reason) -> TopologyMatrixRunResult
{
  TopologyMatrixRunResult result;
  result.skipped = true;
  result.skip_reason = reason;
  return result;
}

auto MakeToleranceSkipResult(const std::string& reason) -> TopologyToleranceComparisonResult
{
  TopologyToleranceComparisonResult result;
  result.skipped = true;
  result.skip_reason = reason;
  return result;
}

auto FindLeafNodeForLoad(const icts::Tree& topology, const icts::Pin* load) -> const icts::TreeNode*
{
  const auto levels = topology.levels();
  if (levels.empty() || load == nullptr) {
    return nullptr;
  }

  for (const auto node_id : levels.back()) {
    const auto* node = topology.get_node(node_id);
    if (node == nullptr) {
      continue;
    }
    const auto& loads = node->get_loads();
    if (std::ranges::find(loads, load) != loads.end()) {
      return node;
    }
  }
  return nullptr;
}

auto CalcLeafLoadDistances(const icts::HTree::Output& htree_output, const std::vector<icts::Pin*>& sinks) -> std::vector<double>
{
  std::vector<double> distances;
  distances.reserve(sinks.size());
  if (sinks.empty()) {
    return distances;
  }

  for (const auto* sink : sinks) {
    const auto* leaf_node = FindLeafNodeForLoad(htree_output.topology, sink);
    if (leaf_node == nullptr || sink == nullptr) {
      continue;
    }
    distances.push_back(static_cast<double>(icts::geometry::Manhattan(leaf_node->get_position(), sink->get_location())));
  }
  return distances;
}

auto CalcDistanceStats(const std::vector<double>& distances) -> std::pair<double, double>
{
  if (distances.empty()) {
    return {0.0, 0.0};
  }
  double sum_distance = 0.0;
  double max_distance = 0.0;
  for (const double distance : distances) {
    sum_distance += distance;
    max_distance = std::max(max_distance, distance);
  }
  return {sum_distance / static_cast<double>(distances.size()), max_distance};
}

auto PopulateDistanceDeltaStats(const std::vector<double>& default_distances, const std::vector<double>& legacy_distances,
                                TopologyToleranceComparisonResult& comparison_result) -> void
{
  const std::size_t distance_count = std::min(default_distances.size(), legacy_distances.size());
  if (distance_count == 0U) {
    return;
  }

  double delta_sum = 0.0;
  double max_abs_delta = 0.0;
  for (std::size_t distance_index = 0; distance_index < distance_count; ++distance_index) {
    const double delta = default_distances.at(distance_index) - legacy_distances.at(distance_index);
    delta_sum += delta;
    max_abs_delta = std::max(max_abs_delta, std::abs(delta));
    if (delta < 0.0) {
      ++comparison_result.improved_load_count;
    } else if (delta > 0.0) {
      ++comparison_result.worsened_load_count;
    } else {
      ++comparison_result.unchanged_load_count;
    }
  }
  comparison_result.mean_distance_delta_dbu = delta_sum / static_cast<double>(distance_count);
  comparison_result.max_abs_distance_delta_dbu = max_abs_delta;
}

struct NetRouteTimingSummary
{
  std::unordered_map<const icts::Pin*, double> load_arrivals_ns;
  std::unordered_map<const icts::Pin*, double> load_wirelengths_dbu;
};

auto GetPinCap(const icts::Pin* pin) -> double
{
  if (pin == nullptr) {
    return 0.0;
  }
  const double pin_cap = icts_test::runtime::CurrentRuntime().wrapper.queryPinCapacitance(pin);
  return std::max(0.0, pin_cap);
}

auto BuildClockRouteSegmentRcFromRealTech() -> icts::ClockRouteSegmentRc
{
  return icts_test::runtime::CurrentRuntime().wrapper.queryConfiguredClockRouteSegmentRc(icts_test::runtime::CurrentRuntime().config);
}

auto CalcRouteTreeWirelengthsByName(const icts::Router::ClockSteinerTreeType& route_tree) -> std::unordered_map<std::string, double>
{
  std::unordered_map<std::string, double> wirelength_by_name;
  for (const auto& [node_name, node_id] : route_tree.get_node_name_map()) {
    const auto* node = route_tree.get_node(node_id);
    if (node == nullptr || !node->is_terminal) {
      continue;
    }

    double wirelength = 0.0;
    std::size_t current_node_id = node_id;
    while (current_node_id != route_tree.get_root()) {
      const auto* current_node = route_tree.get_node(current_node_id);
      if (current_node == nullptr || current_node->parent_edge_id == icts::Router::ClockSteinerTreeType::kInvalidId) {
        wirelength = 0.0;
        break;
      }
      const auto* parent_edge = route_tree.get_edge(current_node->parent_edge_id);
      if (parent_edge == nullptr) {
        wirelength = 0.0;
        break;
      }
      wirelength += static_cast<double>(std::max(parent_edge->distance, parent_edge->routed_distance));
      current_node_id = parent_edge->source_node_id;
    }
    wirelength_by_name[node_name] = wirelength;
  }
  return wirelength_by_name;
}

auto BuildRouteTimingForNet(const icts::Net* net) -> NetRouteTimingSummary
{
  NetRouteTimingSummary summary;
  if (net == nullptr || net->get_driver() == nullptr || net->get_loads().empty()) {
    return summary;
  }

  icts::Router::ClockSteinerTreeType route_tree;
  route_tree.reserveNodes(net->get_loads().size() + 1U);
  route_tree.reserveEdges(net->get_loads().size());
  const auto root_node_id
      = route_tree.addNode(net->get_driver()->get_name(), net->get_driver()->get_location(), true, GetPinCap(net->get_driver()), 0.0);
  if (root_node_id == icts::Router::ClockSteinerTreeType::kInvalidId) {
    return summary;
  }
  route_tree.setRoot(root_node_id);

  for (const auto* load : net->get_loads()) {
    if (load == nullptr) {
      continue;
    }
    const auto load_node_id = route_tree.addNode(load->get_name(), load->get_location(), true, GetPinCap(load), 0.0);
    if (load_node_id == icts::Router::ClockSteinerTreeType::kInvalidId) {
      continue;
    }
    const int wire_distance = icts::geometry::Manhattan(net->get_driver()->get_location(), load->get_location());
    route_tree.addEdge(root_node_id, load_node_id, wire_distance, wire_distance);
  }
  if (route_tree.edge_count() == 0U) {
    return summary;
  }

  const auto wirelengths_by_name = CalcRouteTreeWirelengthsByName(route_tree);
  auto rc_tree = icts::Router::buildRCTree(route_tree, BuildClockRouteSegmentRcFromRealTech());
  icts::TimingEngine::update(rc_tree);

  for (const auto* load : net->get_loads()) {
    if (load == nullptr) {
      continue;
    }
    const auto* vertex = rc_tree.findVertex(load->get_name());
    if (vertex == nullptr) {
      continue;
    }
    summary.load_arrivals_ns[load] = vertex->arrival;

    summary.load_wirelengths_dbu[load] = wirelengths_by_name.contains(load->get_name()) ? wirelengths_by_name.at(load->get_name()) : 0.0;
  }

  return summary;
}

struct TopologyTimingSummary
{
  double sta_arrival_skew_ns = 0.0;
  double wirelength_skew_dbu = 0.0;
  std::size_t sink_count = 0U;
};

auto CollectReachableNets(const icts::Pin* root_pin) -> std::vector<const icts::Net*>
{
  std::vector<const icts::Net*> nets;
  std::queue<const icts::Pin*> pending_pins;
  std::unordered_set<const icts::Net*> visited_nets;
  pending_pins.push(root_pin);

  while (!pending_pins.empty()) {
    const auto* driver_pin = pending_pins.front();
    pending_pins.pop();
    if (driver_pin == nullptr || driver_pin->get_net() == nullptr) {
      continue;
    }
    const auto* net = driver_pin->get_net();
    if (visited_nets.contains(net)) {
      continue;
    }
    visited_nets.insert(net);
    nets.push_back(net);

    for (const auto* load_pin : net->get_loads()) {
      if (load_pin == nullptr || load_pin->get_inst() == nullptr || !load_pin->get_inst()->is_buffer()) {
        continue;
      }
      pending_pins.push(load_pin->get_inst()->findDriverPin());
    }
  }
  return nets;
}

auto CalcTopologyTimingSummary(const icts::Topology::Build& result, const std::vector<icts::Pin*>& sinks) -> TopologyTimingSummary
{
  TopologyTimingSummary summary;
  if (result.output.htree_output.root_output_pin == nullptr || sinks.empty()) {
    return summary;
  }

  const auto nets = CollectReachableNets(result.output.htree_output.root_output_pin);
  std::unordered_map<const icts::Pin*, double> arrival_by_pin;
  std::unordered_map<const icts::Pin*, double> wirelength_by_pin;
  arrival_by_pin[result.output.htree_output.root_output_pin] = 0.0;
  wirelength_by_pin[result.output.htree_output.root_output_pin] = 0.0;

  for (const auto* net : nets) {
    if (net == nullptr || net->get_driver() == nullptr) {
      continue;
    }
    const double driver_arrival = arrival_by_pin.contains(net->get_driver()) ? arrival_by_pin.at(net->get_driver()) : 0.0;
    const double driver_wirelength = wirelength_by_pin.contains(net->get_driver()) ? wirelength_by_pin.at(net->get_driver()) : 0.0;
    const auto route_summary = BuildRouteTimingForNet(net);
    for (const auto& [load_pin, load_arrival] : route_summary.load_arrivals_ns) {
      arrival_by_pin[load_pin] = driver_arrival + load_arrival;
      if (route_summary.load_wirelengths_dbu.contains(load_pin)) {
        wirelength_by_pin[load_pin] = driver_wirelength + route_summary.load_wirelengths_dbu.at(load_pin);
      }

      auto* load_inst = load_pin == nullptr ? nullptr : load_pin->get_inst();
      if (load_inst != nullptr && load_inst->is_buffer()) {
        if (auto* output_pin = load_inst->findDriverPin(); output_pin != nullptr) {
          arrival_by_pin[output_pin] = arrival_by_pin[load_pin];
          wirelength_by_pin[output_pin] = wirelength_by_pin[load_pin];
        }
      }
    }
  }

  double min_arrival = std::numeric_limits<double>::infinity();
  double max_arrival = 0.0;
  double min_wirelength = std::numeric_limits<double>::infinity();
  double max_wirelength = 0.0;
  for (const auto* sink : sinks) {
    if (!arrival_by_pin.contains(sink) || !wirelength_by_pin.contains(sink)) {
      continue;
    }
    min_arrival = std::min(min_arrival, arrival_by_pin.at(sink));
    max_arrival = std::max(max_arrival, arrival_by_pin.at(sink));
    min_wirelength = std::min(min_wirelength, wirelength_by_pin.at(sink));
    max_wirelength = std::max(max_wirelength, wirelength_by_pin.at(sink));
    ++summary.sink_count;
  }

  if (summary.sink_count == 0U || !std::isfinite(min_arrival) || !std::isfinite(min_wirelength)) {
    return {};
  }
  summary.sta_arrival_skew_ns = max_arrival - min_arrival;
  summary.wirelength_skew_dbu = max_wirelength - min_wirelength;
  return summary;
}

auto MakeCasePrefix(unsigned wirelength_iterations, unsigned slew_cap_steps) -> std::string
{
  return "iter=" + std::to_string(wirelength_iterations) + ", step=" + std::to_string(slew_cap_steps) + ": ";
}

auto AppendCaseFailures(unsigned wirelength_iterations, unsigned slew_cap_steps, const icts::Topology::Build& result, double runtime_s,
                        const TopologyExperimentRecord& record, std::vector<std::string>& failure_messages) -> void
{
  const std::string prefix = MakeCasePrefix(wirelength_iterations, slew_cap_steps);
  if (!result.summary.success) {
    failure_messages.push_back(prefix + "failure_reason=" + result.summary.failure_reason);
  }
  if (result.summary.sink_clustering_enabled) {
    failure_messages.push_back(prefix + "sink clustering should be disabled");
  }
  if (!result.output.cluster_buffers.empty()) {
    failure_messages.push_back(prefix + "cluster buffers should be empty");
  }
  if (runtime_s > kBpBeTopSynthesisRuntimeBudgetS) {
    failure_messages.push_back(prefix + "runtime_s=" + std::to_string(runtime_s) + " exceeds budget");
  }
  if (!result.output.htree_output.best_char.has_value()) {
    failure_messages.push_back(prefix + "best htree char is missing");
  }
}

auto AppendToleranceCaseFailures(const TopologyToleranceComparisonRecord& record, std::vector<std::string>& failure_messages) -> void
{
  const std::string prefix = "tolerance=" + std::to_string(record.htree_topology_tolerance) + ": ";
  if (!record.success) {
    failure_messages.push_back(prefix + "failure_reason=" + record.failure_reason);
  }
  if (record.leaf_load_distance_mean_dbu <= 0.0 && record.leaf_load_distance_max_dbu <= 0.0) {
    failure_messages.push_back(prefix + "leaf-load distance stats were not collected");
  }
  if (record.selected_char_delay_ns <= 0.0) {
    failure_messages.push_back(prefix + "selected char delay is not positive");
  }
  if (record.sta_sink_count != record.sink_count) {
    failure_messages.push_back(prefix + "STA sink count " + std::to_string(record.sta_sink_count) + " differs from sink count "
                               + std::to_string(record.sink_count));
  }
}

}  // namespace

auto EvaluateBpBeTopFullSinkNonClusteredExperimentMatrix() -> TopologyMatrixRunResult
{
  const auto& setup_state = common_realtech::EnsureRealTechSetup();
  if (setup_state.mode != common_realtech::RealTechMode::kRealTech || !setup_state.setup_succeeded) {
    return MakeSkipResult(setup_state.summary);
  }

  const auto selected_clock = SelectLargestRealClock(std::numeric_limits<std::size_t>::max(), 2U);
  if (!selected_clock.has_value()) {
    return MakeSkipResult("No DEF-derived clock net exposes source plus at least two sinks.");
  }
  const auto& selected_clock_data = selected_clock.value();

  TopologyMatrixRunResult matrix_result;
  matrix_result.selection = selected_clock_data;
  matrix_result.records.reserve(kBpBeTopExperimentIterations.size() * kBpBeTopExperimentSteps.size());

  for (const unsigned wirelength_iterations : kBpBeTopExperimentIterations) {
    for (const unsigned slew_cap_steps : kBpBeTopExperimentSteps) {
      std::ostringstream scenario_name_stream;
      scenario_name_stream << "topology_bp_be_top_full_sink_iter" << wirelength_iterations << "_step" << slew_cap_steps;
      const std::string scenario_name = scenario_name_stream.str();

      realtech_fixture::RealTechCharFixture char_fixture;
      if (const auto prepare_error
          = char_fixture.prepare(scenario_name, std::nullopt, kSynthesisSmokeMaxSlewNs, kSynthesisSmokeMaxCapPf, true);
          prepare_error.has_value()) {
        return MakeSkipResult(*prepare_error);
      }

      icts_test::runtime::CurrentRuntime().config.set_wirelength_iterations(wirelength_iterations);
      icts_test::runtime::CurrentRuntime().config.set_slew_steps(slew_cap_steps);
      icts_test::runtime::CurrentRuntime().config.set_cap_steps(slew_cap_steps);

      icts::Topology::Config config;
      SetEnableSinkClustering(config, false);

      const auto runtime_start = std::chrono::steady_clock::now();
      icts::Net root_net(selected_clock_data.net_name + "_synthesis_root_iter" + std::to_string(wirelength_iterations) + "_step"
                         + std::to_string(slew_cap_steps));
      ConnectRootNet(root_net, selected_clock_data.source, selected_clock_data.sinks);
      const auto result = BuildTopology(root_net, config);
      const auto runtime_end = std::chrono::steady_clock::now();
      const double runtime_s = std::chrono::duration<double>(runtime_end - runtime_start).count();
      TopologyExperimentRecord record{
          .wirelength_iterations = wirelength_iterations,
          .slew_cap_steps = slew_cap_steps,
          .runtime_s = runtime_s,
          .success = result.summary.success,
          .sink_count = selected_clock_data.sinks.size(),
          .selected_depth = result.summary.selected_htree_depth.value_or(0U),
          .failure_reason = result.summary.failure_reason,
      };
      if (result.output.htree_output.best_char.has_value()) {
        record.best_pattern_id = result.output.htree_output.best_char->get_pattern_id().local_id;
        record.best_delay_ns = result.output.htree_output.best_char->get_delay();
        record.best_power_w = result.output.htree_output.best_char->get_power();
      }
      matrix_result.records.push_back(record);
      AppendCaseFailures(wirelength_iterations, slew_cap_steps, result, runtime_s, record, matrix_result.failure_messages);
    }
  }

  matrix_result.report_written = WriteTopologyMatrixReport(
      kBpBeTopTopologyScenario, "matrix_report.txt",
      FormatTopologyExperimentReport(kBpBeTopTopologyScenario, selected_clock_data, true, matrix_result.records));
  return matrix_result;
}

auto EvaluateArm9FullSinkTopologyToleranceComparison() -> TopologyToleranceComparisonResult
{
  const auto& setup_state = common_realtech::EnsureRealTechSetup();
  if (setup_state.mode != common_realtech::RealTechMode::kRealTech || !setup_state.setup_succeeded) {
    return MakeToleranceSkipResult(setup_state.summary);
  }

  const auto selected_clock = SelectLargestRealClock(std::numeric_limits<std::size_t>::max(), 2U);
  if (!selected_clock.has_value()) {
    return MakeToleranceSkipResult("No DEF-derived clock net exposes source plus at least two sinks.");
  }
  const auto& selected_clock_data = selected_clock.value();

  realtech_fixture::RealTechCharFixture char_fixture;
  if (const auto prepare_error = char_fixture.prepare("topology_arm9_full_sink_topology_tolerance", std::nullopt, kSynthesisSmokeMaxSlewNs,
                                                      kSynthesisSmokeMaxCapPf, true);
      prepare_error.has_value()) {
    return MakeToleranceSkipResult(*prepare_error);
  }

  TopologyToleranceComparisonResult comparison_result;
  comparison_result.selection = selected_clock_data;
  comparison_result.records.reserve(2U);
  std::vector<std::vector<double>> distance_sets;
  distance_sets.reserve(2U);

  icts_test::runtime::CurrentRuntime().config.set_wirelength_iterations(3U);
  icts_test::runtime::CurrentRuntime().config.set_slew_steps(15U);
  icts_test::runtime::CurrentRuntime().config.set_cap_steps(15U);

  for (const double topology_tolerance : {0.1, 0.0}) {
    icts_test::runtime::CurrentRuntime().config.set_htree_topology_tolerance(topology_tolerance);
    icts::Topology::Config config;
    SetEnableSinkClustering(config, false);

    const auto runtime_start = std::chrono::steady_clock::now();
    icts::Net root_net(selected_clock_data.net_name + "_synthesis_root_tol" + std::to_string(topology_tolerance));
    ConnectRootNet(root_net, selected_clock_data.source, selected_clock_data.sinks);
    const auto result = BuildTopology(root_net, config);
    const auto runtime_end = std::chrono::steady_clock::now();
    const double runtime_s = std::chrono::duration<double>(runtime_end - runtime_start).count();
    const auto leaf_load_distances = CalcLeafLoadDistances(result.output.htree_output, selected_clock_data.sinks);
    const auto [leaf_load_distance_mean, leaf_load_distance_max] = CalcDistanceStats(leaf_load_distances);
    const auto timing_summary = CalcTopologyTimingSummary(result, selected_clock_data.sinks);

    TopologyToleranceComparisonRecord record{
        .htree_topology_tolerance = topology_tolerance,
        .runtime_s = runtime_s,
        .success = result.summary.success,
        .sink_count = selected_clock_data.sinks.size(),
        .leaf_load_distance_mean_dbu = leaf_load_distance_mean,
        .leaf_load_distance_max_dbu = leaf_load_distance_max,
        .sta_arrival_skew_ns = timing_summary.sta_arrival_skew_ns,
        .wirelength_skew_dbu = timing_summary.wirelength_skew_dbu,
        .sta_sink_count = timing_summary.sink_count,
        .selected_char_delay_ns
        = result.output.htree_output.best_char.has_value() ? result.output.htree_output.best_char->get_delay() : 0.0,
        .failure_reason = result.summary.failure_reason,
    };
    AppendToleranceCaseFailures(record, comparison_result.failure_messages);
    distance_sets.push_back(leaf_load_distances);
    comparison_result.records.push_back(std::move(record));
  }

  if (distance_sets.size() == 2U) {
    PopulateDistanceDeltaStats(distance_sets.at(0), distance_sets.at(1), comparison_result);
  }

  comparison_result.report_written = WriteTopologyMatrixReport(
      "topology_arm9_full_sink_topology_tolerance", "topology_tolerance_comparison.csv",
      FormatTopologyToleranceComparisonReport("topology_arm9_full_sink_topology_tolerance", selected_clock_data, comparison_result));
  return comparison_result;
}

}  // namespace icts_test::synthesis_realtech_smoke
