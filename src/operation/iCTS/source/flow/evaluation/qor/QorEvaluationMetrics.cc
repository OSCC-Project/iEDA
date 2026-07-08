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
 * @file QorEvaluationMetrics.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief CTS QoR evaluation metric and summary helpers.
 */

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "ClockDAG.hh"
#include "ClockRouteSegmentRc.hh"
#include "Log.hh"
#include "Point.hh"
#include "Qor.hh"
#include "SteinerTree.hh"
#include "TimingEngine.hh"
#include "design/Clock.hh"
#include "design/Inst.hh"
#include "design/Net.hh"
#include "design/Pin.hh"
#include "evaluation/qor/ClockQorMetricCollector.hh"
#include "evaluation/qor/QorEvaluation.hh"
#include "io/Wrapper.hh"
#include "logger/Schema.hh"
#include "router/Router.hh"

namespace icts::qor_evaluation {
namespace {

auto calcRouteWirelength(const Router::ClockSteinerTreeType& route_tree) -> int64_t
{
  int64_t wirelength = 0;
  for (const auto& edge : route_tree.get_edges()) {
    wirelength += std::max(edge.distance, edge.routed_distance);
  }
  return wirelength;
}

auto dbuToUm(int64_t dbu, const QorSummary& summary) -> double
{
  if (summary.design_dbu_per_um <= 0) {
    return 0.0;
  }
  const auto dbu_per_um = static_cast<double>(summary.design_dbu_per_um);
  return static_cast<double>(dbu) / dbu_per_um;
}

auto hasValidLocation(const Point<int>& point) -> bool
{
  return point.get_x() >= 0 && point.get_y() >= 0;
}

auto calcHpwlDbu(const Net* net) -> int64_t
{
  if (net == nullptr || net->get_driver() == nullptr) {
    return 0;
  }

  std::vector<Point<int>> locations;
  if (hasValidLocation(net->get_driver()->get_location())) {
    locations.push_back(net->get_driver()->get_location());
  }
  for (const auto* load : net->get_loads()) {
    if (load != nullptr && hasValidLocation(load->get_location())) {
      locations.push_back(load->get_location());
    }
  }
  if (locations.size() < 2U) {
    return 0;
  }

  int min_x = locations.front().get_x();
  int max_x = locations.front().get_x();
  int min_y = locations.front().get_y();
  int max_y = locations.front().get_y();
  for (const auto& location : locations) {
    min_x = std::min(min_x, location.get_x());
    max_x = std::max(max_x, location.get_x());
    min_y = std::min(min_y, location.get_y());
    max_y = std::max(max_y, location.get_y());
  }
  return static_cast<int64_t>(max_x - min_x) + static_cast<int64_t>(max_y - min_y);
}

auto containsClockLoad(const Clock& clock, const Pin* pin) -> bool
{
  const auto& clock_loads = clock.get_loads();
  return std::ranges::find(clock_loads, pin) != clock_loads.end();
}

auto addWirelengthByRole(Qor& statistics, ClockNetRole role, double wirelength_um, double hpwl_um) -> void
{
  switch (role) {
    case ClockNetRole::kSourceToRoot:
      statistics.top_wirelength_um += wirelength_um;
      statistics.hpwl_top_wirelength_um += hpwl_um;
      break;
    case ClockNetRole::kTrunk:
      statistics.trunk_wirelength_um += wirelength_um;
      statistics.hpwl_trunk_wirelength_um += hpwl_um;
      break;
    case ClockNetRole::kLeaf:
      statistics.leaf_wirelength_um += wirelength_um;
      statistics.hpwl_leaf_wirelength_um += hpwl_um;
      break;
  }
  statistics.total_wirelength_um += wirelength_um;
  statistics.hpwl_total_wirelength_um += hpwl_um;
  statistics.max_net_wirelength_um = std::max(statistics.max_net_wirelength_um, wirelength_um);
  statistics.hpwl_max_net_wirelength_um = std::max(statistics.hpwl_max_net_wirelength_um, hpwl_um);
}

auto instTypeName(const Inst& inst) -> std::string
{
  if (inst.is_buffer()) {
    return "Buffer";
  }
  if (inst.is_inverter()) {
    return "Inverter";
  }
  if (inst.is_clock_gate()) {
    return "ICG";
  }
  if (inst.is_mux()) {
    return "Mux";
  }
  if (inst.is_macro_block()) {
    return "Macro";
  }
  if (inst.is_flipflop()) {
    return "FlipFlop";
  }
  if (inst.is_latch()) {
    return "Latch";
  }
  if (inst.is_clock_logic()) {
    return "ClockLogic";
  }
  if (inst.is_boundary_load()) {
    return "BoundaryLoad";
  }
  return "Others";
}

auto calcInstInputPinCapPf(Wrapper& wrapper, const Inst& inst) -> double
{
  double total_cap_pf = 0.0;
  for (const auto* pin : inst.get_pins()) {
    if (pin == nullptr || pin->get_inst() != &inst) {
      continue;
    }
    total_cap_pf += wrapper.queryPinCapacitance(pin);
  }
  return total_cap_pf;
}

}  // namespace

auto ClearStatistics(Qor& statistics) -> void
{
  statistics = Qor{};
}

auto ClearSummary(QorSummary& summary) -> void
{
  summary.has_evaluation_result = false;
  summary.qor_metric_status = "unavailable";
  summary.physical_metric_source = "unavailable";
  summary.path_depth_metric_status = "unavailable";
  summary.final_clock_buffer_count = 0;
  summary.final_buffer_area_um2 = 0.0;
  summary.clock_member_buffer_count = 0;
  summary.max_clock_net_wirelength_um = 0.0;
  summary.total_clock_network_wirelength_um = 0.0;
  summary.max_clock_net_wirelength_dbu = 0;
  summary.total_clock_network_wirelength_dbu = 0.0;
  summary.design_dbu_per_um = 0;
  summary.buffer_num = 0;
  summary.buffer_area = 0.0;
  summary.clock_path_min_buffer = 0;
  summary.clock_path_max_buffer = 0;
  summary.feature_max_clock_network_level = 0;
  summary.max_clock_wirelength = 0;
  summary.total_clock_wirelength = 0.0;
}

auto SyncCompatibilityAliases(QorSummary& summary) -> void
{
  summary.buffer_num = summary.final_clock_buffer_count;
  summary.buffer_area = summary.final_buffer_area_um2;
  summary.max_clock_wirelength = summary.max_clock_net_wirelength_dbu;
  summary.total_clock_wirelength = summary.total_clock_network_wirelength_dbu;
}

auto AppendPathDepthStats(const ClockDAG::PathBufferStats& path_stats, QorSummary& summary) -> void
{
  summary.path_depth_metric_status = path_stats.status;
  summary.clock_path_min_buffer = 0;
  summary.clock_path_max_buffer = 0;
  summary.feature_max_clock_network_level = 0;
  if (!path_stats.available) {
    return;
  }
  summary.clock_path_min_buffer = path_stats.min_buffer_count;
  summary.clock_path_max_buffer = path_stats.max_buffer_count;
  summary.feature_max_clock_network_level = path_stats.max_buffer_count;
}

auto ClassifyClockNet(const Clock& clock, const Net* net) -> ClockNetRole
{
  if (net == nullptr) {
    return ClockNetRole::kTrunk;
  }
  if (net == clock.get_clock_source_net()) {
    return ClockNetRole::kSourceToRoot;
  }

  for (const auto* load : net->get_loads()) {
    if (load == nullptr) {
      continue;
    }
    const auto* inst = load->get_inst();
    if (containsClockLoad(clock, load) || inst == nullptr || !inst->is_clock_propagation_cell()) {
      return ClockNetRole::kLeaf;
    }
  }
  return ClockNetRole::kTrunk;
}

auto AccumulateInstStatistics(Wrapper& wrapper, const Inst& inst, Qor& statistics) -> void
{
  if (!inst.is_buffer()) {
    return;
  }

  const auto& cell_master = inst.get_cell_master();
  if (cell_master.empty()) {
    return;
  }

  const std::string cell_type = instTypeName(inst);
  const double area_um2 = wrapper.queryCellAreaUm2(cell_master);
  const double cap_pf = calcInstInputPinCapPf(wrapper, inst);

  auto& cell_stat = statistics.cell_stats[cell_type];
  ++cell_stat.count;
  cell_stat.total_area_um2 += area_um2;
  cell_stat.total_cap_pf += cap_pf;

  auto& lib_dist = statistics.lib_cell_dist[cell_master];
  lib_dist.cell_type = cell_type;
  ++lib_dist.count;
  lib_dist.total_area_um2 += area_um2;
}

auto MeasureClockNet(const ClockNetMeasurementInput& input) -> std::optional<ClockNetMeasurement>
{
  LOG_FATAL_IF(input.config == nullptr) << "CTS QoR clock-net measurement requires config.";
  LOG_FATAL_IF(input.wrapper == nullptr) << "CTS QoR clock-net measurement requires wrapper.";
  const auto& config = *input.config;
  auto& wrapper = *input.wrapper;
  auto* net = input.net;
  auto route_tree = net == nullptr ? Router::ClockSteinerTreeType{} : Router::buildClockNetTree(*net);
  if (route_tree.node_count() == 0 || route_tree.edge_count() == 0) {
    return std::nullopt;
  }

  const auto wirelength = calcRouteWirelength(route_tree);
  const auto hpwl = calcHpwlDbu(net);

  if (wrapper.is_design_ready()) {
    auto rc_tree = Router::buildRCTree(route_tree, wrapper.queryConfiguredClockRouteSegmentRc(config));
    auto timing_metrics = TimingEngine::update(rc_tree);
    (void) timing_metrics;
  }

  return ClockNetMeasurement{
      .role = input.role,
      .wirelength_dbu = wirelength,
      .hpwl_dbu = hpwl,
  };
}

auto AppendClockNetStatistics(const std::vector<ClockNetMeasurement>& measurements, QorSummary& summary, Qor& statistics) -> void
{
  for (const auto& measurement : measurements) {
    const double wirelength_um = dbuToUm(measurement.wirelength_dbu, summary);
    const double hpwl_um = dbuToUm(measurement.hpwl_dbu, summary);
    const auto wirelength = measurement.wirelength_dbu;

    summary.total_clock_network_wirelength_um += wirelength_um;
    summary.max_clock_net_wirelength_um = std::max(summary.max_clock_net_wirelength_um, wirelength_um);
    summary.total_clock_network_wirelength_dbu += static_cast<double>(wirelength);
    summary.max_clock_net_wirelength_dbu = std::max(summary.max_clock_net_wirelength_dbu, static_cast<int32_t>(wirelength));
    addWirelengthByRole(statistics, measurement.role, wirelength_um, hpwl_um);
  }
}

auto EmitEvaluationSummary(SchemaWriter& reporter, const QorSummary& summary) -> void
{
  const bool path_depth_available = summary.path_depth_metric_status == "available";
  EmitKeyValueTable(reporter, "CTS Evaluation Overview",
                    {
                        {"qor_metric_status", summary.qor_metric_status},
                        {"physical_metric_source", summary.physical_metric_source},
                        {"clock_member_buffer_count", std::to_string(summary.clock_member_buffer_count)},
                        {"path_depth_metric_status", summary.path_depth_metric_status},
                        {"clock_path_min_buffer", path_depth_available ? std::to_string(summary.clock_path_min_buffer) : "n/a"},
                        {"clock_path_max_buffer", path_depth_available ? std::to_string(summary.clock_path_max_buffer) : "n/a"},
                        {"max_level_of_clock_tree", path_depth_available ? std::to_string(summary.feature_max_clock_network_level) : "n/a"},
                        {"design_units", std::to_string(summary.design_dbu_per_um) + " DBU/um"},
                        {"statistics_reports", "wirelength.rpt, cell_stats.rpt, lib_cell_dist.rpt"},
                    });
}

}  // namespace icts::qor_evaluation
