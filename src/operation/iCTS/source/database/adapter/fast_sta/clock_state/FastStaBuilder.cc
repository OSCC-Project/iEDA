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
 * @file FastStaBuilder.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief Initialization bridge from committed CTS state to fast STA context.
 */

#include "FastStaBuilder.hh"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "FastSta.hh"
#include "FastStaClockState.hh"
#include "FastStaClockTree.hh"
#include "FastStaLiberty.hh"
#include "FastStaLibertyModel.hh"
#include "FastStaParasitics.hh"
#include "Log.hh"
#include "design/Clock.hh"
#include "design/Design.hh"
#include "design/Net.hh"
#include "io/Wrapper.hh"

namespace icts {

class Pin;

namespace {

auto elapsedSeconds(std::chrono::steady_clock::time_point start_time) -> double
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
}

auto logBuilderStage(const Clock& clock, std::string_view stage_name, double runtime_s, const FastStaClockContext& context) -> void
{
  LOG_INFO << "FastStaBuilder: " << stage_name << " for clock \"" << clock.get_clock_name() << "\" finished in " << runtime_s
           << " s, nodes=" << context.nodes.size() << ", nets=" << context.nets.size()
           << ", liberty_cells=" << context.liberty_cell_by_master.size() << ".";
}

auto applyEnvironment(const FastStaEnvironment& environment, FastStaClockContext& context) -> void
{
  LOG_FATAL_IF(environment.wrapper == nullptr) << "FastStaBuilder: Wrapper is not bound.";
  LOG_FATAL_IF(environment.dbu_per_um <= 0) << "FastStaBuilder: DBU-per-micron is unavailable.";
  LOG_FATAL_IF(environment.routing_layer <= 0) << "FastStaBuilder: routing layer is not configured.";
  context.wrapper = environment.wrapper;
  context.dbu_per_um = environment.dbu_per_um;
  context.routing_layer = environment.routing_layer;
  context.wire_width_um = environment.wire_width_um;
  context.root_input_slew_ns = std::max(0.0, environment.root_input_slew_ns);
}

auto queryWrapperBackedSinkPinCap(Wrapper& wrapper, const Pin* pin) -> double
{
  return wrapper.queryPinCapacitance(pin);
}

auto queryWrapperBackedSinkSlewLimit(const FastStaEnvironment& environment, const Pin* pin) -> double
{
  return environment.wrapper->queryPinSlewLimit(Wrapper::PinSlewLimitInput{
      .pin = pin,
      .configured_max_sink_tran_ns = environment.max_sink_tran_ns,
  });
}

auto queryWrapperBackedSourceCapLimit(const FastStaEnvironment& environment, const Clock& clock) -> double
{
  return environment.wrapper->queryClockSourceDriveCapLimit(Wrapper::ClockSourceDriveCapLimitInput{
      .clock_source = clock.get_clock_source(),
      .configured_max_cap_pf = environment.max_cap_pf,
  });
}

auto isSourceBoundaryNet(const Clock& clock, const FastStaClockContext& context, const FastStaNet& net) -> bool
{
  if (net.driver_node_id == context.source_node_id) {
    return true;
  }
  const auto* source_net = clock.get_clock_source_net();
  return source_net != nullptr && net.name == source_net->get_name();
}

auto collectClockCellTimingData(const FastStaEnvironment& environment, const Clock& clock, FastStaClockContext& context) -> void
{
  auto& wrapper = *environment.wrapper;
  for (auto& node : context.nodes) {
    if (node.cell_master.empty()) {
      continue;
    }
    if (!context.liberty_cell_by_master.contains(node.cell_master)) {
      context.liberty_cell_by_master[node.cell_master] = FastStaLiberty::extractBufferCell(wrapper, node.cell_master);
    }
    const auto& liberty_cell = context.liberty_cell_by_master.at(node.cell_master);
    if (node.kind == FastStaNodeKind::kBufferInput) {
      node.input_cap_pf = liberty_cell.input_cap_pf;
      node.max_slew_ns = liberty_cell.input_slew_limit_ns;
    }
  }
  for (auto& net : context.nets) {
    if (environment.max_cap_pf.has_value() && *environment.max_cap_pf > 0.0) {
      net.max_cap_pf = *environment.max_cap_pf;
      continue;
    }
    if (isSourceBoundaryNet(clock, context, net)) {
      const double source_cap_limit_pf = queryWrapperBackedSourceCapLimit(environment, clock);
      if (source_cap_limit_pf > 0.0) {
        net.max_cap_pf = source_cap_limit_pf;
        continue;
      }
    }
    if (net.driver_node_id == kInvalidFastStaNodeId) {
      continue;
    }
    const auto& driver = context.nodes.at(net.driver_node_id);
    if (const auto iter = context.liberty_cell_by_master.find(driver.cell_master); iter != context.liberty_cell_by_master.end()) {
      net.max_cap_pf = iter->second.output_cap_limit_pf;
    }
  }
  FastStaParasitics::updateNetLoads(context);
}

auto collectSinkPinCaps(const FastStaEnvironment& environment, const Clock& clock, FastStaClockContext& context) -> void
{
  auto& wrapper = *environment.wrapper;
  for (auto* pin : clock.get_loads()) {
    if (pin == nullptr) {
      continue;
    }
    const auto node_iter = context.node_id_by_name.find(Design::getPinFullName(pin));
    if (node_iter == context.node_id_by_name.end() || node_iter->second >= context.nodes.size()) {
      continue;
    }
    auto& node = context.nodes.at(node_iter->second);
    node.input_cap_pf = queryWrapperBackedSinkPinCap(wrapper, pin);
    node.max_slew_ns = queryWrapperBackedSinkSlewLimit(environment, pin);
  }
}

}  // namespace

auto FastStaBuilder::buildClockContext(const FastStaEnvironment& environment, const FastStaClockBuildInput& input) -> FastStaClockContext
{
  LOG_FATAL_IF(input.clock == nullptr) << "FastStaBuilder: clock build input is null.";
  const auto& clock = *input.clock;
  const auto total_start = std::chrono::steady_clock::now();
  auto stage_start = std::chrono::steady_clock::now();
  auto context = input.route_geometry == nullptr ? FastStaClockTree::buildFromClock(clock)
                                                 : FastStaClockTree::buildFromClockRouteGeometry(clock, *input.route_geometry);
  logBuilderStage(clock, input.route_geometry == nullptr ? "build committed clock-tree graph" : "build clock route geometry graph",
                  elapsedSeconds(stage_start), context);

  stage_start = std::chrono::steady_clock::now();
  applyEnvironment(environment, context);
  logBuilderStage(clock, "apply bound environment", elapsedSeconds(stage_start), context);

  if (input.route_geometry != nullptr) {
    stage_start = std::chrono::steady_clock::now();
    FastStaClockTree::applyRouteGeometry(context, *input.route_geometry);
    logBuilderStage(clock, "apply route geometry parasitics", elapsedSeconds(stage_start), context);
  }

  stage_start = std::chrono::steady_clock::now();
  collectSinkPinCaps(environment, clock, context);
  logBuilderStage(clock, "collect sink pin caps", elapsedSeconds(stage_start), context);

  stage_start = std::chrono::steady_clock::now();
  collectClockCellTimingData(environment, clock, context);
  logBuilderStage(clock, "collect Liberty and clock timing records", elapsedSeconds(stage_start), context);
  LOG_INFO << "FastStaBuilder: build clock context for clock \"" << clock.get_clock_name() << "\" finished in "
           << elapsedSeconds(total_start) << " s.";
  return context;
}

auto FastStaBuilder::injectNetRouteTree(FastStaClockContext& context, const Net& net, const ClockSteinerTree<int>& route_tree) -> bool
{
  const auto net_iter = context.net_id_by_name.find(net.get_name());
  if (net_iter == context.net_id_by_name.end() || net_iter->second >= context.nets.size()) {
    return false;
  }
  return FastStaParasitics::buildNetParasiticFromRouteTree(context, net_iter->second, net, route_tree);
}

}  // namespace icts
