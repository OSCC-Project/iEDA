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
 * @file FastStaClockTree.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief CTS clock-tree graph construction helpers for fast STA.
 */

#include "FastStaClockTree.hh"

#include <glog/logging.h>

#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "FastSta.hh"
#include "FastStaClockState.hh"
#include "FastStaParasitics.hh"
#include "Log.hh"
#include "clock_net_parasitic/FastStaClockNetParasitic.hh"
#include "design/Clock.hh"
#include "design/Design.hh"
#include "design/Inst.hh"
#include "design/Net.hh"
#include "design/Pin.hh"
#include "spatial/Point.hh"

namespace icts {
namespace {

auto makeNodeName(const Pin* pin) -> std::string
{
  return Design::getPinFullName(pin);
}

auto makeLocationKey(const FastStaPoint& point) -> std::pair<int, int>
{
  return {point.x_dbu, point.y_dbu};
}

auto makeNodeKind(const Clock& clock, const Pin* pin) -> FastStaNodeKind
{
  if (pin == clock.get_clock_source()) {
    return FastStaNodeKind::kSource;
  }
  const auto* inst = pin != nullptr ? pin->get_inst() : nullptr;
  if (inst != nullptr && inst->is_clock_propagation_cell()) {
    return pin == inst->findDriverPin() ? FastStaNodeKind::kBufferOutput : FastStaNodeKind::kBufferInput;
  }
  return FastStaNodeKind::kSink;
}

auto appendPinNode(const Clock& clock, Pin* pin, FastStaClockContext& context) -> FastStaNodeId
{
  if (pin == nullptr) {
    return kInvalidFastStaNodeId;
  }
  const auto node_name = makeNodeName(pin);
  if (const auto iter = context.node_id_by_name.find(node_name); iter != context.node_id_by_name.end()) {
    return iter->second;
  }

  const auto* inst = pin->get_inst();
  const auto location = FastStaPoint{.x_dbu = pin->get_location().get_x(), .y_dbu = pin->get_location().get_y()};
  const auto node_id = context.nodes.size();
  context.node_id_by_name[node_name] = node_id;
  context.node_id_by_location.emplace(makeLocationKey(location), node_id);
  context.nodes.push_back(FastStaNode{
      .kind = makeNodeKind(clock, pin),
      .name = node_name,
      .inst_name = inst != nullptr ? inst->get_name() : std::string{},
      .pin_name = pin->get_name(),
      .cell_master = inst != nullptr ? inst->get_cell_master() : std::string{},
      .location = location,
      .output_net_ids = {},
      .timing = {},
  });
  return node_id;
}

auto appendRouteGeometry(const FastStaClockNetRouteGeometry& clock_net_route, FastStaClockContext& context) -> void
{
  const auto net_iter = context.net_id_by_name.find(clock_net_route.net_name);
  if (net_iter == context.net_id_by_name.end() || net_iter->second >= context.nets.size()) {
    return;
  }

  std::vector<FastStaRcSegment> segments;
  segments.reserve(clock_net_route.routed_segments.size());
  segments.insert(segments.end(), clock_net_route.routed_segments.begin(), clock_net_route.routed_segments.end());
  if (!segments.empty()) {
    context.nets.at(net_iter->second).parasitic.rc_nodes.clear();
    context.nets.at(net_iter->second).parasitic.rc_edges.clear();
    context.nets.at(net_iter->second).parasitic.rc_node_id_by_name.clear();
    context.nets.at(net_iter->second).parasitic.valid = false;
    (void) FastStaParasitics::buildNetParasiticFromSegments(context, net_iter->second, segments);
  }
}

}  // namespace

auto FastStaClockTree::buildFromClock(const Clock& clock) -> FastStaClockContext
{
  FastStaClockContext context;
  context.clock_name = clock.get_clock_name();
  context.clock_net_name = clock.get_clock_net_name();
  context.clock_period_ns = clock.get_clock_period_ns();
  context.source_node_id = appendPinNode(clock, clock.get_clock_source(), context);

  for (auto* pin : clock.get_loads()) {
    (void) appendPinNode(clock, pin, context);
  }
  for (auto* inst : clock.get_insts()) {
    if (inst == nullptr) {
      continue;
    }
    for (auto* pin : inst->get_pins()) {
      (void) appendPinNode(clock, pin, context);
    }
  }

  const auto append_net = [&](Net* net) -> void {
    if (net == nullptr || context.net_id_by_name.contains(net->get_name())) {
      return;
    }
    const auto net_id = context.nets.size();
    context.net_id_by_name[net->get_name()] = net_id;
    FastStaNet fast_net;
    fast_net.name = net->get_name();
    fast_net.driver_node_id = appendPinNode(clock, net->get_driver(), context);
    if (fast_net.driver_node_id != kInvalidFastStaNodeId) {
      context.nodes.at(fast_net.driver_node_id).output_net_ids.push_back(net_id);
    }
    for (auto* load : net->get_loads()) {
      const auto load_node_id = appendPinNode(clock, load, context);
      if (load_node_id == kInvalidFastStaNodeId) {
        continue;
      }
      fast_net.load_node_ids.push_back(load_node_id);
      context.nodes.at(load_node_id).incoming_net_id = net_id;
    }
    context.nets.push_back(std::move(fast_net));
  };

  append_net(clock.get_clock_source_net());
  for (auto* net : clock.get_nets()) {
    append_net(net);
  }
  return context;
}

auto FastStaClockTree::buildFromClockRouteGeometry(const Clock& clock, const FastStaClockRouteGeometry& route_geometry)
    -> FastStaClockContext
{
  auto context = buildFromClock(clock);
  context.dbu_per_um = route_geometry.design_dbu_per_um;
  return context;
}

auto FastStaClockTree::applyRouteGeometry(FastStaClockContext& context, const FastStaClockRouteGeometry& route_geometry) -> void
{
  const auto route_geometry_dbu_per_um = route_geometry.design_dbu_per_um;
  LOG_FATAL_IF(route_geometry_dbu_per_um <= 0) << "FastStaClockTree: clock route geometry DBU-per-micron is invalid.";
  LOG_FATAL_IF(context.dbu_per_um <= 0) << "FastStaClockTree: Fast STA DBU-per-micron is invalid.";
  LOG_FATAL_IF(context.dbu_per_um != route_geometry_dbu_per_um)
      << "FastStaClockTree: Fast STA DBU-per-micron (" << context.dbu_per_um << ") does not match clock route geometry DBU-per-micron ("
      << route_geometry_dbu_per_um << ").";
  LOG_FATAL_IF(context.routing_layer <= 0) << "FastStaClockTree: routing layer must be initialized before route-geometry RC extraction.";
  for (const auto& clock_net_route : route_geometry.clock_nets) {
    appendRouteGeometry(clock_net_route, context);
  }
}

}  // namespace icts
