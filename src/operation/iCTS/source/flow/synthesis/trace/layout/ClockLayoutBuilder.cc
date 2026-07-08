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
 * @file ClockLayoutBuilder.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-29
 * @brief CTS clock layout projection construction helper implementation.
 */

#include "synthesis/trace/layout/ClockLayoutBuilder.hh"

#include <optional>
#include <string>
#include <unordered_set>
#include <utility>

#include "ClockLayout.hh"
#include "Point.hh"
#include "SteinerTree.hh"
#include "design/Clock.hh"
#include "design/Inst.hh"
#include "design/Net.hh"
#include "design/Pin.hh"
#include "router/Router.hh"
#include "synthesis/trace/layout/ClockLayoutSynthesisTopology.hh"

namespace icts {
namespace {

enum class ClockLayoutSegmentSource
{
  kRoutedTree,
  kFlylinePins,
  kDegradedPins
};

auto isValidLocation(const Point<int>& point) -> bool
{
  return point.get_x() >= 0 && point.get_y() >= 0;
}

auto makeSegment(const Clock& clock, std::size_t clock_index, const Net& net, const Point<int>& begin, const Point<int>& end,
                 LayoutNetRole role, SinkDomainKind sink_domain, ClockLayoutPhase synthesis_phase, int selected_depth, int topology_level,
                 bool routed, bool degraded) -> ClockLayoutSegment
{
  return ClockLayoutSegment{
      .clock_name = clock.get_clock_name(),
      .net_name = net.get_name(),
      .begin = begin,
      .end = end,
      .net_role = role,
      .sink_domain = sink_domain,
      .synthesis_phase = synthesis_phase,
      .clock_index = clock_index,
      .topology_depth = selected_depth,
      .topology_level = topology_level,
      .routed = routed,
      .degraded = degraded,
  };
}

auto appendPinToPinSegments(const Clock& clock, std::size_t clock_index, const Net& net, LayoutNetRole role, SinkDomainKind sink_domain,
                            ClockLayoutPhase synthesis_phase, ClockLayoutNet& layout_net, ClockLayoutSegmentSource segment_source) -> void
{
  auto* driver = net.get_driver();
  if (driver == nullptr || !isValidLocation(driver->get_location())) {
    return;
  }

  const bool degraded = segment_source == ClockLayoutSegmentSource::kDegradedPins;
  const auto driver_location = driver->get_location();
  for (auto* load : net.get_loads()) {
    if (load == nullptr || !isValidLocation(load->get_location())) {
      continue;
    }
    auto segment = makeSegment(clock, clock_index, net, driver_location, load->get_location(), role, sink_domain, synthesis_phase,
                               layout_net.selected_depth, layout_net.topology_level, false, degraded);
    if (segment_source == ClockLayoutSegmentSource::kFlylinePins) {
      layout_net.flyline_segments.push_back(std::move(segment));
    } else {
      layout_net.routed_segments.push_back(std::move(segment));
    }
  }
}

auto appendClockTreeNetSegments(const Clock& clock, std::size_t clock_index, const Net& net, LayoutNetRole role, SinkDomainKind sink_domain,
                                ClockLayoutPhase synthesis_phase, ClockLayoutNet& layout_net, ClockLayoutSegmentSource segment_source)
    -> void
{
  if (segment_source != ClockLayoutSegmentSource::kRoutedTree) {
    appendPinToPinSegments(clock, clock_index, net, role, sink_domain, synthesis_phase, layout_net, segment_source);
    return;
  }

  auto route_tree = Router::buildClockNetTree(net);
  if (route_tree.node_count() == 0U || route_tree.edge_count() == 0U) {
    appendPinToPinSegments(clock, clock_index, net, role, sink_domain, synthesis_phase, layout_net,
                           ClockLayoutSegmentSource::kDegradedPins);
    return;
  }

  bool appended_routed_segment = false;
  for (const auto& edge : route_tree.get_edges()) {
    const auto* source = route_tree.get_node(edge.source_node_id);
    const auto* target = route_tree.get_node(edge.target_node_id);
    if (source == nullptr || target == nullptr || !isValidLocation(source->location) || !isValidLocation(target->location)) {
      continue;
    }
    layout_net.routed_segments.push_back(makeSegment(clock, clock_index, net, source->location, target->location, role, sink_domain,
                                                     synthesis_phase, layout_net.selected_depth, layout_net.topology_level, true, false));
    appended_routed_segment = true;
  }

  if (!appended_routed_segment) {
    appendPinToPinSegments(clock, clock_index, net, role, sink_domain, synthesis_phase, layout_net,
                           ClockLayoutSegmentSource::kDegradedPins);
  }
}

auto makeLayoutNet(const Clock& clock, std::size_t clock_index, const Net& net, LayoutNetRole role, SinkDomainKind sink_domain,
                   ClockLayoutPhase synthesis_phase, int selected_depth, int topology_level_count, int topology_level) -> ClockLayoutNet
{
  ClockLayoutNet layout_net{
      .clock_name = clock.get_clock_name(),
      .net_name = net.get_name(),
      .role = role,
      .sink_domain = sink_domain,
      .synthesis_phase = synthesis_phase,
      .clock_index = clock_index,
      .selected_depth = selected_depth,
      .topology_level = topology_level,
      .topology_level_count = topology_level_count,
      .routed_segments = {},
      .flyline_segments = {},
  };
  appendClockTreeNetSegments(clock, clock_index, net, role, sink_domain, synthesis_phase, layout_net,
                             ClockLayoutSegmentSource::kRoutedTree);
  appendClockTreeNetSegments(clock, clock_index, net, role, sink_domain, synthesis_phase, layout_net,
                             ClockLayoutSegmentSource::kFlylinePins);
  return layout_net;
}

auto makeLayoutInst(const Clock& clock, std::size_t clock_index, const Inst& inst, LayoutInstRole role, SinkDomainKind sink_domain,
                    ClockLayoutPhase synthesis_phase, int selected_depth, int topology_level) -> ClockLayoutInst
{
  return ClockLayoutInst{
      .clock_name = clock.get_clock_name(),
      .inst_name = inst.get_name(),
      .cell_master = inst.get_cell_master(),
      .origin = inst.get_location(),
      .width_dbu = 0,
      .height_dbu = 0,
      .role = role,
      .sink_domain = sink_domain,
      .synthesis_phase = synthesis_phase,
      .clock_index = clock_index,
      .topology_depth = selected_depth,
      .topology_level = topology_level,
  };
}

auto makeSinkLayoutInst(const Clock& clock, std::size_t clock_index, const Pin& sink, SinkDomainKind sink_domain)
    -> std::optional<ClockLayoutInst>
{
  auto* inst = sink.get_inst();
  if (inst == nullptr) {
    return std::nullopt;
  }

  auto layout_inst
      = makeLayoutInst(clock, clock_index, *inst, LayoutInstRole::kClockLoad, sink_domain, ClockLayoutPhase::kReadData, -1, -1);
  if (!isValidLocation(layout_inst.origin) && isValidLocation(sink.get_location())) {
    layout_inst.origin = sink.get_location();
  }
  return layout_inst;
}

auto inferNetTopologyLevel(LayoutNetRole role, ClockLayoutPhase synthesis_phase, int selected_depth) -> int
{
  if (role == LayoutNetRole::kSinkTree) {
    return selected_depth >= 0 ? selected_depth : 0;
  }
  if (role == LayoutNetRole::kSourceToRoot && synthesis_phase == ClockLayoutPhase::kSourceToRootHTree) {
    return selected_depth >= 0 ? selected_depth : 0;
  }
  return 0;
}

}  // namespace

auto ClockLayoutBuilder::appendSinkInsts(ClockLayout& clock_layout, const Clock& clock, std::size_t clock_index,
                                         const std::vector<Pin*>& sinks, SinkDomainKind sink_domain) -> void
{
  std::unordered_set<const Inst*> seen_insts;
  for (const auto* sink : sinks) {
    if (sink == nullptr || sink->get_inst() == nullptr || seen_insts.contains(sink->get_inst())) {
      continue;
    }
    auto sink_layout_inst = makeSinkLayoutInst(clock, clock_index, *sink, sink_domain);
    if (!sink_layout_inst.has_value()) {
      continue;
    }
    seen_insts.insert(sink->get_inst());
    clock_layout.addInst(*sink_layout_inst);
  }
}

auto ClockLayoutBuilder::appendDirectSinkDomain(ClockLayout& clock_layout, const Clock& clock, std::size_t clock_index,
                                                const SinkDomainLayoutAnchor& sink_domain_anchor) -> void
{
  if (sink_domain_anchor.root_buffer != nullptr) {
    clock_layout.addInst(makeLayoutInst(clock, clock_index, *sink_domain_anchor.root_buffer, LayoutInstRole::kRootBuffer,
                                        sink_domain_anchor.sink_domain, ClockLayoutPhase::kDownstreamHTree, -1, -1));
  }
  if (sink_domain_anchor.downstream_net != nullptr) {
    clock_layout.addNet(makeLayoutNet(clock, clock_index, *sink_domain_anchor.downstream_net, LayoutNetRole::kDownstream,
                                      sink_domain_anchor.sink_domain, ClockLayoutPhase::kDownstreamHTree, -1, 0,
                                      inferNetTopologyLevel(LayoutNetRole::kDownstream, ClockLayoutPhase::kDownstreamHTree, -1)));
  }
}

auto ClockLayoutBuilder::makeSinkDomainLayout(const Clock& clock, std::size_t clock_index, const SinkDomainLayoutAnchor& sink_domain_anchor,
                                              const SinkDomainSynthesisTopology& layout_topology) -> ClockLayout
{
  ClockLayout clock_layout;
  if (sink_domain_anchor.root_buffer != nullptr) {
    clock_layout.addInst(makeLayoutInst(clock, clock_index, *sink_domain_anchor.root_buffer, LayoutInstRole::kRootBuffer,
                                        sink_domain_anchor.sink_domain, ClockLayoutPhase::kDownstreamHTree, layout_topology.selected_depth,
                                        -1));
  }
  if (sink_domain_anchor.downstream_net != nullptr) {
    clock_layout.addNet(makeLayoutNet(
        clock, clock_index, *sink_domain_anchor.downstream_net, LayoutNetRole::kDownstream, sink_domain_anchor.sink_domain,
        ClockLayoutPhase::kDownstreamHTree, layout_topology.selected_depth, layout_topology.topology_level_count,
        inferNetTopologyLevel(LayoutNetRole::kDownstream, ClockLayoutPhase::kDownstreamHTree, layout_topology.selected_depth)));
  }
  for (const auto& inst : layout_topology.inserted_insts) {
    if (inst.inst != nullptr) {
      clock_layout.addInst(makeLayoutInst(clock, clock_index, *inst.inst, LayoutInstRole::kHTreeBuffer, sink_domain_anchor.sink_domain,
                                          ClockLayoutPhase::kDownstreamHTree, layout_topology.selected_depth, inst.topology_level));
    }
  }
  for (const auto& net : layout_topology.inserted_nets) {
    if (net.net != nullptr) {
      clock_layout.addNet(makeLayoutNet(clock, clock_index, *net.net, LayoutNetRole::kSinkTree, sink_domain_anchor.sink_domain,
                                        ClockLayoutPhase::kDownstreamHTree, layout_topology.selected_depth,
                                        layout_topology.topology_level_count, net.topology_level));
    }
  }
  return clock_layout;
}

auto ClockLayoutBuilder::makeSourceToRootLayout(const Clock& clock, std::size_t clock_index, const Net& source_net,
                                                const SourceToRootSynthesisTopology& layout_topology, ClockLayoutPhase synthesis_phase)
    -> ClockLayout
{
  ClockLayout clock_layout;
  clock_layout.addNet(makeLayoutNet(clock, clock_index, source_net, LayoutNetRole::kSourceToRoot, SinkDomainKind::kSourceToRoot,
                                    synthesis_phase, layout_topology.selected_depth, layout_topology.topology_level_count, 0));
  for (const auto& inst : layout_topology.inserted_insts) {
    if (inst.inst != nullptr) {
      clock_layout.addInst(makeLayoutInst(clock, clock_index, *inst.inst, LayoutInstRole::kSourceRootBuffer, SinkDomainKind::kSourceToRoot,
                                          synthesis_phase, layout_topology.selected_depth, inst.topology_level));
    }
  }
  for (const auto& net : layout_topology.inserted_nets) {
    if (net.net != nullptr) {
      clock_layout.addNet(makeLayoutNet(clock, clock_index, *net.net, LayoutNetRole::kSourceToRoot, SinkDomainKind::kSourceToRoot,
                                        synthesis_phase, layout_topology.selected_depth, layout_topology.topology_level_count,
                                        net.topology_level));
    }
  }
  return clock_layout;
}

auto ClockLayoutBuilder::merge(ClockLayout& target, const ClockLayout& source) -> void
{
  for (const auto& source_clock : source.get_clocks()) {
    auto& target_clock = target.ensureClock(source_clock.clock_name, source_clock.clock_net_name, source_clock.clock_index);
    target_clock.nets.insert(target_clock.nets.end(), source_clock.nets.begin(), source_clock.nets.end());
    target_clock.insts.insert(target_clock.insts.end(), source_clock.insts.begin(), source_clock.insts.end());
  }
}

}  // namespace icts
