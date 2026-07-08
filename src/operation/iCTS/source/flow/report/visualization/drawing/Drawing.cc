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
 * @file Drawing.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-29
 * @brief Normalized CTS clock-tree visualization model implementation.
 */

#include "report/visualization/drawing/Drawing.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <ostream>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ClockLayout.hh"
#include "Log.hh"
#include "Point.hh"
#include "SteinerTree.hh"
#include "design/Clock.hh"
#include "design/ClockDAG.hh"
#include "design/Design.hh"
#include "design/Inst.hh"
#include "design/Net.hh"
#include "design/Pin.hh"
#include "io/Wrapper.hh"
#include "router/Router.hh"

namespace icts {
namespace {

auto isValidLocation(const Point<int>& point) -> bool
{
  return point.get_x() >= 0 && point.get_y() >= 0;
}

auto makeSegment(const ClockLayoutClock& clock, const ClockLayoutSegment& segment, ClockLayoutMode view) -> DrawingSegment
{
  return DrawingSegment{
      .clock_name = clock.clock_name,
      .net_name = segment.net_name,
      .begin = segment.begin,
      .end = segment.end,
      .view = view,
      .net_role = segment.net_role,
      .sink_domain = segment.sink_domain,
      .synthesis_phase = segment.synthesis_phase,
      .clock_index = clock.clock_index,
      .topology_depth = segment.topology_depth,
      .topology_level = segment.topology_level,
      .routed = segment.routed,
      .degraded = segment.degraded,
  };
}

auto appendViewSegments(const ClockLayout& clock_layout, Drawing& model) -> void
{
  for (const auto& clock : clock_layout.get_clocks()) {
    for (const auto& net : clock.nets) {
      for (const auto& segment : net.routed_segments) {
        if (isValidLocation(segment.begin) && isValidLocation(segment.end)) {
          model.design_segments.push_back(makeSegment(clock, segment, ClockLayoutMode::kDesign));
        }
      }
      for (const auto& segment : net.flyline_segments) {
        if (isValidLocation(segment.begin) && isValidLocation(segment.end)) {
          model.flyline_segments.push_back(makeSegment(clock, segment, ClockLayoutMode::kFlyline));
        }
      }
    }
  }
}

auto appendViewInsts(const ClockLayout& clock_layout, Drawing& model) -> void
{
  for (const auto& clock : clock_layout.get_clocks()) {
    for (const auto& inst : clock.insts) {
      model.insts.push_back(DrawingInst{
          .clock_name = clock.clock_name,
          .inst_name = inst.inst_name,
          .cell_master = inst.cell_master,
          .origin = inst.origin,
          .width_dbu = inst.width_dbu,
          .height_dbu = inst.height_dbu,
          .role = inst.role,
          .sink_domain = inst.sink_domain,
          .synthesis_phase = inst.synthesis_phase,
          .clock_index = clock.clock_index,
          .topology_depth = inst.topology_depth,
          .topology_level = inst.topology_level,
      });
    }
  }
}

auto appendDegradedRouteTreeSegments(const Clock& clock, std::size_t clock_index, const Net& net, LayoutNetRole role,
                                     std::vector<DrawingSegment>& segments) -> bool
{
  auto route_tree = Router::buildClockNetTree(net);
  if (route_tree.node_count() == 0U || route_tree.edge_count() == 0U) {
    return false;
  }

  bool appended = false;
  for (const auto& edge : route_tree.get_edges()) {
    const auto* source = route_tree.get_node(edge.source_node_id);
    const auto* target = route_tree.get_node(edge.target_node_id);
    if (source == nullptr || target == nullptr || !isValidLocation(source->location) || !isValidLocation(target->location)) {
      continue;
    }
    segments.push_back(DrawingSegment{
        .clock_name = clock.get_clock_name(),
        .net_name = net.get_name(),
        .begin = source->location,
        .end = target->location,
        .view = ClockLayoutMode::kDesign,
        .net_role = role,
        .sink_domain = role == LayoutNetRole::kSourceToRoot ? SinkDomainKind::kSourceToRoot : SinkDomainKind::kUnknown,
        .synthesis_phase = ClockLayoutPhase::kUnknown,
        .clock_index = clock_index,
        .topology_depth = -1,
        .topology_level = 0,
        .routed = true,
        .degraded = false,
    });
    appended = true;
  }
  return appended;
}

auto appendDegradedPinSegments(const Clock& clock, std::size_t clock_index, const Net& net, LayoutNetRole role, ClockLayoutMode view,
                               std::vector<DrawingSegment>& segments) -> bool
{
  auto* driver = net.get_driver();
  if (driver == nullptr || !isValidLocation(driver->get_location())) {
    return false;
  }

  bool appended = false;
  const auto driver_location = driver->get_location();
  for (auto* load : net.get_loads()) {
    if (load == nullptr || !isValidLocation(load->get_location())) {
      continue;
    }
    segments.push_back(DrawingSegment{
        .clock_name = clock.get_clock_name(),
        .net_name = net.get_name(),
        .begin = driver_location,
        .end = load->get_location(),
        .view = view,
        .net_role = role,
        .sink_domain = role == LayoutNetRole::kSourceToRoot ? SinkDomainKind::kSourceToRoot : SinkDomainKind::kUnknown,
        .synthesis_phase = ClockLayoutPhase::kUnknown,
        .clock_index = clock_index,
        .topology_depth = -1,
        .topology_level = 0,
        .routed = false,
        .degraded = view == ClockLayoutMode::kDesign,
    });
    appended = true;
  }
  return appended;
}

auto collectClockNets(Design& design, const Clock& clock) -> std::vector<Net*>
{
  if (!design.get_clock_dag().is_built()) {
    (void) design.rebuildClockDAG();
  }
  const auto& clock_dag = design.get_clock_dag();
  auto reachable_nets = clock_dag.is_valid() ? clock_dag.reachableNets(&clock) : std::vector<Net*>{};
  if (!reachable_nets.empty()) {
    return reachable_nets;
  }

  std::vector<Net*> nets;
  std::unordered_set<Net*> seen_nets;
  auto append_net = [&nets, &seen_nets](Net* net) -> void {
    if (net == nullptr || seen_nets.contains(net)) {
      return;
    }
    seen_nets.insert(net);
    nets.push_back(net);
  };

  append_net(clock.get_clock_source_net());
  for (auto* net : clock.get_nets()) {
    append_net(net);
  }
  return nets;
}

auto appendDegradedSegments(Design& design, Drawing& model) -> void
{
  const auto clocks = design.get_clocks();
  for (std::size_t clock_index = 0U; clock_index < clocks.size(); ++clock_index) {
    const auto* clock = clocks.at(clock_index);
    if (clock == nullptr) {
      continue;
    }
    for (const auto* net : collectClockNets(design, *clock)) {
      if (net == nullptr) {
        continue;
      }
      const auto role = net == clock->get_clock_source_net() ? LayoutNetRole::kSourceToRoot : LayoutNetRole::kSinkTree;
      if (!appendDegradedRouteTreeSegments(*clock, clock_index, *net, role, model.design_segments)) {
        const bool appended = appendDegradedPinSegments(*clock, clock_index, *net, role, ClockLayoutMode::kDesign, model.design_segments);
        if (appended) {
          LOG_WARNING << "CTS visualization model: design view uses degraded driver-to-load segments for net " << net->get_name();
        }
      }
      (void) appendDegradedPinSegments(*clock, clock_index, *net, role, ClockLayoutMode::kFlyline, model.flyline_segments);
    }
  }
}

auto appendPinMarkers(Design& design, Drawing& model) -> void
{
  std::unordered_set<const Pin*> seen_pins;
  for (const auto* clock : design.get_clocks()) {
    if (clock == nullptr) {
      continue;
    }
    for (const auto* net : collectClockNets(design, *clock)) {
      if (net == nullptr) {
        continue;
      }
      if (const auto* driver = net->get_driver();
          driver != nullptr && isValidLocation(driver->get_location()) && seen_pins.insert(driver).second) {
        model.pins.push_back(DrawingPin{.location = driver->get_location(), .driver = true});
      }
      for (const auto* load : net->get_loads()) {
        if (load != nullptr && isValidLocation(load->get_location()) && seen_pins.insert(load).second) {
          model.pins.push_back(DrawingPin{.location = load->get_location(), .driver = false});
        }
      }
    }
  }
}

auto appendDegradedInsts(Design& design, Drawing& model) -> void
{
  if (!model.insts.empty()) {
    return;
  }

  std::unordered_set<const Inst*> seen_insts;
  const auto clocks = design.get_clocks();
  for (std::size_t clock_index = 0U; clock_index < clocks.size(); ++clock_index) {
    const auto* clock = clocks.at(clock_index);
    if (clock == nullptr) {
      continue;
    }
    for (const auto* inst : clock->get_insts()) {
      if (inst == nullptr || !seen_insts.insert(inst).second) {
        continue;
      }
      model.insts.push_back(DrawingInst{
          .clock_name = clock->get_clock_name(),
          .inst_name = inst->get_name(),
          .cell_master = inst->get_cell_master(),
          .origin = inst->get_location(),
          .width_dbu = 0,
          .height_dbu = 0,
          .role = LayoutInstRole::kHTreeBuffer,
          .sink_domain = SinkDomainKind::kUnknown,
          .synthesis_phase = ClockLayoutPhase::kUnknown,
          .clock_index = clock_index,
          .topology_depth = -1,
          .topology_level = -1,
      });
    }
  }
}

auto ctsInstNames(const Drawing& model) -> std::unordered_set<std::string>
{
  std::unordered_set<std::string> names;
  for (const auto& inst : model.insts) {
    if (!inst.inst_name.empty()) {
      names.insert(inst.inst_name);
    }
  }
  return names;
}

auto appendLogicCells(Wrapper& wrapper, Drawing& model) -> void
{
  const auto cts_names = ctsInstNames(model);
  for (const auto& geometry : wrapper.collectLogicCellGeometries()) {
    if (cts_names.contains(geometry.name) || !isValidLocation(geometry.origin)) {
      continue;
    }
    model.logic_cells.push_back(DrawingLogicCell{
        .inst_name = geometry.name,
        .origin = geometry.origin,
        .width_dbu = geometry.width_dbu,
        .height_dbu = geometry.height_dbu,
    });
  }
}

auto fillInstGeometry(Wrapper& wrapper, Drawing& model) -> void
{
  const int32_t dbu_per_um = std::max(model.dbu_per_um, int32_t{1});
  for (auto& inst : model.insts) {
    const auto geometry = wrapper.queryInstGeometry(inst.inst_name);
    if (geometry.has_value()) {
      if (isValidLocation(geometry->origin)) {
        inst.origin = geometry->origin;
      }
      inst.width_dbu = geometry->width_dbu > 0 ? geometry->width_dbu : inst.width_dbu;
      inst.height_dbu = geometry->height_dbu > 0 ? geometry->height_dbu : inst.height_dbu;
    }
    if (inst.width_dbu <= 0) {
      inst.width_dbu = std::max(dbu_per_um / 5, int32_t{1});
    }
    if (inst.height_dbu <= 0) {
      inst.height_dbu = std::max(dbu_per_um / 5, int32_t{1});
    }
  }
}

}  // namespace

auto DrawingBuilder::build(const DrawingInput& input) -> Drawing
{
  LOG_FATAL_IF(input.design == nullptr) << "DrawingBuilder: design is null.";
  LOG_FATAL_IF(input.wrapper == nullptr) << "DrawingBuilder: wrapper is null.";
  LOG_FATAL_IF(input.clock_layout == nullptr) << "DrawingBuilder: clock layout is null.";

  auto& design = *input.design;
  auto& wrapper = *input.wrapper;
  const auto& clock_layout = *input.clock_layout;
  Drawing model;
  model.has_clocks = !clock_layout.get_clocks().empty() || !design.get_clocks().empty();
  model.dbu_per_um = std::max(clock_layout.get_design_dbu_per_um(), int32_t{1});
  appendViewSegments(clock_layout, model);
  appendViewInsts(clock_layout, model);
  if (model.design_segments.empty() || model.flyline_segments.empty()) {
    appendDegradedSegments(design, model);
  }
  appendDegradedInsts(design, model);
  fillInstGeometry(wrapper, model);
  appendLogicCells(wrapper, model);
  appendPinMarkers(design, model);
  return model;
}

}  // namespace icts
