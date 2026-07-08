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
 * @file OptimizationPreparation.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief Preparation helpers for CTS post-synthesis optimization.
 */

#include "optimization/preparation/OptimizationPreparation.hh"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <compare>
#include <cstddef>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "FastSta.hh"
#include "Log.hh"
#include "Point.hh"
#include "SteinerTree.hh"
#include "design/Clock.hh"
#include "design/ClockDAG.hh"
#include "design/ClockLayout.hh"
#include "design/Design.hh"
#include "design/Inst.hh"
#include "design/Net.hh"
#include "io/Wrapper.hh"
#include "optimization/model/ClockSizingOptimizationData.hh"
#include "optimization/policy/OptimizationPolicy.hh"
#include "optimization/report/OptimizationReport.hh"
#include "router/Router.hh"

namespace icts::clock_sizing_optimization {

auto BuildClockRouteGeometry(const ClockLayout& clock_layout, std::size_t clock_index) -> FastStaClockRouteGeometry
{
  FastStaClockRouteGeometry route_geometry;
  route_geometry.design_dbu_per_um = clock_layout.get_design_dbu_per_um();
  const auto* layout_clock = clock_layout.findClock(clock_index);
  if (layout_clock == nullptr) {
    return route_geometry;
  }
  route_geometry.clock_nets.reserve(layout_clock->nets.size());
  for (const auto& layout_net : layout_clock->nets) {
    FastStaClockNetRouteGeometry clock_net_route;
    clock_net_route.net_name = layout_net.net_name;
    clock_net_route.routed_segments.reserve(layout_net.routed_segments.size());
    for (const auto& segment : layout_net.routed_segments) {
      if (!segment.routed || segment.degraded) {
        continue;
      }
      clock_net_route.routed_segments.push_back(FastStaRcSegment{
          .begin = FastStaPoint{.x_dbu = segment.begin.get_x(), .y_dbu = segment.begin.get_y()},
          .end = FastStaPoint{.x_dbu = segment.end.get_x(), .y_dbu = segment.end.get_y()},
      });
    }
    if (!clock_net_route.routed_segments.empty()) {
      route_geometry.clock_nets.push_back(std::move(clock_net_route));
    }
  }
  return route_geometry;
}

auto CaptureGraphProfile(const FastSTA& fast_sta, FastStaClockId clock_id) -> ClockSizingRuntimeProfile
{
  ClockSizingRuntimeProfile profile;
  const auto graph_profile = fast_sta.queryClockGraphProfile(clock_id);
  if (!graph_profile.has_value()) {
    return profile;
  }
  profile.node_count = graph_profile->node_count;
  profile.net_count = graph_profile->net_count;
  profile.sink_count = graph_profile->sink_count;
  profile.buffer_input_count = graph_profile->buffer_input_count;
  profile.buffer_output_count = graph_profile->buffer_output_count;
  return profile;
}

auto CopyOuterProfile(ClockSizingRuntimeProfile& destination, const ClockSizingRuntimeProfile& source) -> void
{
  destination.build_route_tree_cache_s = source.build_route_tree_cache_s;
  destination.build_fast_sta_context_s = source.build_fast_sta_context_s;
  destination.inject_route_trees_s = source.inject_route_trees_s;
  destination.collect_optimizable_buffers_s = source.collect_optimizable_buffers_s;
  destination.collect_cap_baseline_s = source.collect_cap_baseline_s;
  destination.collect_slew_baseline_s = source.collect_slew_baseline_s;
  destination.solve_clock_s = source.solve_clock_s;
  destination.apply_accepted_edits_s = source.apply_accepted_edits_s;
  destination.node_count = source.node_count;
  destination.net_count = source.net_count;
  destination.sink_count = source.sink_count;
  destination.buffer_input_count = source.buffer_input_count;
  destination.buffer_output_count = source.buffer_output_count;
  destination.optimizable_buffer_count = source.optimizable_buffer_count;
}

namespace {

auto ResolveOutputCapLimit(Wrapper& wrapper, const std::string& cell_master) -> double
{
  double max_cap_pf = wrapper.queryCellOutPinCapLimit(cell_master);
  if (max_cap_pf <= 0.0) {
    max_cap_pf = wrapper.queryCellOutPinCapTableAxisMax(cell_master);
  }
  return max_cap_pf;
}

}  // namespace

auto CollectClockSizingBufferMasters(const ClockSizingMasterQueryInput& input) -> std::vector<ClockSizingBufferMaster>
{
  LOG_FATAL_IF(input.wrapper == nullptr) << "Optimization: buffer master query requires Wrapper.";
  LOG_FATAL_IF(input.buffer_cell_masters == nullptr) << "Optimization: buffer master query requires candidate masters.";
  auto& wrapper = *input.wrapper;
  std::vector<ClockSizingBufferMaster> infos;
  infos.reserve(input.buffer_cell_masters->size());
  for (const auto& cell_master : *input.buffer_cell_masters) {
    if (cell_master.empty()) {
      continue;
    }
    const double output_cap_limit_pf = ResolveOutputCapLimit(wrapper, cell_master);
    const double input_cap_pf = wrapper.queryCharInputPinCap(cell_master);
    if (output_cap_limit_pf <= 0.0 || input_cap_pf <= 0.0) {
      continue;
    }
    infos.push_back(ClockSizingBufferMaster{.cell_master = cell_master,
                                            .input_cap_pf = input_cap_pf,
                                            .output_cap_limit_pf = output_cap_limit_pf,
                                            .area_um2 = std::max(0.0, wrapper.queryCellAreaUm2(cell_master)),
                                            .drive_rank = 0U});
  }

  std::ranges::sort(infos, [](const ClockSizingBufferMaster& lhs, const ClockSizingBufferMaster& rhs) -> bool {
    if (std::abs(lhs.output_cap_limit_pf - rhs.output_cap_limit_pf) > kClockSizingEpsilon) {
      return lhs.output_cap_limit_pf < rhs.output_cap_limit_pf;
    }
    if (std::abs(lhs.area_um2 - rhs.area_um2) > kClockSizingEpsilon) {
      return lhs.area_um2 < rhs.area_um2;
    }
    return lhs.cell_master < rhs.cell_master;
  });
  for (std::size_t index = 0U; index < infos.size(); ++index) {
    infos.at(index).drive_rank = static_cast<unsigned>(index);
  }
  return infos;
}

auto BuildClockSizingRouteTrees(const Design& design, const std::vector<Clock*>& clocks) -> ClockSizingRouteTreeCache
{
  ClockSizingRouteTreeCache route_tree_by_net;
  const auto& clock_dag = design.get_clock_dag();
  for (auto* clock : clocks) {
    if (clock == nullptr) {
      continue;
    }
    const auto* graph = clock_dag.graphForClock(clock);
    if (graph == nullptr) {
      continue;
    }
    for (auto* net : graph->nets) {
      if (net == nullptr || route_tree_by_net.contains(net)) {
        continue;
      }
      route_tree_by_net.emplace(net, Router::buildClockNetTree(*net));
    }
  }
  return route_tree_by_net;
}

auto FindMasterInfo(const std::vector<ClockSizingBufferMaster>& master_infos, std::string_view cell_master)
    -> const ClockSizingBufferMaster*
{
  const auto iter = std::ranges::find_if(
      master_infos, [cell_master](const ClockSizingBufferMaster& info) -> bool { return info.cell_master == cell_master; });
  return iter == master_infos.end() ? nullptr : &(*iter);
}

auto CollectClockSizingCapLimits(const FastSTA& fast_sta, FastStaClockId clock_id) -> std::vector<ClockSizingCapLimit>
{
  std::vector<ClockSizingCapLimit> baseline;
  const auto graph_profile = fast_sta.queryClockGraphProfile(clock_id);
  if (!graph_profile.has_value()) {
    return baseline;
  }
  baseline.reserve(graph_profile->net_count);
  for (FastStaNetId net_id = 0U; net_id < graph_profile->net_count; ++net_id) {
    const auto cap_status = fast_sta.queryCapStatus(clock_id, net_id);
    baseline.push_back(ClockSizingCapLimit{.load_cap_pf = cap_status.has_value() ? cap_status->load_cap_pf : 0.0,
                                           .max_cap_pf = cap_status.has_value() ? cap_status->max_cap_pf : 0.0,
                                           .violated = cap_status.has_value() && cap_status->violated});
  }
  return baseline;
}

auto CollectClockSizingSlewLimits(const FastSTA& fast_sta, FastStaClockId clock_id) -> std::vector<ClockSizingSlewLimit>
{
  std::vector<ClockSizingSlewLimit> baseline;
  const auto graph_profile = fast_sta.queryClockGraphProfile(clock_id);
  if (!graph_profile.has_value()) {
    return baseline;
  }
  baseline.reserve(graph_profile->node_count);
  for (FastStaNodeId node_id = 0U; node_id < graph_profile->node_count; ++node_id) {
    const auto slew_status = fast_sta.querySlewStatus(clock_id, node_id);
    baseline.push_back(ClockSizingSlewLimit{.slew_ns = slew_status.has_value() ? slew_status->slew_ns : 0.0,
                                            .max_slew_ns = slew_status.has_value() ? slew_status->max_slew_ns : 0.0,
                                            .role = slew_status.has_value() ? slew_status->role : FastStaSlewRole::kUnknown,
                                            .available = slew_status.has_value(),
                                            .violated = slew_status.has_value() && slew_status->violated});
  }
  return baseline;
}

auto CollectClockSizingBuffers(const Design& design, const FastSTA& fast_sta, FastStaClockId clock_id,
                               const std::vector<ClockSizingBufferMaster>& master_infos) -> std::vector<ClockSizingBuffer>
{
  std::vector<ClockSizingBuffer> buffers;
  if (master_infos.empty()) {
    return buffers;
  }
  const auto fast_sta_buffers = fast_sta.collectClockSizingBuffers(clock_id);
  buffers.reserve(fast_sta_buffers.size());
  for (const auto& fast_sta_buffer : fast_sta_buffers) {
    auto* inst = design.findInst(fast_sta_buffer.inst_name);
    if (inst == nullptr || !inst->is_buffer() || FindMasterInfo(master_infos, fast_sta_buffer.cell_master) == nullptr) {
      continue;
    }
    buffers.push_back(ClockSizingBuffer{.node_id = fast_sta_buffer.node_id,
                                        .inst = inst,
                                        .inst_name = fast_sta_buffer.inst_name,
                                        .current_master = fast_sta_buffer.cell_master,
                                        .candidates = master_infos});
  }
  return buffers;
}

auto InjectRouteTrees(const Design& design, FastSTA& fast_sta, FastStaClockId clock_id, const Clock& clock,
                      const ClockSizingRouteTreeCache& route_tree_by_net) -> bool
{
  const auto* graph = design.get_clock_dag().graphForClock(&clock);
  if (graph == nullptr) {
    return false;
  }
  const auto total_start = std::chrono::steady_clock::now();
  auto progress_start = total_start;
  const auto progress_interval = DefaultOptimizationPolicy().route_tree_progress_interval;
  const auto initial_detail_net_count = DefaultOptimizationPolicy().route_tree_initial_detail_net_count;
  const auto slow_net_threshold_s = DefaultOptimizationPolicy().route_tree_slow_net_threshold_s;
  std::size_t visited_net_count = 0U;
  std::size_t injected_net_count = 0U;
  std::size_t rc_node_count = 0U;
  std::size_t rc_edge_count = 0U;
  LOG_INFO << "Optimization: start route-tree injection for clock \"" << clock.get_clock_name() << "\", dag_nets=" << graph->nets.size()
           << ", cached_route_trees=" << route_tree_by_net.size() << ".";
  for (auto* net : graph->nets) {
    if (net == nullptr) {
      continue;
    }
    ++visited_net_count;
    const auto route_iter = route_tree_by_net.find(net);
    if (route_iter == route_tree_by_net.end()) {
      LOG_ERROR << "Optimization: route tree is unavailable for net \"" << net->get_name() << "\".";
      return false;
    }
    if (visited_net_count <= initial_detail_net_count) {
      LOG_INFO << "Optimization: route-tree injection start net " << visited_net_count << "/" << graph->nets.size() << " \""
               << net->get_name() << "\", loads=" << net->get_loads().size() << ", route_nodes=" << route_iter->second.node_count()
               << ", route_edges=" << route_iter->second.edge_count() << ".";
    }
    const auto net_start = std::chrono::steady_clock::now();
    FastStaClockNetRcTreeCounts rc_tree_counts;
    if (!fast_sta.injectNetRouteTree(clock_id, *net, route_iter->second, rc_tree_counts)) {
      LOG_ERROR << "Optimization: fast STA route-tree injection failed for net \"" << net->get_name() << "\".";
      return false;
    }
    const double net_runtime_s = ElapsedSeconds(net_start);
    ++injected_net_count;
    rc_node_count += rc_tree_counts.rc_node_count;
    rc_edge_count += rc_tree_counts.rc_edge_count;
    if (visited_net_count <= initial_detail_net_count || net_runtime_s >= slow_net_threshold_s) {
      LOG_INFO << "Optimization: route-tree injection finish net " << visited_net_count << "/" << graph->nets.size() << " \""
               << net->get_name() << "\" in " << net_runtime_s << " s.";
    }
    if (injected_net_count % progress_interval == 0U) {
      LOG_INFO << "Optimization: route-tree injection progress for clock \"" << clock.get_clock_name()
               << "\": injected=" << injected_net_count << "/" << graph->nets.size() << ", visited=" << visited_net_count
               << ", elapsed=" << ElapsedSeconds(total_start) << " s, interval=" << ElapsedSeconds(progress_start) << " s.";
      progress_start = std::chrono::steady_clock::now();
    }
  }
  LOG_INFO << "Optimization: route-tree injection finished for clock \"" << clock.get_clock_name() << "\" in "
           << ElapsedSeconds(total_start) << " s, injected_nets=" << injected_net_count << ", rc_nodes=" << rc_node_count
           << ", rc_edges=" << rc_edge_count << ".";

  auto update_start = std::chrono::steady_clock::now();
  LOG_INFO << "Optimization: start post-injection timing update for clock \"" << clock.get_clock_name() << "\".";
  if (!fast_sta.updateTiming(clock_id)) {
    return false;
  }
  LOG_INFO << "Optimization: post-injection timing update for clock \"" << clock.get_clock_name() << "\" finished in "
           << ElapsedSeconds(update_start) << " s.";

  update_start = std::chrono::steady_clock::now();
  LOG_INFO << "Optimization: start post-injection power update for clock \"" << clock.get_clock_name() << "\".";
  if (!fast_sta.updatePower(clock_id)) {
    return false;
  }
  LOG_INFO << "Optimization: post-injection power update for clock \"" << clock.get_clock_name() << "\" finished in "
           << ElapsedSeconds(update_start) << " s.";
  return true;
}

}  // namespace icts::clock_sizing_optimization
