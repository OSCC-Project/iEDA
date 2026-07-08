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
 * @file FastStaTiming.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief CTS fast STA timing propagation implementation.
 */

#include "FastStaTiming.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "FastStaClockState.hh"
#include "FastStaDmpCeff.hh"
#include "FastStaLibertyModel.hh"
#include "FastStaParasitics.hh"
#include "clock_net_parasitic/FastStaClockNetParasitic.hh"
#include "clock_sizing/FastStaClockSizingEdit.hh"
#include "timing/FastStaClockTiming.hh"

namespace icts {
namespace {

auto findBufferInputNode(const FastStaClockContext& context, const FastStaNode& output_node) -> FastStaNodeId
{
  if (output_node.inst_name.empty()) {
    return kInvalidFastStaNodeId;
  }
  for (FastStaNodeId node_id = 0U; node_id < context.nodes.size(); ++node_id) {
    const auto& node = context.nodes.at(node_id);
    if (node.kind == FastStaNodeKind::kBufferInput && node.inst_name == output_node.inst_name) {
      return node_id;
    }
  }
  return kInvalidFastStaNodeId;
}

auto findRcTerminalElmore(const FastStaNetParasitic& parasitic, FastStaNodeId node_id) -> double
{
  for (const auto& rc_node : parasitic.rc_nodes) {
    if (rc_node.terminal_node_id == node_id) {
      return rc_node.elmore_delay_ns;
    }
  }
  return 0.0;
}

auto findLoadLibertyCell(const FastStaClockContext& context, FastStaNodeId node_id) -> const FastStaLibertyCell*
{
  if (node_id >= context.nodes.size()) {
    return nullptr;
  }
  const auto& node = context.nodes.at(node_id);
  if (node.cell_master.empty()) {
    return nullptr;
  }
  const auto iter = context.liberty_cell_by_master.find(node.cell_master);
  return iter == context.liberty_cell_by_master.end() ? nullptr : &iter->second;
}

auto propagateBufferOutput(FastStaClockContext& context, FastStaNodeId output_node_id, FastStaNet& net) -> void
{
  if (output_node_id >= context.nodes.size()) {
    return;
  }
  auto& output_node = context.nodes.at(output_node_id);
  if (output_node.kind != FastStaNodeKind::kBufferOutput) {
    return;
  }

  const auto input_node_id = findBufferInputNode(context, output_node);
  if (input_node_id == kInvalidFastStaNodeId || input_node_id >= context.nodes.size()) {
    return;
  }
  const auto& input_node = context.nodes.at(input_node_id);
  if (!input_node.timing.valid) {
    return;
  }

  const auto liberty_iter = context.liberty_cell_by_master.find(output_node.cell_master);
  if (liberty_iter == context.liberty_cell_by_master.end()) {
    output_node.timing = input_node.timing;
    net.driver_dmp = FastStaDmpDriverResult{};
    return;
  }

  const auto& liberty_cell = liberty_iter->second;
  net.driver_dmp = FastStaDmpCeff::calcDriverTiming(liberty_cell, net.parasitic.pi, FastStaTransition::kRise, input_node.timing.slew_ns);
  output_node.timing = FastStaTimingPoint{
      .arrival_ns = input_node.timing.arrival_ns + net.driver_dmp.gate_delay_ns,
      .slew_ns = net.driver_dmp.driver_slew_ns > 0.0 ? net.driver_dmp.driver_slew_ns : input_node.timing.slew_ns,
      .valid = net.driver_dmp.valid,
  };
}

auto propagateNetLoads(FastStaClockContext& context, FastStaNet& net, std::queue<FastStaNodeId>& ready_nodes) -> void
{
  if (net.driver_node_id >= context.nodes.size() || !context.nodes.at(net.driver_node_id).timing.valid) {
    return;
  }

  const auto driver_timing = context.nodes.at(net.driver_node_id).timing;
  for (auto load_node_id : net.load_node_ids) {
    if (load_node_id >= context.nodes.size()) {
      continue;
    }
    const auto elmore_delay_ns = findRcTerminalElmore(net.parasitic, load_node_id);
    const auto* load_cell = findLoadLibertyCell(context, load_node_id);
    auto load_timing = FastStaDmpLoadResult{.valid = true, .wire_delay_ns = 0.0, .load_slew_ns = driver_timing.slew_ns};
    if (!net.parasitic.rc_nodes.empty()) {
      if (context.nodes.at(net.driver_node_id).kind == FastStaNodeKind::kSource) {
        load_timing = FastStaDmpCeff::calcInputPortDelaySlew(driver_timing.slew_ns, elmore_delay_ns, FastStaTransition::kRise, load_cell);
      } else {
        auto driver_dmp = net.driver_dmp;
        if (!driver_dmp.valid) {
          driver_dmp.valid = true;
          driver_dmp.driver_slew_ns = driver_timing.slew_ns;
        }
        load_timing = FastStaDmpCeff::calcLoadDelaySlew(driver_dmp, elmore_delay_ns, load_cell);
      }
    }
    auto& load_node = context.nodes.at(load_node_id);
    load_node.timing = FastStaTimingPoint{
        .arrival_ns = driver_timing.arrival_ns + load_timing.wire_delay_ns,
        .slew_ns = std::max(0.0, load_timing.load_slew_ns),
        .valid = load_timing.valid,
    };
    ready_nodes.push(load_node_id);
  }
}

auto resetTiming(FastStaClockContext& context) -> void
{
  for (auto& node : context.nodes) {
    node.timing = FastStaTimingPoint{};
  }
  for (auto& net : context.nets) {
    net.driver_dmp = FastStaDmpDriverResult{};
  }
}

auto resetTiming(FastStaClockContext& context, const FastStaDirtyRegion& dirty_region) -> void
{
  for (const auto node_id : dirty_region.node_ids) {
    if (node_id < context.nodes.size()) {
      context.nodes.at(node_id).timing = FastStaTimingPoint{};
    }
  }
  for (const auto net_id : dirty_region.net_ids) {
    if (net_id < context.nets.size()) {
      context.nets.at(net_id).driver_dmp = FastStaDmpDriverResult{};
    }
  }
}

auto mapBufferOutputByInst(const FastStaClockContext& context) -> std::unordered_map<std::string, FastStaNodeId>
{
  std::unordered_map<std::string, FastStaNodeId> output_by_inst;
  output_by_inst.reserve(context.nodes.size());
  for (FastStaNodeId node_id = 0U; node_id < context.nodes.size(); ++node_id) {
    const auto& node = context.nodes.at(node_id);
    if (node.kind == FastStaNodeKind::kBufferOutput && !node.inst_name.empty()) {
      output_by_inst[node.inst_name] = node_id;
    }
  }
  return output_by_inst;
}

auto propagateReadyQueue(FastStaClockContext& context, std::queue<FastStaNodeId>& ready_nodes) -> void
{
  const auto output_by_inst = mapBufferOutputByInst(context);
  std::size_t visited_steps = 0U;
  const auto max_steps = std::max<std::size_t>(1U, context.nodes.size() + context.nets.size() + 1U) * 4U;
  while (!ready_nodes.empty() && visited_steps < max_steps) {
    ++visited_steps;
    const auto node_id = ready_nodes.front();
    ready_nodes.pop();
    if (node_id >= context.nodes.size()) {
      continue;
    }
    auto& node = context.nodes.at(node_id);
    if (node.kind == FastStaNodeKind::kBufferInput) {
      const auto output_iter = output_by_inst.find(node.inst_name);
      if (output_iter != output_by_inst.end() && output_iter->second < context.nodes.size()) {
        auto& output_node = context.nodes.at(output_iter->second);
        if (!output_node.output_net_ids.empty() && output_node.output_net_ids.front() < context.nets.size()) {
          propagateBufferOutput(context, output_iter->second, context.nets.at(output_node.output_net_ids.front()));
        } else {
          output_node.timing = node.timing;
        }
        if (output_node.timing.valid) {
          ready_nodes.push(output_iter->second);
        }
      }
      continue;
    }
    for (auto net_id : node.output_net_ids) {
      if (net_id < context.nets.size()) {
        propagateNetLoads(context, context.nets.at(net_id), ready_nodes);
      }
    }
  }
}

}  // namespace

auto FastStaTiming::update(FastStaClockContext& context) -> bool
{
  resetTiming(context);
  FastStaParasitics::updateNetLoads(context);
  for (FastStaNetId net_id = 0U; net_id < context.nets.size(); ++net_id) {
    (void) FastStaParasitics::reduceToPiElmore(context, net_id);
  }
  if (context.source_node_id < context.nodes.size()) {
    context.nodes.at(context.source_node_id).timing
        = FastStaTimingPoint{.arrival_ns = 0.0, .slew_ns = std::max(0.0, context.root_input_slew_ns), .valid = true};
  }

  std::queue<FastStaNodeId> ready_nodes;
  if (context.source_node_id < context.nodes.size()) {
    ready_nodes.push(context.source_node_id);
  }

  propagateReadyQueue(context, ready_nodes);

  context.skew = calcSkew(context);
  context.timing_valid = true;
  return true;
}

auto FastStaTiming::updateRegion(FastStaClockContext& context, const FastStaDirtyRegion& dirty_region) -> bool
{
  if (!dirty_region.valid || dirty_region.start_node_id >= context.nodes.size()) {
    return false;
  }
  if (!context.timing_valid) {
    return update(context);
  }

  FastStaParasitics::updateNetLoads(context, dirty_region.net_ids);
  for (const auto net_id : dirty_region.net_ids) {
    (void) FastStaParasitics::reduceToPiElmore(context, net_id);
  }

  const auto preserved_start_timing = context.nodes.at(dirty_region.start_node_id).timing;
  resetTiming(context, dirty_region);
  context.nodes.at(dirty_region.start_node_id).timing = preserved_start_timing;
  if (!context.nodes.at(dirty_region.start_node_id).timing.valid) {
    return update(context);
  }

  std::queue<FastStaNodeId> ready_nodes;
  ready_nodes.push(dirty_region.start_node_id);
  propagateReadyQueue(context, ready_nodes);

  context.skew = calcSkew(context);
  context.timing_valid = true;
  return true;
}

auto FastStaTiming::calcSkew(const FastStaClockContext& context) -> FastStaSkewSummary
{
  FastStaSkewSummary summary;
  summary.min_arrival_ns = std::numeric_limits<double>::infinity();
  summary.max_arrival_ns = -std::numeric_limits<double>::infinity();
  for (FastStaNodeId node_id = 0U; node_id < context.nodes.size(); ++node_id) {
    const auto& node = context.nodes.at(node_id);
    if (node.kind != FastStaNodeKind::kSink || !node.timing.valid) {
      continue;
    }
    if (node.timing.arrival_ns < summary.min_arrival_ns) {
      summary.min_arrival_ns = node.timing.arrival_ns;
      summary.min_sink_node_id = node_id;
      summary.min_sink_name = node.name;
    }
    if (node.timing.arrival_ns > summary.max_arrival_ns) {
      summary.max_arrival_ns = node.timing.arrival_ns;
      summary.max_sink_node_id = node_id;
      summary.max_sink_name = node.name;
    }
  }
  summary.valid = summary.min_sink_node_id != kInvalidFastStaNodeId && summary.max_sink_node_id != kInvalidFastStaNodeId;
  if (summary.valid) {
    summary.skew_ns = std::max(0.0, summary.max_arrival_ns - summary.min_arrival_ns);
  } else {
    summary.min_arrival_ns = 0.0;
    summary.max_arrival_ns = 0.0;
  }
  return summary;
}

}  // namespace icts
