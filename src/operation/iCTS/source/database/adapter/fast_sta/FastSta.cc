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
 * @file FastSta.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief CTS-facing facade implementation for fast timing and power calculation.
 */

#include "FastSta.hh"

#include <glog/logging.h>

#include <chrono>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "FastStaBuilder.hh"
#include "FastStaChar.hh"
#include "FastStaIncremental.hh"
#include "FastStaLibertyModel.hh"
#include "FastStaPower.hh"
#include "FastStaTiming.hh"
#include "Log.hh"
#include "clock_net_parasitic/FastStaClockNetParasitic.hh"
#include "clock_state/FastStaClockState.hh"
#include "design/Clock.hh"
#include "design/Net.hh"
#include "timing/FastStaClockTiming.hh"

namespace icts {
namespace {

auto elapsedSeconds(std::chrono::steady_clock::time_point start_time) -> double
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
}

auto logContextSize(std::string_view owner, const FastStaClockContext& context) -> void
{
  LOG_INFO << owner << ": clock=\"" << context.clock_name << "\", nodes=" << context.nodes.size() << ", nets=" << context.nets.size()
           << ", liberty_cells=" << context.liberty_cell_by_master.size() << ".";
}

auto requireEnvironment(const std::optional<FastStaEnvironment>& environment) -> const FastStaEnvironment&
{
  LOG_FATAL_IF(!environment.has_value()) << "FastSTA: runtime environment is not bound.";
  LOG_FATAL_IF(environment->wrapper == nullptr) << "FastSTA: bound Wrapper is null.";
  LOG_FATAL_IF(environment->dbu_per_um <= 0) << "FastSTA: bound DBU-per-micron is invalid.";
  LOG_FATAL_IF(environment->routing_layer <= 0) << "FastSTA: bound routing layer is invalid.";
  return *environment;
}

auto toSlewRole(FastStaNodeKind kind) -> FastStaSlewRole
{
  switch (kind) {
    case FastStaNodeKind::kBufferInput:
      return FastStaSlewRole::kBufferInput;
    case FastStaNodeKind::kSink:
      return FastStaSlewRole::kSink;
    case FastStaNodeKind::kSource:
    case FastStaNodeKind::kBufferOutput:
      return FastStaSlewRole::kUnknown;
  }
  return FastStaSlewRole::kUnknown;
}

auto makeClockGraphProfile(const FastStaClockContext& context) -> FastStaClockGraphProfile
{
  FastStaClockGraphProfile profile;
  profile.node_count = context.nodes.size();
  profile.net_count = context.nets.size();
  for (const auto& node : context.nodes) {
    switch (node.kind) {
      case FastStaNodeKind::kSink:
        ++profile.sink_count;
        break;
      case FastStaNodeKind::kBufferInput:
        ++profile.buffer_input_count;
        break;
      case FastStaNodeKind::kBufferOutput:
        ++profile.buffer_output_count;
        break;
      case FastStaNodeKind::kSource:
        break;
    }
  }
  return profile;
}

auto makeClockTreeTopology(const FastStaClockContext& context) -> FastStaClockTreeTopology
{
  FastStaClockTreeTopology topology;
  topology.source_node_id = context.source_node_id;
  topology.parent_by_node.assign(context.nodes.size(), kInvalidFastStaNodeId);

  std::unordered_map<std::string, FastStaNodeId> input_by_inst;
  input_by_inst.reserve(context.nodes.size());
  for (FastStaNodeId node_id = 0U; node_id < context.nodes.size(); ++node_id) {
    const auto& node = context.nodes.at(node_id);
    if (node.kind == FastStaNodeKind::kBufferInput && !node.inst_name.empty()) {
      input_by_inst[node.inst_name] = node_id;
    }
  }

  for (FastStaNodeId node_id = 0U; node_id < context.nodes.size(); ++node_id) {
    const auto& node = context.nodes.at(node_id);
    if (node.kind == FastStaNodeKind::kBufferOutput) {
      const auto input_iter = input_by_inst.find(node.inst_name);
      if (input_iter != input_by_inst.end()) {
        topology.parent_by_node.at(node_id) = input_iter->second;
      }
      continue;
    }
    if (node.incoming_net_id < context.nets.size()) {
      topology.parent_by_node.at(node_id) = context.nets.at(node.incoming_net_id).driver_node_id;
    }
  }

  topology.children_by_node.assign(context.nodes.size(), {});
  for (FastStaNodeId node_id = 0U; node_id < topology.parent_by_node.size(); ++node_id) {
    const auto parent_id = topology.parent_by_node.at(node_id);
    if (parent_id < topology.children_by_node.size()) {
      topology.children_by_node.at(parent_id).push_back(node_id);
    }
  }
  return topology;
}

}  // namespace

struct FastSTA::ContextStore
{
  std::vector<std::unique_ptr<FastStaClockContext>> clock_contexts;
  std::vector<bool> clock_context_valid;
  std::vector<std::unique_ptr<FastStaClockContext>> char_contexts;
  std::vector<bool> char_context_valid;
};

FastSTA::FastSTA() : _contexts(std::make_unique<ContextStore>())
{
}

FastSTA::~FastSTA() = default;

auto FastSTA::bindEnvironment(const FastStaEnvironment& environment) -> void
{
  LOG_FATAL_IF(environment.wrapper == nullptr) << "FastSTA: cannot bind a null Wrapper.";
  LOG_FATAL_IF(environment.dbu_per_um <= 0) << "FastSTA: cannot bind invalid DBU-per-micron.";
  LOG_FATAL_IF(environment.routing_layer <= 0) << "FastSTA: cannot bind invalid routing layer.";
  _environment = environment;
}

auto FastSTA::buildClockContext(const FastStaClockBuildInput& input) -> FastStaClockId
{
  LOG_FATAL_IF(input.clock == nullptr) << "FastSTA: clock context build input has no clock.";
  const auto& environment = requireEnvironment(_environment);
  const auto& clock = *input.clock;
  const auto total_start = std::chrono::steady_clock::now();
  auto stage_start = std::chrono::steady_clock::now();
  LOG_INFO << "FastSTA: start build clock context for clock \"" << clock.get_clock_name() << "\".";
  auto context = FastStaBuilder::buildClockContext(environment, input);
  LOG_INFO << "FastSTA: clock context build finished in " << elapsedSeconds(stage_start) << " s.";
  logContextSize(input.route_geometry == nullptr ? "FastSTA committed-tree context" : "FastSTA route geometry context", context);
  const auto clock_id = _contexts->clock_contexts.size();
  _contexts->clock_contexts.push_back(std::make_unique<FastStaClockContext>(std::move(context)));
  _contexts->clock_context_valid.push_back(true);

  stage_start = std::chrono::steady_clock::now();
  LOG_INFO << "FastSTA: start initial timing update for clock \"" << clock.get_clock_name() << "\".";
  (void) updateTiming(clock_id);
  LOG_INFO << "FastSTA: initial timing update for clock \"" << clock.get_clock_name() << "\" finished in " << elapsedSeconds(stage_start)
           << " s.";

  stage_start = std::chrono::steady_clock::now();
  LOG_INFO << "FastSTA: start initial power update for clock \"" << clock.get_clock_name() << "\".";
  (void) updatePower(clock_id);
  LOG_INFO << "FastSTA: initial power update for clock \"" << clock.get_clock_name() << "\" finished in " << elapsedSeconds(stage_start)
           << " s.";
  LOG_INFO << "FastSTA: build clock context for clock \"" << clock.get_clock_name() << "\" finished in " << elapsedSeconds(total_start)
           << " s.";
  return clock_id;
}

auto FastSTA::eraseClockContext(FastStaClockId clock_id) -> bool
{
  if (clock_id >= _contexts->clock_context_valid.size() || !_contexts->clock_context_valid.at(clock_id)) {
    return false;
  }
  _contexts->clock_context_valid.at(clock_id) = false;
  return true;
}

auto FastSTA::reset() -> void
{
  _contexts->clock_contexts.clear();
  _contexts->clock_context_valid.clear();
  _contexts->char_contexts.clear();
  _contexts->char_context_valid.clear();
  _environment = std::nullopt;
}

auto FastSTA::buildCharContext(const FastStaCharTopologySpec& spec) -> FastStaCharContextId
{
  auto context = FastStaChar::buildContext(spec);
  const auto char_context_id = _contexts->char_contexts.size();
  _contexts->char_contexts.push_back(std::make_unique<FastStaClockContext>(std::move(context)));
  _contexts->char_context_valid.push_back(true);
  return char_context_id;
}

auto FastSTA::eraseCharContext(FastStaCharContextId char_context_id) -> bool
{
  if (char_context_id >= _contexts->char_context_valid.size() || !_contexts->char_context_valid.at(char_context_id)) {
    return false;
  }
  _contexts->char_context_valid.at(char_context_id) = false;
  return true;
}

auto FastSTA::setCharLoad(FastStaCharContextId char_context_id, double effective_load_pf) -> bool
{
  if (char_context_id >= _contexts->char_contexts.size() || char_context_id >= _contexts->char_context_valid.size()
      || !_contexts->char_context_valid.at(char_context_id)) {
    LOG_ERROR << "FastSTA: characterization load update skipped because char context id is invalid.";
    return false;
  }
  return FastStaChar::setLoad(*_contexts->char_contexts.at(char_context_id), effective_load_pf);
}

auto FastSTA::runCharSample(FastStaCharContextId char_context_id, double input_slew_ns) -> FastStaCharSampleResult
{
  if (char_context_id >= _contexts->char_contexts.size() || char_context_id >= _contexts->char_context_valid.size()
      || !_contexts->char_context_valid.at(char_context_id)) {
    LOG_ERROR << "FastSTA: characterization sample skipped because char context id is invalid.";
    return {};
  }
  return FastStaChar::runSample(*_contexts->char_contexts.at(char_context_id), input_slew_ns);
}

auto FastSTA::changeBufferMasters(FastStaClockId clock_id, const std::vector<FastStaBufferMasterChange>& changes) -> bool
{
  auto* context = mutableClockContext(clock_id);
  if (context == nullptr) {
    LOG_ERROR << "FastSTA: buffer master batch change skipped because clock context id is invalid.";
    return false;
  }
  if (changes.empty()) {
    return context->timing_valid && context->power_valid;
  }
  if (!FastStaIncremental::changeBufferMasters(*context, changes)) {
    return false;
  }
  return FastStaTiming::update(*context) && FastStaPower::update(*context);
}

auto FastSTA::changeBufferMastersTimingOnly(FastStaClockId clock_id, const std::vector<FastStaBufferMasterChange>& changes) -> bool
{
  auto* context = mutableClockContext(clock_id);
  if (context == nullptr) {
    LOG_ERROR << "FastSTA: timing-only buffer master batch change skipped because clock context id is invalid.";
    return false;
  }
  if (changes.empty()) {
    return context->timing_valid;
  }
  if (!FastStaIncremental::changeBufferMasters(*context, changes)) {
    return false;
  }
  const bool timing_updated = FastStaTiming::update(*context);
  context->power_valid = false;
  return timing_updated;
}

auto FastSTA::updateTiming(FastStaClockId clock_id) -> bool
{
  auto* context = mutableClockContext(clock_id);
  if (context == nullptr) {
    LOG_ERROR << "FastSTA: timing update skipped because clock context id is invalid.";
    return false;
  }
  return FastStaTiming::update(*context);
}

auto FastSTA::updatePower(FastStaClockId clock_id) -> bool
{
  auto* context = mutableClockContext(clock_id);
  if (context == nullptr) {
    LOG_ERROR << "FastSTA: power update skipped because clock context id is invalid.";
    return false;
  }
  return FastStaPower::update(*context);
}

auto FastSTA::injectNetRouteTree(FastStaClockId clock_id, const Net& net, const ClockSteinerTree<int>& route_tree,
                                 FastStaClockNetRcTreeCounts& rc_tree_counts) -> bool
{
  rc_tree_counts = {};
  auto* context = mutableClockContext(clock_id);
  if (context == nullptr) {
    LOG_ERROR << "FastSTA: route-tree injection skipped because clock context id is invalid.";
    return false;
  }
  if (!FastStaBuilder::injectNetRouteTree(*context, net, route_tree)) {
    return false;
  }
  const auto net_iter = context->net_id_by_name.find(net.get_name());
  if (net_iter == context->net_id_by_name.end() || net_iter->second >= context->nets.size()) {
    return true;
  }
  const auto& parasitic = context->nets.at(net_iter->second).parasitic;
  rc_tree_counts.rc_node_count = parasitic.rc_nodes.size();
  rc_tree_counts.rc_edge_count = parasitic.rc_edges.size();
  return true;
}

auto FastSTA::queryClockGraphProfile(FastStaClockId clock_id) const -> std::optional<FastStaClockGraphProfile>
{
  const auto* context = queryClockContext(clock_id);
  if (context == nullptr) {
    return std::nullopt;
  }
  return makeClockGraphProfile(*context);
}

auto FastSTA::queryClockAnalysisStatus(FastStaClockId clock_id) const -> std::optional<FastStaClockAnalysisStatus>
{
  const auto* context = queryClockContext(clock_id);
  if (context == nullptr) {
    return std::nullopt;
  }
  return FastStaClockAnalysisStatus{.timing_valid = context->timing_valid, .power_valid = context->power_valid};
}

auto FastSTA::queryClockTreeTopology(FastStaClockId clock_id) const -> std::optional<FastStaClockTreeTopology>
{
  const auto* context = queryClockContext(clock_id);
  if (context == nullptr) {
    return std::nullopt;
  }
  return makeClockTreeTopology(*context);
}

auto FastSTA::collectClockSizingBuffers(FastStaClockId clock_id) const -> std::vector<FastStaClockSizingBuffer>
{
  std::vector<FastStaClockSizingBuffer> buffers;
  const auto* context = queryClockContext(clock_id);
  if (context == nullptr) {
    return buffers;
  }
  buffers.reserve(context->nodes.size());
  for (FastStaNodeId node_id = 0U; node_id < context->nodes.size(); ++node_id) {
    const auto& node = context->nodes.at(node_id);
    if (node.kind != FastStaNodeKind::kBufferOutput || node.inst_name.empty() || node.cell_master.empty()) {
      continue;
    }
    buffers.push_back(FastStaClockSizingBuffer{.node_id = node_id, .inst_name = node.inst_name, .cell_master = node.cell_master});
  }
  return buffers;
}

auto FastSTA::collectClockSinkArrivals(FastStaClockId clock_id) const -> std::vector<FastStaClockSinkArrival>
{
  std::vector<FastStaClockSinkArrival> sinks;
  const auto* context = queryClockContext(clock_id);
  if (context == nullptr) {
    return sinks;
  }
  sinks.reserve(context->nodes.size());
  for (FastStaNodeId node_id = 0U; node_id < context->nodes.size(); ++node_id) {
    const auto& node = context->nodes.at(node_id);
    if (node.kind != FastStaNodeKind::kSink || !node.timing.valid) {
      continue;
    }
    sinks.push_back(FastStaClockSinkArrival{.node_id = node_id, .sink_name = node.name, .arrival_ns = node.timing.arrival_ns});
  }
  return sinks;
}

auto FastSTA::queryClockNodeArrival(FastStaClockId clock_id, FastStaNodeId node_id) const -> std::optional<double>
{
  const auto* context = queryClockContext(clock_id);
  if (context == nullptr || node_id >= context->nodes.size() || !context->nodes.at(node_id).timing.valid) {
    return std::nullopt;
  }
  return context->nodes.at(node_id).timing.arrival_ns;
}

auto FastSTA::querySkew(FastStaClockId clock_id) const -> FastStaSkewSummary
{
  const auto* context = queryClockContext(clock_id);
  return context == nullptr ? FastStaSkewSummary{} : context->skew;
}

auto FastSTA::queryCapStatus(FastStaClockId clock_id, FastStaNetId net_id) const -> std::optional<FastStaCapStatus>
{
  const auto* context = queryClockContext(clock_id);
  if (context == nullptr || net_id >= context->nets.size()) {
    return std::nullopt;
  }
  const auto& net = context->nets.at(net_id);
  return FastStaCapStatus{.net_id = net_id,
                          .net_name = net.name,
                          .load_cap_pf = net.load_cap_pf,
                          .max_cap_pf = net.max_cap_pf,
                          .violated = net.max_cap_pf > 0.0 && net.load_cap_pf > net.max_cap_pf};
}

auto FastSTA::querySlewStatus(FastStaClockId clock_id, FastStaNodeId node_id) const -> std::optional<FastStaSlewStatus>
{
  const auto* context = queryClockContext(clock_id);
  if (context == nullptr || node_id >= context->nodes.size()) {
    return std::nullopt;
  }
  const auto& node = context->nodes.at(node_id);
  if (!node.timing.valid) {
    return std::nullopt;
  }
  return FastStaSlewStatus{.node_id = node_id,
                           .node_name = node.name,
                           .role = toSlewRole(node.kind),
                           .slew_ns = node.timing.slew_ns,
                           .max_slew_ns = node.max_slew_ns,
                           .violated = node.max_slew_ns > 0.0 && node.timing.slew_ns > node.max_slew_ns};
}

auto FastSTA::queryPower(FastStaClockId clock_id) const -> FastStaPowerSummary
{
  const auto* context = queryClockContext(clock_id);
  return context == nullptr ? FastStaPowerSummary{} : context->power;
}

auto FastSTA::queryClockContext(FastStaClockId clock_id) const -> const FastStaClockContext*
{
  if (clock_id >= _contexts->clock_contexts.size() || clock_id >= _contexts->clock_context_valid.size()
      || !_contexts->clock_context_valid.at(clock_id)) {
    return nullptr;
  }
  return _contexts->clock_contexts.at(clock_id).get();
}

auto FastSTA::mutableClockContext(FastStaClockId clock_id) -> FastStaClockContext*
{
  if (clock_id >= _contexts->clock_contexts.size() || clock_id >= _contexts->clock_context_valid.size()
      || !_contexts->clock_context_valid.at(clock_id)) {
    return nullptr;
  }
  return _contexts->clock_contexts.at(clock_id).get();
}

}  // namespace icts
