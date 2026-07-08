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
 * @file FastStaChar.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief Characterization sample context construction implementation for CTS fast STA.
 */

#include "FastStaChar.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "FastStaClockState.hh"
#include "FastStaLiberty.hh"
#include "FastStaLibertyModel.hh"
#include "FastStaParasitics.hh"
#include "FastStaPower.hh"
#include "FastStaTiming.hh"
#include "Log.hh"
#include "clock_net_parasitic/FastStaClockNetParasitic.hh"
#include "io/Wrapper.hh"
#include "timing/FastStaClockTiming.hh"

namespace icts {
namespace {

constexpr double kMilliOhmPerOhm = 1000.0;

auto resolveDbuPerUm(const FastStaCharTopologySpec& spec) -> int
{
  if (spec.dbu_per_um.has_value()) {
    LOG_FATAL_IF(*spec.dbu_per_um <= 0) << "FastStaChar: explicit DBU-per-micron must be positive.";
    return *spec.dbu_per_um;
  }

  LOG_FATAL << "FastStaChar: DBU-per-micron must be explicitly provided in the characterization topology spec.";
  return 0;
}

auto makePoint(double x_um, int dbu_per_um) -> FastStaPoint
{
  return FastStaPoint{.x_dbu = static_cast<int>(x_um * static_cast<double>(dbu_per_um)), .y_dbu = 0};
}

auto appendNode(FastStaClockContext& context, FastStaNode node) -> FastStaNodeId
{
  const auto node_id = context.nodes.size();
  context.node_id_by_name[node.name] = node_id;
  context.node_id_by_location[{node.location.x_dbu, node.location.y_dbu}] = node_id;
  context.nodes.push_back(std::move(node));
  return node_id;
}

auto appendNet(FastStaClockContext& context, FastStaNet net) -> FastStaNetId
{
  const auto net_id = context.nets.size();
  context.net_id_by_name[net.name] = net_id;
  if (net.driver_node_id < context.nodes.size()) {
    context.nodes.at(net.driver_node_id).output_net_ids.push_back(net_id);
  }
  for (const auto load_node_id : net.load_node_ids) {
    if (load_node_id < context.nodes.size()) {
      context.nodes.at(load_node_id).incoming_net_id = net_id;
    }
  }
  context.nets.push_back(std::move(net));
  return net_id;
}

auto sourceOutputNodeId(const FastStaClockContext& context) -> FastStaNodeId
{
  const auto iter = context.node_id_by_name.find("cts_char_source/Y");
  return iter == context.node_id_by_name.end() ? kInvalidFastStaNodeId : iter->second;
}

auto sinkNodeId(const FastStaClockContext& context) -> FastStaNodeId
{
  const auto iter = context.node_id_by_name.find("cts_char_sink/A");
  return iter == context.node_id_by_name.end() ? kInvalidFastStaNodeId : iter->second;
}

auto observationNodeId(const FastStaClockContext& context) -> FastStaNodeId
{
  FastStaNodeId observation_node_id = kInvalidFastStaNodeId;
  for (FastStaNodeId node_id = 0U; node_id < context.nodes.size(); ++node_id) {
    const auto& node = context.nodes.at(node_id);
    if (node.kind == FastStaNodeKind::kBufferOutput && node.inst_name.starts_with("cts_char_buf_")) {
      observation_node_id = node_id;
    }
  }
  return observation_node_id != kInvalidFastStaNodeId ? observation_node_id : sinkNodeId(context);
}

auto sourceBoundaryNetId(const FastStaClockContext& context) -> FastStaNetId
{
  const auto source_output_id = sourceOutputNodeId(context);
  if (source_output_id >= context.nodes.size() || context.nodes.at(source_output_id).output_net_ids.empty()) {
    return kInvalidFastStaNetId;
  }
  return context.nodes.at(source_output_id).output_net_ids.front();
}

auto makeLinearParasitic(const FastStaClockContext& context, const FastStaNet& net, FastStaNodeId driver_node_id,
                         FastStaNodeId load_node_id, double wirelength_um) -> FastStaNetParasitic
{
  LOG_FATAL_IF(context.wrapper == nullptr) << "FastStaChar: Wrapper is unavailable.";
  const auto wire_cap_pf = context.wrapper->queryRequiredWireCapacitance(context.routing_layer, wirelength_um, context.wire_width_um);
  const auto wire_resistance_ohm
      = context.wrapper->queryRequiredWireResistance(context.routing_layer, wirelength_um, context.wire_width_um) / kMilliOhmPerOhm;
  FastStaNetParasitic parasitic;
  parasitic.rc_nodes.push_back(FastStaRcNode{
      .name = net.name + "@root",
      .wire_cap_pf = wire_cap_pf / 2.0,
      .terminal_node_id = driver_node_id,
  });
  parasitic.rc_nodes.push_back(FastStaRcNode{
      .name = net.name + "@load",
      .wire_cap_pf = wire_cap_pf / 2.0,
      .terminal_node_id = load_node_id,
  });
  parasitic.rc_edges.push_back(FastStaRcEdge{.from = 0U, .to = 1U, .resistance_ohm = wire_resistance_ohm});
  parasitic.rc_node_id_by_name = {{parasitic.rc_nodes.front().name, 0U}, {parasitic.rc_nodes.back().name, 1U}};
  parasitic.root_rc_node_id = 0U;
  return parasitic;
}

auto rootTiming(const FastStaClockContext& context) -> FastStaTimingPoint
{
  const auto source_output_id = sourceOutputNodeId(context);
  if (source_output_id >= context.nodes.size()) {
    return {};
  }
  return context.nodes.at(source_output_id).timing;
}

auto isCharacterizedBufferOutput(const FastStaNode& node) -> bool
{
  return node.kind == FastStaNodeKind::kBufferOutput && node.inst_name.starts_with("cts_char_buf_");
}

auto selectedBufferInternalPower(const FastStaClockContext& context) -> double
{
  return std::accumulate(context.nodes.begin(), context.nodes.end(), 0.0, [](double sum, const FastStaNode& node) -> double {
    return isCharacterizedBufferOutput(node) ? sum + node.internal_power_w : sum;
  });
}

auto selectedBufferLeakagePower(const FastStaClockContext& context) -> double
{
  return std::accumulate(context.nodes.begin(), context.nodes.end(), 0.0, [](double sum, const FastStaNode& node) -> double {
    return isCharacterizedBufferOutput(node) ? sum + node.leakage_power_w : sum;
  });
}

}  // namespace

auto FastStaChar::buildContext(const FastStaCharTopologySpec& spec) -> FastStaClockContext
{
  FastStaClockContext context;
  LOG_FATAL_IF(spec.wrapper == nullptr) << "FastStaChar: Wrapper must be provided.";
  const auto dbu_per_um = resolveDbuPerUm(spec);
  LOG_FATAL_IF(spec.routing_layer <= 0) << "FastStaChar: routing layer must be explicitly provided.";
  context.clock_name = "cts_char_clk";
  context.wrapper = spec.wrapper;
  context.clock_net_name = "cts_char_net_0";
  context.clock_period_ns = spec.clock_period_ns;
  context.root_input_slew_ns = std::max(0.0, spec.root_input_slew_ns);
  context.dbu_per_um = dbu_per_um;
  context.routing_layer = spec.routing_layer;
  context.wire_width_um = spec.wire_width_um;

  context.liberty_cell_by_master[spec.source_cell_master] = FastStaLiberty::extractBufferCell(*spec.wrapper, spec.source_cell_master);
  context.liberty_cell_by_master[spec.sink_cell_master] = FastStaLiberty::extractBufferCell(*spec.wrapper, spec.sink_cell_master);
  for (const auto& cell_master : spec.buffer_cell_masters) {
    if (!context.liberty_cell_by_master.contains(cell_master)) {
      context.liberty_cell_by_master[cell_master] = FastStaLiberty::extractBufferCell(*spec.wrapper, cell_master);
    }
  }

  const auto source_input
      = appendNode(context, FastStaNode{
                                .kind = FastStaNodeKind::kBufferInput,
                                .name = "cts_char_source/A",
                                .inst_name = "cts_char_source",
                                .pin_name = "A",
                                .cell_master = spec.source_cell_master,
                                .location = makePoint(0.0, context.dbu_per_um),
                                .input_cap_pf = context.liberty_cell_by_master.at(spec.source_cell_master).input_cap_pf,
                                .max_slew_ns = context.liberty_cell_by_master.at(spec.source_cell_master).input_slew_limit_ns,
                                .output_net_ids = {},
                                .timing = {},
                            });
  const auto source_output = appendNode(context, FastStaNode{
                                                     .kind = FastStaNodeKind::kBufferOutput,
                                                     .name = "cts_char_source/Y",
                                                     .inst_name = "cts_char_source",
                                                     .pin_name = "Y",
                                                     .cell_master = spec.source_cell_master,
                                                     .location = makePoint(0.0, context.dbu_per_um),
                                                     .output_net_ids = {},
                                                     .timing = {},
                                                 });
  (void) source_output;
  context.source_node_id = source_input;

  auto current_x_um = 0.0;
  auto driver_node_id = source_output;
  for (std::size_t segment_index = 0U; segment_index < spec.wire_segments_um.size(); ++segment_index) {
    current_x_um += std::max(0.0, spec.wire_segments_um.at(segment_index));
    FastStaNodeId load_node_id = kInvalidFastStaNodeId;
    if (segment_index < spec.buffer_cell_masters.size()) {
      const auto& cell_master = spec.buffer_cell_masters.at(segment_index);
      const auto input_name = "cts_char_buf_" + std::to_string(segment_index) + "/A";
      const auto output_name = "cts_char_buf_" + std::to_string(segment_index) + "/Y";
      const auto inst_name = "cts_char_buf_" + std::to_string(segment_index);
      load_node_id = appendNode(context, FastStaNode{
                                             .kind = FastStaNodeKind::kBufferInput,
                                             .name = input_name,
                                             .inst_name = inst_name,
                                             .pin_name = "A",
                                             .cell_master = cell_master,
                                             .location = makePoint(current_x_um, context.dbu_per_um),
                                             .input_cap_pf = context.liberty_cell_by_master.at(cell_master).input_cap_pf,
                                             .max_slew_ns = context.liberty_cell_by_master.at(cell_master).input_slew_limit_ns,
                                             .output_net_ids = {},
                                             .timing = {},
                                         });
      (void) appendNode(context, FastStaNode{
                                     .kind = FastStaNodeKind::kBufferOutput,
                                     .name = output_name,
                                     .inst_name = inst_name,
                                     .pin_name = "Y",
                                     .cell_master = cell_master,
                                     .location = makePoint(current_x_um, context.dbu_per_um),
                                     .output_net_ids = {},
                                     .timing = {},
                                 });
    } else {
      load_node_id = appendNode(context, FastStaNode{
                                             .kind = FastStaNodeKind::kSink,
                                             .name = "cts_char_sink/A",
                                             .inst_name = "cts_char_sink",
                                             .pin_name = "A",
                                             .cell_master = spec.sink_cell_master,
                                             .location = makePoint(current_x_um, context.dbu_per_um),
                                             .input_cap_pf = context.liberty_cell_by_master.at(spec.sink_cell_master).input_cap_pf,
                                             .max_slew_ns = context.liberty_cell_by_master.at(spec.sink_cell_master).input_slew_limit_ns,
                                             .output_net_ids = {},
                                             .timing = {},
                                         });
    }

    const auto net_name = "cts_char_net_" + std::to_string(segment_index);
    const auto net_id = appendNet(
        context, FastStaNet{
                     .name = net_name,
                     .driver_node_id = driver_node_id,
                     .load_node_ids = {load_node_id},
                     .max_cap_pf = context.liberty_cell_by_master.at(context.nodes.at(driver_node_id).cell_master).output_cap_limit_pf,
                     .parasitic = {},
                     .driver_dmp = {},
                 });
    context.nets.at(net_id).parasitic = makeLinearParasitic(context, context.nets.at(net_id), driver_node_id, load_node_id,
                                                            std::max(0.0, spec.wire_segments_um.at(segment_index)));

    if (segment_index < spec.buffer_cell_masters.size()) {
      for (FastStaNodeId node_id = 0U; node_id < context.nodes.size(); ++node_id) {
        if (context.nodes.at(node_id).kind == FastStaNodeKind::kBufferOutput
            && context.nodes.at(node_id).inst_name == context.nodes.at(load_node_id).inst_name) {
          driver_node_id = node_id;
          break;
        }
      }
    }
  }

  FastStaParasitics::updateNetLoads(context);
  return context;
}

auto FastStaChar::setLoad(FastStaClockContext& context, double effective_load_pf) -> bool
{
  const auto sink_id = sinkNodeId(context);
  if (sink_id >= context.nodes.size()) {
    return false;
  }
  const auto& sink_node = context.nodes.at(sink_id);
  const auto sink_cell_iter = context.liberty_cell_by_master.find(sink_node.cell_master);
  const auto sink_input_cap_pf = sink_cell_iter == context.liberty_cell_by_master.end() ? 0.0 : sink_cell_iter->second.input_cap_pf;
  context.nodes.at(sink_id).input_cap_pf = sink_input_cap_pf + std::max(0.0, effective_load_pf);
  FastStaParasitics::updateNetLoads(context);
  context.timing_valid = false;
  context.power_valid = false;
  return true;
}

auto FastStaChar::runSample(FastStaClockContext& context, double input_slew_ns) -> FastStaCharSampleResult
{
  context.root_input_slew_ns = std::max(0.0, input_slew_ns);
  const auto timing_updated = FastStaTiming::update(context);
  if (!timing_updated) {
    return {};
  }

  const auto source_output_timing = rootTiming(context);
  if (!source_output_timing.valid) {
    return {};
  }

  const auto observation_id = observationNodeId(context);
  if (observation_id >= context.nodes.size() || !context.nodes.at(observation_id).timing.valid) {
    return {};
  }

  (void) FastStaPower::update(context);
  const auto boundary_net_id = sourceBoundaryNetId(context);
  const auto boundary_power = boundary_net_id < context.nets.size() ? context.nets.at(boundary_net_id).switching_power_w : 0.0;
  const auto delay_ns = std::max(0.0, context.nodes.at(observation_id).timing.arrival_ns - source_output_timing.arrival_ns);
  return FastStaCharSampleResult{
      .valid = true,
      .delay_ns = delay_ns,
      .output_slew_ns = context.nodes.at(observation_id).timing.slew_ns,
      .power_w = selectedBufferInternalPower(context) + selectedBufferLeakagePower(context) + context.power.switching_power_w,
      .source_boundary_net_switch_power_w = boundary_power,
  };
}

}  // namespace icts
