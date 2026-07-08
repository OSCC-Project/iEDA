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
 * @file ClockDAG.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-06
 * @brief Design-owned read-only clock DAG projection implementation
 */

#include "database/design/ClockDAG.hh"

#include <algorithm>
#include <deque>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "database/design/Clock.hh"
#include "database/design/Inst.hh"
#include "database/design/Net.hh"
#include "database/design/Pin.hh"

namespace icts {
namespace {

auto markInvalid(ClockDAG::ClockGraph& graph, const std::string& reason) -> void
{
  graph.valid = false;
  if (graph.status == "valid" || graph.status.empty()) {
    graph.status = reason;
  } else if (!reason.empty() && graph.status.find(reason) == std::string::npos) {
    graph.status += "; " + reason;
  }
}

auto appendPin(ClockDAG::ClockGraph& graph, Pin* pin) -> void
{
  if (pin == nullptr || graph.pin_set.contains(pin)) {
    return;
  }
  graph.pin_set.insert(pin);
  graph.pins.push_back(pin);
}

auto appendNet(ClockDAG::ClockGraph& graph, Net* net) -> void
{
  if (net == nullptr || graph.net_set.contains(net)) {
    return;
  }
  graph.net_set.insert(net);
  graph.nets.push_back(net);
}

auto appendArc(ClockDAG::ClockGraph& graph, Pin* from, Pin* to, Net* net, int32_t path_buffer_weight) -> void
{
  if (from == nullptr || to == nullptr) {
    markInvalid(graph, "null_graph_arc_pin");
    return;
  }
  appendPin(graph, from);
  appendPin(graph, to);
  graph.outgoing_arcs[from].push_back(ClockDAG::Arc{
      .from = from,
      .to = to,
      .net = net,
      .path_buffer_weight = path_buffer_weight,
  });
}

auto instTypeName(const Inst& inst) -> std::string
{
  switch (inst.get_type()) {
    case InstType::kBuffer:
      return "buffer";
    case InstType::kFlipFlop:
      return "flipflop";
    case InstType::kLatch:
      return "latch";
    case InstType::kInverter:
      return "inverter";
    case InstType::kClockGate:
      return "clock_gate";
    case InstType::kMux:
      return "clock_mux";
    case InstType::kClockLogic:
      return "clock_logic";
    case InstType::kBoundaryLoad:
      return "boundary_load";
    case InstType::kMacroBlock:
      return "macro_block";
    case InstType::kUnknown:
      return "unknown";
  }
  return "unknown";
}

auto formatClockCellError(const std::string& reason, const Inst& inst) -> std::string
{
  std::string pins;
  for (auto* pin : inst.get_pins()) {
    if (pin == nullptr) {
      continue;
    }
    if (!pins.empty()) {
      pins += ",";
    }
    pins += pin->get_name();
  }
  return reason + "{inst=" + inst.get_name() + ",cell_master=" + inst.get_cell_master() + ",type=" + instTypeName(inst) + ",pins=" + pins
         + "}";
}

auto collectClockNets(const Clock& clock) -> std::vector<Net*>
{
  std::vector<Net*> nets;
  std::unordered_set<const Net*> seen_nets;
  auto append_net = [&nets, &seen_nets](Net* net) -> void {
    if (net == nullptr || seen_nets.contains(net)) {
      return;
    }
    seen_nets.insert(net);
    nets.push_back(net);
  };

  append_net(clock.get_clock_source_net());
  if (clock.get_clock_source() != nullptr) {
    append_net(clock.get_clock_source()->get_net());
  }
  for (auto* net : clock.get_nets()) {
    append_net(net);
  }
  return nets;
}

auto collectClockInsts(const Clock& clock, const std::vector<Net*>& nets) -> std::vector<Inst*>
{
  std::vector<Inst*> insts;
  std::unordered_set<const Inst*> seen_insts;
  auto append_inst = [&insts, &seen_insts](Inst* inst) -> void {
    if (inst == nullptr || seen_insts.contains(inst)) {
      return;
    }
    seen_insts.insert(inst);
    insts.push_back(inst);
  };

  for (auto* inst : clock.get_insts()) {
    append_inst(inst);
  }
  for (auto* net : nets) {
    if (net == nullptr) {
      continue;
    }
    if (net->get_driver() != nullptr) {
      append_inst(net->get_driver()->get_inst());
    }
    for (auto* load : net->get_loads()) {
      if (load != nullptr) {
        append_inst(load->get_inst());
      }
    }
  }
  return insts;
}

auto buildNetArcs(ClockDAG::ClockGraph& graph, const std::vector<Net*>& nets) -> void
{
  for (auto* net : nets) {
    if (net == nullptr) {
      continue;
    }
    appendNet(graph, net);
    auto* driver = net->get_driver();
    if (driver == nullptr) {
      markInvalid(graph, "net_driver_pin_is_null");
      continue;
    }
    appendPin(graph, driver);
    for (auto* load : net->get_loads()) {
      if (load == nullptr) {
        markInvalid(graph, "net_load_pin_is_null");
        continue;
      }
      appendArc(graph, driver, load, net, 0);
    }
  }
}

auto buildBufferCellArcs(ClockDAG::ClockGraph& graph, const std::vector<Inst*>& insts) -> void
{
  for (auto* inst : insts) {
    if (inst == nullptr || !inst->is_clock_propagation_cell()) {
      continue;
    }
    auto* output_pin = inst->findDriverPin();
    if (output_pin == nullptr) {
      markInvalid(graph, formatClockCellError("clock_cell_output_pin_is_null", *inst));
      continue;
    }
    appendPin(graph, output_pin);

    bool has_input_pin = false;
    for (auto* input_pin : inst->get_pins()) {
      if (input_pin == nullptr || input_pin == output_pin) {
        continue;
      }
      has_input_pin = true;
      appendArc(graph, input_pin, output_pin, nullptr, 1);
    }
    if (!has_input_pin) {
      markInvalid(graph, formatClockCellError("clock_cell_input_pin_is_null", *inst));
    }
  }
}

auto finishTopologicalOrder(ClockDAG::ClockGraph& graph) -> void
{
  std::unordered_map<const Pin*, std::size_t> in_degree;
  in_degree.reserve(graph.pins.size());
  for (auto* pin : graph.pins) {
    in_degree[pin] = 0U;
  }
  for (const auto& [from, arcs] : graph.outgoing_arcs) {
    (void) from;
    for (const auto& arc : arcs) {
      ++in_degree[arc.to];
    }
  }

  std::deque<Pin*> ready;
  for (auto* pin : graph.pins) {
    if (in_degree[pin] == 0U) {
      ready.push_back(pin);
    }
  }

  graph.topological_pins.clear();
  graph.topological_pins.reserve(graph.pins.size());
  while (!ready.empty()) {
    auto* pin = ready.front();
    ready.pop_front();
    graph.topological_pins.push_back(pin);

    const auto arc_iter = graph.outgoing_arcs.find(pin);
    if (arc_iter == graph.outgoing_arcs.end()) {
      continue;
    }
    for (const auto& arc : arc_iter->second) {
      auto degree_iter = in_degree.find(arc.to);
      if (degree_iter == in_degree.end() || degree_iter->second == 0U) {
        continue;
      }
      --degree_iter->second;
      if (degree_iter->second == 0U) {
        ready.push_back(arc.to);
      }
    }
  }

  if (graph.topological_pins.size() != graph.pins.size()) {
    graph.has_cycle = true;
    markInvalid(graph, "cycle_detected");
    graph.topological_pins.clear();
  }
}

auto buildClockGraph(const Clock& clock) -> ClockDAG::ClockGraph
{
  ClockDAG::ClockGraph graph;
  graph.clock = &clock;

  auto* source_pin = clock.get_clock_source();
  if (source_pin == nullptr) {
    markInvalid(graph, "clock_source_pin_is_null");
  } else {
    appendPin(graph, source_pin);
  }

  auto nets = collectClockNets(clock);
  buildNetArcs(graph, nets);
  buildBufferCellArcs(graph, collectClockInsts(clock, nets));
  finishTopologicalOrder(graph);
  return graph;
}

auto isFlipFlopSinkTerminal(const Clock& clock, const Pin* pin) -> bool
{
  if (pin == nullptr || pin->get_inst() == nullptr || !pin->get_inst()->is_sequential_sink()) {
    return false;
  }
  const auto& loads = clock.get_loads();
  return std::ranges::find(loads, pin) != loads.end();
}

auto countDeclaredFlipFlopSinkTerminals(const Clock& clock) -> std::size_t
{
  std::size_t count = 0U;
  for (const auto* load : clock.get_loads()) {
    if (isFlipFlopSinkTerminal(clock, load)) {
      ++count;
    }
  }
  return count;
}

auto statusForUnavailableClockStats(const Clock& clock, std::size_t reachable_ff_sink_count) -> std::string
{
  if (reachable_ff_sink_count > 0U) {
    return "available";
  }
  return countDeclaredFlipFlopSinkTerminals(clock) > 0U ? "no_reachable_ff_sink_terminal" : "no_ff_sink_terminal";
}

}  // namespace

auto ClockDAG::rebuild(const std::vector<Clock*>& clocks) -> bool
{
  clear();
  _built = true;
  _valid = true;
  _status = clocks.empty() ? "empty" : "valid";
  _clock_order.reserve(clocks.size());

  for (auto* clock : clocks) {
    if (clock == nullptr) {
      _valid = false;
      _status = "null_clock_pointer";
      continue;
    }

    auto graph = buildClockGraph(*clock);
    if (!graph.valid) {
      _valid = false;
      if (_status == "valid" || _status == "empty") {
        _status = graph.status;
      } else if (_status.find(graph.status) == std::string::npos) {
        _status += "; " + graph.status;
      }
    }
    _clock_order.push_back(clock);
    _graphs_by_clock.emplace(clock, std::move(graph));
  }
  return is_valid();
}

auto ClockDAG::clear() -> void
{
  _built = false;
  _valid = false;
  _status = "not_built";
  _clock_order.clear();
  _graphs_by_clock.clear();
}

auto ClockDAG::invalidate(const std::string& reason) -> void
{
  clear();
  _status = reason.empty() ? "invalidated" : reason;
}

auto ClockDAG::findGraph(const Clock* clock) const -> const ClockGraph*
{
  if (clock == nullptr) {
    return nullptr;
  }
  const auto iter = _graphs_by_clock.find(clock);
  return iter == _graphs_by_clock.end() ? nullptr : &iter->second;
}

auto ClockDAG::hasCycle(const Clock* clock) const -> bool
{
  const auto* graph = findGraph(clock);
  return graph != nullptr && graph->has_cycle;
}

auto ClockDAG::topologicalPins(const Clock* clock) const -> std::vector<Pin*>
{
  const auto* graph = findGraph(clock);
  if (graph == nullptr || !graph->valid) {
    return {};
  }
  return graph->topological_pins;
}

auto ClockDAG::reachablePins(const Clock* clock) const -> std::vector<Pin*>
{
  return reachablePinsFrom(clock, clock != nullptr ? clock->get_clock_source() : nullptr);
}

auto ClockDAG::reachablePinsFrom(const Clock* clock, Pin* start_pin) const -> std::vector<Pin*>
{
  const auto* graph = findGraph(clock);
  if (graph == nullptr || !graph->valid || start_pin == nullptr || !graph->pin_set.contains(start_pin)) {
    return {};
  }
  std::vector<Pin*> reachable;
  std::deque<Pin*> pending;
  std::unordered_set<const Pin*> visited;
  pending.push_back(start_pin);
  while (!pending.empty()) {
    auto* pin = pending.front();
    pending.pop_front();
    if (pin == nullptr || !visited.insert(pin).second) {
      continue;
    }
    reachable.push_back(pin);

    const auto arc_iter = graph->outgoing_arcs.find(pin);
    if (arc_iter == graph->outgoing_arcs.end()) {
      continue;
    }
    for (const auto& arc : arc_iter->second) {
      pending.push_back(arc.to);
    }
  }
  return reachable;
}

auto ClockDAG::reachableNets(const Clock* clock) const -> std::vector<Net*>
{
  const auto* graph = findGraph(clock);
  if (graph == nullptr || !graph->valid) {
    return {};
  }

  const auto reachable_pins = reachablePins(clock);
  std::unordered_set<const Pin*> reachable_pin_set;
  reachable_pin_set.reserve(reachable_pins.size());
  for (auto* pin : reachable_pins) {
    reachable_pin_set.insert(pin);
  }

  std::vector<Net*> nets;
  nets.reserve(graph->nets.size());
  for (auto* net : graph->nets) {
    if (net == nullptr || net->get_driver() == nullptr || !reachable_pin_set.contains(net->get_driver())) {
      continue;
    }
    nets.push_back(net);
  }
  return nets;
}

auto ClockDAG::graphForClock(const Clock* clock) const -> const ClockGraph*
{
  const auto* graph = findGraph(clock);
  return graph != nullptr && graph->valid ? graph : nullptr;
}

auto ClockDAG::pathBufferStats(const Clock* clock) const -> PathBufferStats
{
  PathBufferStats stats;
  if (!_built) {
    stats.status = "not_built";
    return stats;
  }
  if (!_valid) {
    stats.status = "invalid_topology";
    return stats;
  }

  const auto* graph = findGraph(clock);
  if (graph == nullptr || clock == nullptr) {
    stats.status = "clock_not_found";
    return stats;
  }
  if (!graph->valid) {
    stats.status = "invalid_topology";
    return stats;
  }
  stats.topology_valid = true;
  if (clock->get_clock_source() == nullptr) {
    stats.status = "clock_source_pin_is_null";
    return stats;
  }

  constexpr int32_t unreachable_count = std::numeric_limits<int32_t>::max() / 4;
  std::unordered_map<const Pin*, int32_t> min_count;
  std::unordered_map<const Pin*, int32_t> max_count;
  min_count.reserve(graph->pins.size());
  max_count.reserve(graph->pins.size());
  for (auto* pin : graph->pins) {
    min_count[pin] = unreachable_count;
    max_count[pin] = -1;
  }
  min_count[clock->get_clock_source()] = 0;
  max_count[clock->get_clock_source()] = 0;

  for (auto* pin : graph->topological_pins) {
    if (min_count[pin] == unreachable_count) {
      continue;
    }
    const auto arc_iter = graph->outgoing_arcs.find(pin);
    if (arc_iter == graph->outgoing_arcs.end()) {
      continue;
    }
    for (const auto& arc : arc_iter->second) {
      min_count[arc.to] = std::min(min_count[arc.to], min_count[pin] + arc.path_buffer_weight);
      max_count[arc.to] = std::max(max_count[arc.to], max_count[pin] + arc.path_buffer_weight);
    }
  }

  int32_t min_path_count = unreachable_count;
  int32_t max_path_count = 0;
  for (auto* pin : graph->pins) {
    if (min_count[pin] == unreachable_count || !isFlipFlopSinkTerminal(*clock, pin)) {
      continue;
    }
    min_path_count = std::min(min_path_count, min_count[pin]);
    max_path_count = std::max(max_path_count, max_count[pin]);
    ++stats.ff_sink_terminal_count;
  }

  stats.status = statusForUnavailableClockStats(*clock, stats.ff_sink_terminal_count);
  stats.has_ff_sink_terminal = stats.ff_sink_terminal_count > 0U;
  stats.available = stats.has_ff_sink_terminal;
  if (stats.available) {
    stats.min_buffer_count = min_path_count;
    stats.max_buffer_count = max_path_count;
  }
  return stats;
}

auto ClockDAG::pathBufferStats() const -> PathBufferStats
{
  PathBufferStats aggregate;
  if (!_built) {
    aggregate.status = "not_built";
    return aggregate;
  }
  if (!_valid) {
    aggregate.status = "invalid_topology";
    return aggregate;
  }

  int32_t min_path_count = std::numeric_limits<int32_t>::max();
  int32_t max_path_count = 0;
  std::string first_unavailable_status = "no_ff_sink_terminal";
  bool has_unavailable_status = false;
  for (const auto* clock : _clock_order) {
    const auto stats = pathBufferStats(clock);
    if (stats.available) {
      aggregate.available = true;
      aggregate.has_ff_sink_terminal = true;
      aggregate.ff_sink_terminal_count += stats.ff_sink_terminal_count;
      min_path_count = std::min(min_path_count, stats.min_buffer_count);
      max_path_count = std::max(max_path_count, stats.max_buffer_count);
      continue;
    }
    if (!has_unavailable_status) {
      first_unavailable_status = stats.status;
      has_unavailable_status = true;
    }
  }

  aggregate.topology_valid = true;
  if (aggregate.available) {
    aggregate.min_buffer_count = min_path_count;
    aggregate.max_buffer_count = max_path_count;
    aggregate.status = "available";
  } else {
    aggregate.status = first_unavailable_status;
  }
  return aggregate;
}

}  // namespace icts
