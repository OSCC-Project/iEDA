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
 * @file ClockTraceResolve.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Net resolution and traversal logic for SDC clock tracing.
 */

#include <algorithm>
#include <cstddef>
#include <deque>
#include <ranges>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "IdbDesign.h"
#include "IdbEnum.h"
#include "IdbInstance.h"
#include "IdbNet.h"
#include "IdbPins.h"
#include "IdbTerm.h"
// use iEDA Rust liberty parser
#include "SdcClockReader.hh"
#include "SdcClockTraceAlgorithm.hh"
#include "liberty/Lib.hh"

namespace icts::clock_trace {

namespace {

auto CollectOutputPins(idb::IdbInstance* inst) -> std::vector<idb::IdbPin*>
{
  std::vector<idb::IdbPin*> outputs;
  if (inst == nullptr || inst->get_pin_list() == nullptr) {
    return outputs;
  }
  for (auto* pin : inst->get_pin_list()->get_pin_list()) {
    if (IsOutputLike(pin)) {
      outputs.push_back(pin);
    }
  }
  return outputs;
}

auto CollectInputPins(idb::IdbInstance* inst) -> std::vector<idb::IdbPin*>
{
  std::vector<idb::IdbPin*> inputs;
  if (inst == nullptr || inst->get_pin_list() == nullptr) {
    return inputs;
  }
  for (auto* pin : inst->get_pin_list()->get_pin_list()) {
    if (IsInputLike(pin)) {
      inputs.push_back(pin);
    }
  }
  return inputs;
}

auto IsCaseConstrained(idb::IdbPin* pin, const CaseConstraintSet& case_constraints) -> bool
{
  if (pin == nullptr) {
    return false;
  }
  const auto full_name = PinFullName(pin);
  if (case_constraints.pin_names.contains(full_name) || case_constraints.pin_names.contains(TermName(pin))) {
    return true;
  }
  auto* net = pin->get_net();
  return net != nullptr && case_constraints.net_names.contains(net->get_net_name());
}

auto OtherInputsCaseConstrained(idb::IdbInstance* inst, idb::IdbPin* clock_input_pin, const CaseConstraintSet& case_constraints) -> bool
{
  bool has_other_input = false;
  for (auto* input_pin : CollectInputPins(inst)) {
    if (input_pin == nullptr || input_pin == clock_input_pin) {
      continue;
    }
    has_other_input = true;
    if (!IsCaseConstrained(input_pin, case_constraints)) {
      return false;
    }
  }
  return has_other_input;
}

auto LibertyExpressionUsesPort(RustLibertyExpr* expression, const std::string& port_name) -> bool
{
  if (expression == nullptr || port_name.empty()) {
    return false;
  }

  std::vector<RustLibertyExpr*> stack = {expression};
  std::vector<RustLibertyExpr*> owned_exprs;
  while (!stack.empty()) {
    auto* current = stack.back();
    stack.pop_back();
    if (current == nullptr) {
      continue;
    }
    if (current->port_name != nullptr && port_name == current->port_name) {
      for (auto* owned_expr : owned_exprs) {
        liberty_free_expr(owned_expr);
      }
      return true;
    }

    auto* left_expr = liberty_get_expr_left(current);
    if (left_expr != nullptr) {
      stack.push_back(left_expr);
      owned_exprs.push_back(left_expr);
    }
    auto* right_expr = liberty_get_expr_right(current);
    if (right_expr != nullptr) {
      stack.push_back(right_expr);
      owned_exprs.push_back(right_expr);
    }
  }
  for (auto* owned_expr : owned_exprs) {
    liberty_free_expr(owned_expr);
  }
  return false;
}

auto OutputFunctionUsesInput(idb::LibCell* lib_cell, idb::IdbPin* output_pin, idb::IdbPin* input_pin) -> bool
{
  if (lib_cell == nullptr) {
    return true;
  }
  auto* output_port = FindLibPort(lib_cell, output_pin);
  if (output_port == nullptr || output_port->get_func_expr() == nullptr) {
    return true;
  }
  return LibertyExpressionUsesPort(output_port->get_func_expr(), TermName(input_pin));
}

auto NetHasDirectClockSinks(const SdcLibertyCellLookup& liberty_cell_lookup, idb::IdbNet* net) -> bool
{
  return net != nullptr && IsClockTarget(CountDirectClockSinks(liberty_cell_lookup, net));
}

auto CountInputPinsOnNet(idb::IdbInstance* inst, idb::IdbNet* net) -> std::size_t
{
  if (inst == nullptr || net == nullptr) {
    return 0U;
  }
  std::size_t input_pin_count = 0U;
  for (auto* input_pin : CollectInputPins(inst)) {
    if (input_pin != nullptr && input_pin->get_net() == net) {
      ++input_pin_count;
    }
  }
  return input_pin_count;
}

auto CountClockTargetOutputs(const SdcLibertyCellLookup& liberty_cell_lookup, const std::vector<idb::IdbPin*>& output_pins) -> std::size_t
{
  std::size_t clock_target_output_count = 0U;
  for (auto* output_pin : output_pins) {
    auto* output_net = output_pin == nullptr ? nullptr : output_pin->get_net();
    if (NetHasDirectClockSinks(liberty_cell_lookup, output_net)) {
      ++clock_target_output_count;
    }
  }
  return clock_target_output_count;
}

auto LibertyMarksClockInput(idb::IdbPin* input_pin, idb::LibCell* lib_cell) -> bool
{
  if (input_pin == nullptr || lib_cell == nullptr) {
    return false;
  }
  auto* lib_port = FindLibPort(lib_cell, input_pin);
  return lib_port != nullptr && (lib_port->isClock() || lib_port->get_clock_gate_clock_pin());
}

auto OtherInputsAreControlCandidates(idb::IdbInstance* inst, idb::IdbPin* clock_input_pin, idb::LibCell* lib_cell,
                                     const CaseConstraintSet& case_constraints, const SdcLibertyCellLookup& liberty_cell_lookup) -> bool
{
  bool has_other_input = false;
  for (auto* input_pin : CollectInputPins(inst)) {
    if (input_pin == nullptr || input_pin == clock_input_pin) {
      continue;
    }
    has_other_input = true;
    if (IsCaseConstrained(input_pin, case_constraints)) {
      continue;
    }
    if (LibertyMarksClockInput(input_pin, lib_cell) || NetHasDirectClockSinks(liberty_cell_lookup, input_pin->get_net())) {
      return false;
    }
  }
  return has_other_input;
}

auto AddOutputTransition(std::vector<TraceTransition>& transitions, idb::IdbPin* output_pin, const std::string& reason) -> void
{
  if (output_pin == nullptr || output_pin->get_net() == nullptr) {
    return;
  }
  transitions.push_back({
      output_pin->get_net(),
      PinDisplayName(output_pin) + "->" + output_pin->get_net()->get_net_name(),
      reason,
  });
}

auto CollectSafeTransitions(const SdcLibertyCellLookup& liberty_cell_lookup, idb::IdbNet* net, const CaseConstraintSet& case_constraints)
    -> std::vector<TraceTransition>
{
  std::vector<TraceTransition> transitions;
  const auto net_pins = CollectNetPins(net);
  for (auto* load_pin : net_pins.loads) {
    if (load_pin == nullptr || load_pin->is_io_pin() || !IsInputLike(load_pin)) {
      continue;
    }
    auto* inst = load_pin->get_instance();
    auto* lib_cell = FindLibCell(liberty_cell_lookup, inst);
    if (inst == nullptr || IsSequentialCell(inst, lib_cell)) {
      continue;
    }

    if (lib_cell != nullptr && (lib_cell->isBuffer() || lib_cell->isInverter())) {
      idb::LibPort* input_port = nullptr;
      idb::LibPort* output_port = nullptr;
      lib_cell->bufferPorts(input_port, output_port);
      if (input_port != nullptr && output_port != nullptr) {
        auto* input_pin = ResolveInstPinByLibPort(inst, input_port);
        auto* output_pin = ResolveInstPinByLibPort(inst, output_port);
        if (input_pin == load_pin) {
          AddOutputTransition(transitions, output_pin, lib_cell->isBuffer() ? "buffer" : "inverter");
        }
      }
      continue;
    }

    if (lib_cell != nullptr && lib_cell->isICG()) {
      auto* input_port = FindLibPort(lib_cell, load_pin);
      const bool is_clock_gate_clock_pin = input_port != nullptr && (input_port->get_clock_gate_clock_pin() || input_port->isClock());
      auto* term = load_pin->get_term();
      if (is_clock_gate_clock_pin || (term != nullptr && term->get_type() == idb::IdbConnectType::kClock)) {
        for (auto* output_pin : CollectOutputPins(inst)) {
          AddOutputTransition(transitions, output_pin, "clock_gate");
        }
      }
      continue;
    }

    const auto input_pins = CollectInputPins(inst);
    const auto output_pins = CollectOutputPins(inst);
    const bool single_input_output_trace = input_pins.size() == 1U && input_pins.front() == load_pin && output_pins.size() == 1U;
    const bool constrained_gate = OtherInputsCaseConstrained(inst, load_pin, case_constraints);
    const bool single_clock_provenance_input = CountInputPinsOnNet(inst, net) == 1U && load_pin->get_net() == net;
    const auto clock_target_output_count = CountClockTargetOutputs(liberty_cell_lookup, output_pins);
    const bool constrained_output_count = output_pins.size() == 1U || clock_target_output_count == 1U;
    const bool control_candidate_gate = constrained_output_count && single_clock_provenance_input
                                        && OtherInputsAreControlCandidates(inst, load_pin, lib_cell, case_constraints, liberty_cell_lookup);
    if (!single_input_output_trace && !constrained_gate && !control_candidate_gate) {
      continue;
    }

    for (auto* output_pin : output_pins) {
      auto* output_net = output_pin == nullptr ? nullptr : output_pin->get_net();
      if (output_net == nullptr) {
        continue;
      }
      const auto output_stats = CountDirectClockSinks(liberty_cell_lookup, output_net);
      if (!OutputFunctionUsesInput(lib_cell, output_pin, load_pin) && !(control_candidate_gate && IsClockTarget(output_stats))) {
        continue;
      }
      if (IsClockTarget(output_stats)) {
        std::string reason = "single_input_clock_buffer_like";
        if (constrained_gate) {
          reason = "case_constrained_comb_clock_gate";
        } else if (control_candidate_gate) {
          reason = "comb_output_direct_clock_sinks";
        }
        AddOutputTransition(transitions, output_pin, reason);
      } else if (output_net->is_clock() && constrained_gate) {
        AddOutputTransition(transitions, output_pin, "case_constrained_clock_net");
      }
    }
  }
  return transitions;
}

}  // namespace

auto ObjectKindName(SdcObjectKind kind) -> std::string
{
  switch (kind) {
    case SdcObjectKind::kPort:
      return "port";
    case SdcObjectKind::kPin:
      return "pin";
    case SdcObjectKind::kNet:
      return "net";
    case SdcObjectKind::kClock:
      return "clock";
    case SdcObjectKind::kUnknown:
      return "unknown";
  }
  return "unknown";
}

auto ResolvePortNet(idb::IdbDesign* idb_design, const std::string& port_name) -> idb::IdbNet*
{
  auto* io_pin_list = idb_design == nullptr ? nullptr : idb_design->get_io_pin_list();
  if (io_pin_list != nullptr) {
    for (auto* pin : io_pin_list->get_pin_list()) {
      if (pin == nullptr) {
        continue;
      }
      if (pin->get_pin_name() == port_name || TermName(pin) == port_name) {
        return pin->get_net();
      }
    }
  }
  auto* net_list = idb_design == nullptr ? nullptr : idb_design->get_net_list();
  return net_list == nullptr ? nullptr : net_list->find_net(port_name);
}

auto ResolvePinNet(idb::IdbDesign* idb_design, const std::string& pin_name) -> idb::IdbNet*
{
  auto separator_pos = pin_name.rfind('/');
  if (separator_pos == std::string::npos) {
    separator_pos = pin_name.rfind(':');
  }
  if (separator_pos == std::string::npos || separator_pos == 0U || separator_pos + 1U >= pin_name.size()) {
    auto* io_pin_list = idb_design == nullptr ? nullptr : idb_design->get_io_pin_list();
    if (io_pin_list == nullptr) {
      return nullptr;
    }
    for (auto* pin : io_pin_list->get_pin_list()) {
      if (pin != nullptr && (pin->get_pin_name() == pin_name || TermName(pin) == pin_name)) {
        return pin->get_net();
      }
    }
    return nullptr;
  }
  auto* inst_list = idb_design == nullptr ? nullptr : idb_design->get_instance_list();
  if (inst_list == nullptr) {
    return nullptr;
  }
  const auto inst_name = pin_name.substr(0U, separator_pos);
  const auto port_name = pin_name.substr(separator_pos + 1U);
  auto* inst = inst_list->find_instance(inst_name);
  if (inst == nullptr || inst->get_pin_list() == nullptr) {
    return nullptr;
  }
  for (auto* pin : inst->get_pin_list()->get_pin_list()) {
    if (pin != nullptr && (pin->get_pin_name() == port_name || TermName(pin) == port_name)) {
      return pin->get_net();
    }
  }
  return nullptr;
}

auto ResolveRefNets(idb::IdbDesign* idb_design, const SdcObjectRef& ref) -> std::vector<idb::IdbNet*>
{
  std::vector<idb::IdbNet*> nets;
  auto* net_list = idb_design == nullptr ? nullptr : idb_design->get_net_list();
  auto append_net = [&nets](idb::IdbNet* net) -> void {
    if (net != nullptr && std::ranges::find(nets, net) == nets.end()) {
      nets.push_back(net);
    }
  };

  switch (ref.kind) {
    case SdcObjectKind::kPort:
      append_net(ResolvePortNet(idb_design, ref.pattern));
      break;
    case SdcObjectKind::kPin:
      append_net(ResolvePinNet(idb_design, ref.pattern));
      break;
    case SdcObjectKind::kNet:
      append_net(net_list == nullptr ? nullptr : net_list->find_net(ref.pattern));
      break;
    case SdcObjectKind::kClock:
      break;
    case SdcObjectKind::kUnknown:
      append_net(net_list == nullptr ? nullptr : net_list->find_net(ref.pattern));
      append_net(ResolvePortNet(idb_design, ref.pattern));
      append_net(ResolvePinNet(idb_design, ref.pattern));
      break;
  }
  return nets;
}

auto BuildCaseConstraintSet(const SdcClockData& clock_data) -> CaseConstraintSet
{
  CaseConstraintSet constraints;
  for (const auto& analysis : clock_data.case_analyses) {
    for (const auto& object : analysis.objects) {
      if (object.kind == SdcObjectKind::kPin) {
        constraints.pin_names.insert(object.pattern);
      } else if (object.kind == SdcObjectKind::kNet) {
        constraints.net_names.insert(object.pattern);
      } else if (object.kind == SdcObjectKind::kUnknown) {
        constraints.pin_names.insert(object.pattern);
        constraints.net_names.insert(object.pattern);
      }
    }
  }
  return constraints;
}

auto BuildGeneratedBoundaryOwners(idb::IdbDesign* idb_design, const SdcClockData& clock_data)
    -> std::unordered_map<idb::IdbNet*, std::string>
{
  std::unordered_map<idb::IdbNet*, std::string> boundary_owner_by_net;
  for (const auto& clock : clock_data.clocks) {
    if (clock.kind != SdcClockDecl::Kind::kGenerated) {
      continue;
    }
    for (const auto& target : clock.targets) {
      for (auto* net : ResolveRefNets(idb_design, target)) {
        boundary_owner_by_net[net] = clock.clock_name;
      }
    }
  }
  return boundary_owner_by_net;
}

auto TraceClock(const SdcLibertyCellLookup& liberty_cell_lookup, idb::IdbDesign* idb_design, const SdcClockDecl& clock,
                const CaseConstraintSet& case_constraints,
                const std::unordered_map<idb::IdbNet*, std::string>& generated_boundary_owner_by_net) -> std::vector<ClockTraceRecord>
{
  std::vector<ClockTraceRecord> records;
  if (clock.is_virtual || clock.targets.empty()) {
    records.push_back({clock.clock_name, "", "skipped", "virtual_clock", 0U, 0U, "", "no_netlist_source"});
    return records;
  }

  std::vector<idb::IdbNet*> seed_nets;
  for (const auto& target : clock.targets) {
    auto resolved_nets = ResolveRefNets(idb_design, target);
    seed_nets.insert(seed_nets.end(), resolved_nets.begin(), resolved_nets.end());
    if (resolved_nets.empty()) {
      records.push_back(
          {clock.clock_name, target.pattern, "rejected", ObjectKindName(target.kind), 0U, 0U, target.pattern, "unresolved_sdc_object"});
    }
  }
  std::ranges::sort(seed_nets);
  const auto unique_seed_nets = std::ranges::unique(seed_nets);
  seed_nets.erase(unique_seed_nets.begin(), unique_seed_nets.end());
  if (seed_nets.empty()) {
    records.push_back({clock.clock_name, "", "rejected", "sdc_source", 0U, 0U, "", "no_resolved_seed_net"});
    return records;
  }

  std::deque<TraceNode> queue;
  std::unordered_set<idb::IdbNet*> visited;
  visited.reserve(1024U);
  for (auto* seed_net : seed_nets) {
    if (seed_net != nullptr) {
      queue.push_back({seed_net, seed_net->get_net_name(), 0U});
    }
  }

  while (!queue.empty()) {
    auto node = queue.front();
    queue.pop_front();
    if (node.net == nullptr || !visited.insert(node.net).second) {
      continue;
    }
    if (visited.size() > kTraceNetVisitLimit) {
      records.push_back({clock.clock_name, NetName(node.net), "rejected", "trace_limit", 0U, 0U, node.path, "trace_net_visit_limit"});
      break;
    }

    if (const auto boundary_iter = generated_boundary_owner_by_net.find(node.net);
        boundary_iter != generated_boundary_owner_by_net.end() && boundary_iter->second != clock.clock_name) {
      records.push_back(
          {clock.clock_name, NetName(node.net), "trace_stop", "generated_clock_boundary", 0U, 0U, node.path, boundary_iter->second});
      continue;
    }

    const auto stats = CountDirectClockSinks(liberty_cell_lookup, node.net);
    if (IsClockTarget(stats)) {
      records.push_back({clock.clock_name, node.net->get_net_name(), "accepted", TargetKind(stats), stats.sequential_clock_sinks,
                         stats.macro_clock_sinks, node.path, "sdc_reachable_clock_sink_net"});
    }

    if (node.depth >= kTraceDepthLimit) {
      records.push_back({clock.clock_name, NetName(node.net), "trace_stop", "depth_limit", stats.sequential_clock_sinks,
                         stats.macro_clock_sinks, node.path, "trace_depth_limit"});
      continue;
    }

    for (const auto& transition : CollectSafeTransitions(liberty_cell_lookup, node.net, case_constraints)) {
      if (transition.net == nullptr || visited.contains(transition.net)) {
        continue;
      }
      auto next_path = node.path;
      if (!next_path.empty()) {
        next_path += " -> ";
      }
      next_path += transition.path_step;
      queue.push_back({transition.net, std::move(next_path), node.depth + 1U});
      records.push_back(
          {clock.clock_name, transition.net->get_net_name(), "trace_through", transition.reason, 0U, 0U, node.path, transition.reason});
    }
  }
  return records;
}

}  // namespace icts::clock_trace
