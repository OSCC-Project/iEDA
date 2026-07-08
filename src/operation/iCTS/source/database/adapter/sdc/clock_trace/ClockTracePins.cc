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
 * @file ClockTracePins.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Pin, liberty, and transition helpers for SDC clock tracing.
 */

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "IdbCellMaster.h"
#include "IdbEnum.h"
#include "IdbInstance.h"
#include "IdbNet.h"
#include "IdbPins.h"
#include "IdbTerm.h"
#include "SdcClockReader.hh"
#include "SdcClockTraceAlgorithm.hh"
#include "liberty/Lib.hh"

namespace icts::clock_trace {

auto NetName(idb::IdbNet* net) -> std::string
{
  return net == nullptr ? std::string{} : net->get_net_name();
}

auto TermName(idb::IdbPin* pin) -> std::string
{
  if (pin == nullptr) {
    return {};
  }
  auto* term = pin->get_term();
  return term == nullptr ? pin->get_pin_name() : term->get_name();
}

auto PinFullName(idb::IdbPin* pin) -> std::string
{
  if (pin == nullptr) {
    return {};
  }
  auto* inst = pin->get_instance();
  if (inst == nullptr) {
    return TermName(pin);
  }
  return inst->get_name() + "/" + TermName(pin);
}

auto PinDisplayName(idb::IdbPin* pin) -> std::string
{
  if (pin == nullptr) {
    return {};
  }
  const auto full_name = PinFullName(pin);
  return full_name.empty() ? pin->get_pin_name() : full_name;
}

namespace {

auto AppendUniquePin(std::vector<idb::IdbPin*>& pins, std::unordered_set<idb::IdbPin*>& seen_pins, idb::IdbPin* pin) -> void
{
  if (pin == nullptr || !seen_pins.insert(pin).second) {
    return;
  }
  pins.push_back(pin);
}

}  // namespace

auto CollectNetPins(idb::IdbNet* net) -> IdbNetPins
{
  IdbNetPins net_pins;
  if (net == nullptr) {
    return net_pins;
  }

  const auto io_pin_count = net->get_io_pins() == nullptr ? 0U : net->get_io_pins()->get_pin_list().size();
  const auto inst_pin_count = net->get_instance_pin_list() == nullptr ? 0U : net->get_instance_pin_list()->get_pin_list().size();
  net_pins.all_pins.reserve(io_pin_count + inst_pin_count);
  std::unordered_set<idb::IdbPin*> seen_pins;
  seen_pins.reserve(io_pin_count + inst_pin_count);
  if (net->get_io_pins() != nullptr) {
    for (auto* pin : net->get_io_pins()->get_pin_list()) {
      AppendUniquePin(net_pins.all_pins, seen_pins, pin);
    }
  }
  if (net->get_instance_pin_list() != nullptr) {
    for (auto* pin : net->get_instance_pin_list()->get_pin_list()) {
      AppendUniquePin(net_pins.all_pins, seen_pins, pin);
    }
  }

  for (auto* pin : net_pins.all_pins) {
    auto* term = pin == nullptr ? nullptr : pin->get_term();
    if (term == nullptr) {
      continue;
    }
    if (!pin->is_io_pin()
        && (term->get_direction() == idb::IdbConnectDirection::kOutput
            || term->get_direction() == idb::IdbConnectDirection::kOutputTriState)) {
      net_pins.driver = pin;
      break;
    }
  }
  if (net_pins.driver == nullptr) {
    for (auto* pin : net_pins.all_pins) {
      auto* term = pin == nullptr ? nullptr : pin->get_term();
      if (term != nullptr && pin->is_io_pin() && term->get_direction() == idb::IdbConnectDirection::kInput) {
        net_pins.driver = pin;
        break;
      }
    }
  }

  for (auto* pin : net_pins.all_pins) {
    if (pin != net_pins.driver) {
      net_pins.loads.push_back(pin);
    }
  }
  return net_pins;
}

auto IsInputLike(idb::IdbPin* pin) -> bool
{
  auto* term = pin == nullptr ? nullptr : pin->get_term();
  if (term == nullptr) {
    return false;
  }
  return term->get_direction() == idb::IdbConnectDirection::kInput || term->get_direction() == idb::IdbConnectDirection::kInOut;
}

auto IsOutputLike(idb::IdbPin* pin) -> bool
{
  auto* term = pin == nullptr ? nullptr : pin->get_term();
  if (term == nullptr) {
    return false;
  }
  return term->get_direction() == idb::IdbConnectDirection::kOutput || term->get_direction() == idb::IdbConnectDirection::kOutputTriState
         || term->get_direction() == idb::IdbConnectDirection::kInOut;
}

auto FindLibCell(const SdcLibertyCellLookup& liberty_cell_lookup, idb::IdbInstance* inst) -> idb::LibCell*
{
  auto* cell_master = inst == nullptr ? nullptr : inst->get_cell_master();
  if (cell_master == nullptr) {
    return nullptr;
  }
  return liberty_cell_lookup ? liberty_cell_lookup(cell_master->get_name()) : nullptr;
}

auto FindLibPort(idb::LibCell* lib_cell, idb::IdbPin* pin) -> idb::LibPort*
{
  if (lib_cell == nullptr || pin == nullptr) {
    return nullptr;
  }
  const auto port_name = TermName(pin);
  if (port_name.empty()) {
    return nullptr;
  }
  return lib_cell->get_cell_port_or_port_bus(port_name.c_str());
}

auto IsSequentialCell(idb::IdbInstance* inst, idb::LibCell* lib_cell) -> bool
{
  if (lib_cell != nullptr && lib_cell->isSequentialCell() && !lib_cell->isICG()) {
    return true;
  }
  return inst != nullptr && inst->is_flip_flop();
}

auto IsClockSinkPin(idb::IdbPin* pin, idb::LibCell* lib_cell) -> bool
{
  if (pin == nullptr || pin->is_io_pin() || !IsInputLike(pin)) {
    return false;
  }
  auto* term = pin->get_term();
  auto* lib_port = FindLibPort(lib_cell, pin);
  if (lib_port != nullptr && lib_port->isClock()) {
    return true;
  }
  if (pin->is_flip_flop_clk()) {
    return true;
  }
  return term != nullptr && term->get_type() == idb::IdbConnectType::kClock && pin->get_instance() != nullptr
         && pin->get_instance()->is_flip_flop();
}

auto IsMacroClockSinkPin(idb::IdbPin* pin, idb::LibCell* lib_cell) -> bool
{
  if (pin == nullptr || pin->is_io_pin() || !IsInputLike(pin)) {
    return false;
  }
  auto* inst = pin->get_instance();
  auto* cell_master = inst == nullptr ? nullptr : inst->get_cell_master();
  if (cell_master == nullptr || !cell_master->is_block()) {
    return false;
  }
  auto* term = pin->get_term();
  auto* lib_port = FindLibPort(lib_cell, pin);
  return (term != nullptr && term->get_type() == idb::IdbConnectType::kClock) || (lib_port != nullptr && lib_port->isClock());
}

auto CountDirectClockSinks(const SdcLibertyCellLookup& liberty_cell_lookup, idb::IdbNet* net) -> ClockSinkStats
{
  ClockSinkStats stats;
  const auto net_pins = CollectNetPins(net);
  for (auto* load_pin : net_pins.loads) {
    auto* inst = load_pin == nullptr ? nullptr : load_pin->get_instance();
    auto* lib_cell = FindLibCell(liberty_cell_lookup, inst);
    if (IsClockSinkPin(load_pin, lib_cell)) {
      ++stats.sequential_clock_sinks;
    } else if (IsMacroClockSinkPin(load_pin, lib_cell)) {
      ++stats.macro_clock_sinks;
    }
  }
  return stats;
}

auto CountDirectClockSinksForReport(const SdcLibertyCellLookup& liberty_cell_lookup, idb::IdbNet* net) -> ClockSinkStats
{
  ClockSinkStats stats;
  auto* pin_list = net == nullptr ? nullptr : net->get_instance_pin_list();
  if (pin_list == nullptr) {
    return stats;
  }
  for (auto* pin : pin_list->get_pin_list()) {
    auto* inst = pin == nullptr ? nullptr : pin->get_instance();
    auto* lib_cell = FindLibCell(liberty_cell_lookup, inst);
    if (IsClockSinkPin(pin, lib_cell)) {
      ++stats.sequential_clock_sinks;
    } else if (IsMacroClockSinkPin(pin, lib_cell)) {
      ++stats.macro_clock_sinks;
    }
  }
  return stats;
}

auto IsClockTarget(const ClockSinkStats& stats) -> bool
{
  return stats.sequential_clock_sinks > 0U || stats.macro_clock_sinks > 0U;
}

auto TargetKind(const ClockSinkStats& stats) -> std::string
{
  if (stats.sequential_clock_sinks > 0U && stats.macro_clock_sinks > 0U) {
    return "sequential_ck+macro_clock";
  }
  if (stats.sequential_clock_sinks > 0U) {
    return "sequential_ck";
  }
  if (stats.macro_clock_sinks > 0U) {
    return "macro_clock";
  }
  return "trace_through";
}

auto ClockKindName(const SdcClockDecl& clock) -> std::string
{
  return clock.kind == SdcClockDecl::Kind::kGenerated ? "generated" : "primary";
}

auto MasterClockName(const SdcClockDecl& clock) -> std::string
{
  if (clock.kind == SdcClockDecl::Kind::kGenerated) {
    return clock.master_clock_name.empty() ? "n/a" : clock.master_clock_name;
  }
  return "self";
}

auto DominanceForRecord(const ClockTraceRecord& record, const std::string& clock_kind) -> std::string
{
  if (record.status == "ambiguous") {
    return "ambiguous_sdc_ownership";
  }
  if (record.target_kind == "generated_clock_boundary") {
    return "blocked_by_generated_clock";
  }
  if (record.status == "skipped") {
    return "no_netlist_source";
  }
  if (record.status == "rejected") {
    return "no_physical_dominance";
  }
  if (clock_kind == "generated") {
    return "owned_by_generated_clock";
  }
  if (clock_kind == "primary") {
    return "owned_by_primary_clock";
  }
  return "undetermined";
}

auto StrongTargetSinkThreshold(std::size_t max_fanout) -> std::size_t
{
  constexpr std::size_t min_clock_target_sinks = 4U;
  return std::max(min_clock_target_sinks, max_fanout);
}

auto IsStrongClockTarget(const ClockTraceRecord& record, std::size_t sink_threshold) -> bool
{
  return record.macro_clock_sinks > 0U || record.sequential_clock_sinks > sink_threshold;
}

auto ResolveInstPinByLibPort(idb::IdbInstance* inst, idb::LibPort* lib_port) -> idb::IdbPin*
{
  if (inst == nullptr || lib_port == nullptr || inst->get_pin_list() == nullptr) {
    return nullptr;
  }
  const std::string port_name = lib_port->get_port_name();
  for (auto* pin : inst->get_pin_list()->get_pin_list()) {
    if (pin == nullptr) {
      continue;
    }
    if (TermName(pin) == port_name || pin->get_pin_name() == port_name) {
      return pin;
    }
  }
  return nullptr;
}

auto BuildPreclusteredSinkAnchor(const SdcLibertyCellLookup& liberty_cell_lookup, idb::IdbNet* leaf_net)
    -> std::optional<ClockTracePreclusteredSinkAnchor>
{
  if (leaf_net == nullptr || !IsClockTarget(CountDirectClockSinks(liberty_cell_lookup, leaf_net))) {
    return std::nullopt;
  }

  const auto net_pins = CollectNetPins(leaf_net);
  auto* output_pin = net_pins.driver;
  auto* driver_inst = output_pin == nullptr ? nullptr : output_pin->get_instance();
  auto* lib_cell = FindLibCell(liberty_cell_lookup, driver_inst);
  if (driver_inst == nullptr || lib_cell == nullptr || !(lib_cell->isBuffer() || lib_cell->isInverter())) {
    return std::nullopt;
  }

  idb::LibPort* input_port = nullptr;
  idb::LibPort* output_port = nullptr;
  lib_cell->bufferPorts(input_port, output_port);
  if (input_port == nullptr || output_port == nullptr) {
    return std::nullopt;
  }

  auto* input_pin = ResolveInstPinByLibPort(driver_inst, input_port);
  auto* resolved_output_pin = ResolveInstPinByLibPort(driver_inst, output_port);
  if (input_pin == nullptr || resolved_output_pin == nullptr || resolved_output_pin != output_pin || input_pin->get_net() == nullptr) {
    return std::nullopt;
  }

  return ClockTracePreclusteredSinkAnchor{
      .leaf_net_name = leaf_net->get_net_name(),
      .driver_inst_name = driver_inst->get_name(),
      .input_pin_name = TermName(input_pin),
      .output_pin_name = TermName(output_pin),
      .input_net_name = input_pin->get_net()->get_net_name(),
  };
}

}  // namespace icts::clock_trace
