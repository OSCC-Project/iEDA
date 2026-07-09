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
 * @file WrapperClockReader.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Clock readback helpers for the iCTS iDB wrapper
 */
#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "IdbCellMaster.h"
#include "IdbDesign.h"
#include "IdbEnum.h"
#include "IdbInstance.h"
#include "IdbNet.h"
#include "IdbPins.h"
#include "IdbTerm.h"
// use iEDA Rust liberty parser
#include "Log.hh"
#include "Vector.hh"
#include "Wrapper.hh"
#include "adapter/sdc/SdcClockReader.hh"
#include "builder.h"
#include "def_service.h"
#include "design/Clock.hh"
#include "design/Design.hh"
#include "design/Inst.hh"
#include "design/Net.hh"
#include "design/Pin.hh"
#include "liberty/Lib.hh"
#include "logger/Schema.hh"

namespace icts {
namespace {

auto convertIdbPinType(idb::IdbConnectType idb_pin_type, idb::IdbConnectDirection idb_pin_direction) -> PinType
{
  if (idb_pin_type == idb::IdbConnectType::kClock) {
    return PinType::kClock;
  }
  if (idb_pin_direction == idb::IdbConnectDirection::kOutput) {
    return PinType::kOut;
  }
  if (idb_pin_direction == idb::IdbConnectDirection::kInput) {
    return PinType::kIn;
  }
  if (idb_pin_direction == idb::IdbConnectDirection::kInOut) {
    return PinType::kInOut;
  }
  return PinType::kOther;
}

struct IdbClockNetPins
{
  idb::IdbPin* driver = nullptr;
  std::vector<idb::IdbPin*> loads;
};

struct CtsInstClassification
{
  InstType type = InstType::kUnknown;
  std::string role = "unknown_boundary";
  std::string reason = "unclassified";
  std::string input_pin_name;
  std::string output_pin_name;
};

auto appendUniqueIdbPin(std::vector<idb::IdbPin*>& pins, std::unordered_set<idb::IdbPin*>& seen_pins, idb::IdbPin* idb_pin) -> void
{
  if (idb_pin == nullptr || !seen_pins.insert(idb_pin).second) {
    return;
  }
  pins.push_back(idb_pin);
}

auto collectIdbClockNetPins(idb::IdbNet* idb_net) -> IdbClockNetPins
{
  IdbClockNetPins net_pins;
  if (idb_net == nullptr) {
    return net_pins;
  }

  std::vector<idb::IdbPin*> all_pins;
  const auto io_pin_count = idb_net->get_io_pins() == nullptr ? 0U : idb_net->get_io_pins()->get_pin_list().size();
  const auto inst_pin_count = idb_net->get_instance_pin_list() == nullptr ? 0U : idb_net->get_instance_pin_list()->get_pin_list().size();
  all_pins.reserve(io_pin_count + inst_pin_count);
  std::unordered_set<idb::IdbPin*> seen_pins;
  seen_pins.reserve(io_pin_count + inst_pin_count);
  if (idb_net->get_io_pins() != nullptr) {
    for (auto* idb_pin : idb_net->get_io_pins()->get_pin_list()) {
      appendUniqueIdbPin(all_pins, seen_pins, idb_pin);
    }
  }
  if (idb_net->get_instance_pin_list() != nullptr) {
    for (auto* idb_pin : idb_net->get_instance_pin_list()->get_pin_list()) {
      appendUniqueIdbPin(all_pins, seen_pins, idb_pin);
    }
  }

  for (auto* idb_pin : all_pins) {
    auto* idb_term = idb_pin == nullptr ? nullptr : idb_pin->get_term();
    if (idb_term == nullptr) {
      continue;
    }
    if (!idb_pin->is_io_pin()
        && (idb_term->get_direction() == idb::IdbConnectDirection::kOutput
            || idb_term->get_direction() == idb::IdbConnectDirection::kOutputTriState)) {
      net_pins.driver = idb_pin;
      break;
    }
  }

  if (net_pins.driver == nullptr) {
    for (auto* idb_pin : all_pins) {
      auto* idb_term = idb_pin == nullptr ? nullptr : idb_pin->get_term();
      if (idb_term != nullptr && idb_pin->is_io_pin() && idb_term->get_direction() == idb::IdbConnectDirection::kInput) {
        net_pins.driver = idb_pin;
        break;
      }
    }
  }

  if (net_pins.driver == nullptr) {
    for (auto* idb_pin : all_pins) {
      auto* idb_term = idb_pin == nullptr ? nullptr : idb_pin->get_term();
      if (idb_term != nullptr && idb_pin->is_io_pin() && idb_term->get_direction() == idb::IdbConnectDirection::kInOut) {
        net_pins.driver = idb_pin;
        break;
      }
    }
  }

  if (net_pins.driver == nullptr && !all_pins.empty()) {
    net_pins.driver = all_pins.front();
  }

  for (auto* idb_pin : all_pins) {
    if (idb_pin != net_pins.driver) {
      net_pins.loads.push_back(idb_pin);
    }
  }
  return net_pins;
}

auto termName(idb::IdbPin* idb_pin) -> std::string
{
  auto* term = idb_pin == nullptr ? nullptr : idb_pin->get_term();
  return term == nullptr ? std::string{} : term->get_name();
}

auto isInputLike(idb::IdbPin* idb_pin) -> bool
{
  auto* term = idb_pin == nullptr ? nullptr : idb_pin->get_term();
  if (term == nullptr) {
    return false;
  }
  return term->get_direction() == idb::IdbConnectDirection::kInput || term->get_direction() == idb::IdbConnectDirection::kInOut;
}

auto isOutputLike(idb::IdbPin* idb_pin) -> bool
{
  auto* term = idb_pin == nullptr ? nullptr : idb_pin->get_term();
  if (term == nullptr) {
    return false;
  }
  return term->get_direction() == idb::IdbConnectDirection::kOutput || term->get_direction() == idb::IdbConnectDirection::kOutputTriState
         || term->get_direction() == idb::IdbConnectDirection::kInOut;
}

auto findLibCell(const Wrapper& wrapper, idb::IdbInstance* idb_inst) -> idb::LibCell*
{
  auto* cell_master = idb_inst == nullptr ? nullptr : idb_inst->get_cell_master();
  if (cell_master == nullptr) {
    return nullptr;
  }
  return wrapper.findLibertyCell(cell_master->get_name());
}

auto findLibPort(idb::LibCell* lib_cell, idb::IdbPin* idb_pin) -> idb::LibPort*
{
  if (lib_cell == nullptr || idb_pin == nullptr) {
    return nullptr;
  }
  const auto port_name = termName(idb_pin);
  if (port_name.empty()) {
    return nullptr;
  }
  return lib_cell->get_cell_port_or_port_bus(port_name.c_str());
}

auto resolveInstPinByLibPort(idb::IdbInstance* idb_inst, idb::LibPort* lib_port) -> idb::IdbPin*
{
  if (idb_inst == nullptr || lib_port == nullptr || idb_inst->get_pin_list() == nullptr) {
    return nullptr;
  }
  const std::string port_name = lib_port->get_port_name();
  for (auto* idb_pin : idb_inst->get_pin_list()->get_pin_list()) {
    if (idb_pin == nullptr) {
      continue;
    }
    if (termName(idb_pin) == port_name || idb_pin->get_pin_name() == port_name) {
      return idb_pin;
    }
  }
  return nullptr;
}

auto libertyExpressionUsesPort(RustLibertyExpr* expression, const std::string& port_name) -> bool
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

auto outputFunctionUsesInput(idb::LibCell* lib_cell, idb::IdbPin* output_pin, idb::IdbPin* input_pin) -> bool
{
  if (lib_cell == nullptr || output_pin == nullptr || input_pin == nullptr) {
    return false;
  }
  auto* output_port = findLibPort(lib_cell, output_pin);
  if (output_port == nullptr || output_port->get_func_expr() == nullptr) {
    return false;
  }
  return libertyExpressionUsesPort(output_port->get_func_expr(), termName(input_pin));
}

auto collectInputPins(idb::IdbInstance* idb_inst) -> std::vector<idb::IdbPin*>
{
  std::vector<idb::IdbPin*> input_pins;
  if (idb_inst == nullptr || idb_inst->get_pin_list() == nullptr) {
    return input_pins;
  }
  for (auto* idb_pin : idb_inst->get_pin_list()->get_pin_list()) {
    if (isInputLike(idb_pin)) {
      input_pins.push_back(idb_pin);
    }
  }
  return input_pins;
}

auto collectOutputPins(idb::IdbInstance* idb_inst) -> std::vector<idb::IdbPin*>
{
  std::vector<idb::IdbPin*> output_pins;
  if (idb_inst == nullptr || idb_inst->get_pin_list() == nullptr) {
    return output_pins;
  }
  for (auto* idb_pin : idb_inst->get_pin_list()->get_pin_list()) {
    if (isOutputLike(idb_pin)) {
      output_pins.push_back(idb_pin);
    }
  }
  return output_pins;
}

auto collectClockInputPins(idb::IdbInstance* idb_inst) -> std::vector<idb::IdbPin*>
{
  std::vector<idb::IdbPin*> clock_input_pins;
  for (auto* idb_pin : collectInputPins(idb_inst)) {
    auto* idb_net = idb_pin == nullptr ? nullptr : idb_pin->get_net();
    if (idb_net != nullptr && idb_net->is_clock()) {
      clock_input_pins.push_back(idb_pin);
    }
  }
  return clock_input_pins;
}

auto isLibertyClockPin(idb::LibCell* lib_cell, idb::IdbPin* idb_pin) -> bool
{
  auto* lib_port = findLibPort(lib_cell, idb_pin);
  return lib_port != nullptr && (lib_port->isClock() || lib_port->get_is_clock_pin());
}

auto isClockGateClockPin(idb::LibCell* lib_cell, idb::IdbPin* idb_pin) -> bool
{
  auto* lib_port = findLibPort(lib_cell, idb_pin);
  return lib_port != nullptr && (lib_port->get_clock_gate_clock_pin() || lib_port->isClock());
}

auto isSequentialCheckClockPin(idb::LibCell* lib_cell, idb::IdbPin* idb_pin) -> bool
{
  if (lib_cell == nullptr || idb_pin == nullptr) {
    return false;
  }
  const auto pin_name = termName(idb_pin);
  if (pin_name.empty()) {
    return false;
  }
  for (auto& arc_set : lib_cell->get_cell_arcs()) {
    if (arc_set == nullptr) {
      continue;
    }
    for (auto& arc : arc_set->get_arcs()) {
      auto* lib_arc = arc.get();
      if (lib_arc != nullptr && lib_arc->isCheckArc() != 0U && pin_name == lib_arc->get_src_port()) {
        return true;
      }
    }
  }
  return false;
}

auto isSequentialClockToOutputArc(idb::LibCell* lib_cell, idb::IdbPin* idb_pin) -> bool
{
  if (lib_cell == nullptr || idb_pin == nullptr) {
    return false;
  }
  const auto pin_name = termName(idb_pin);
  if (pin_name.empty()) {
    return false;
  }
  for (auto& arc_set : lib_cell->get_cell_arcs()) {
    if (arc_set == nullptr) {
      continue;
    }
    for (auto& arc : arc_set->get_arcs()) {
      auto* lib_arc = arc.get();
      if (lib_arc == nullptr || pin_name != lib_arc->get_src_port()) {
        continue;
      }
      const auto timing_type = lib_arc->get_timing_type();
      if (timing_type != idb::LibArc::TimingType::kRisingEdge && timing_type != idb::LibArc::TimingType::kFallingEdge) {
        continue;
      }
      auto* output_port = lib_cell->get_cell_port_or_port_bus(lib_arc->get_snk_port());
      if (output_port != nullptr && output_port->isOutput() != 0U) {
        return true;
      }
    }
  }
  return false;
}

auto isCombinationalTimingType(idb::LibArc::TimingType timing_type) -> bool
{
  return timing_type == idb::LibArc::TimingType::kComb || timing_type == idb::LibArc::TimingType::kCombRise
         || timing_type == idb::LibArc::TimingType::kCombFall || timing_type == idb::LibArc::TimingType::kDefault;
}

auto hasTransparentDataArcForClockPin(idb::LibCell* lib_cell, idb::IdbPin* clock_pin) -> bool
{
  if (lib_cell == nullptr || clock_pin == nullptr) {
    return false;
  }
  const auto clock_pin_name = termName(clock_pin);
  if (clock_pin_name.empty()) {
    return false;
  }
  for (auto& arc_set : lib_cell->get_cell_arcs()) {
    if (arc_set == nullptr) {
      continue;
    }
    for (auto& arc : arc_set->get_arcs()) {
      auto* lib_arc = arc.get();
      if (lib_arc == nullptr || !isCombinationalTimingType(lib_arc->get_timing_type())) {
        continue;
      }
      if (clock_pin_name == lib_arc->get_src_port()) {
        continue;
      }
      auto* src_port = lib_cell->get_cell_port_or_port_bus(lib_arc->get_src_port());
      auto* snk_port = lib_cell->get_cell_port_or_port_bus(lib_arc->get_snk_port());
      if (src_port != nullptr && src_port->isInput() != 0U && snk_port != nullptr && snk_port->isOutput() != 0U) {
        return true;
      }
    }
  }
  return false;
}

auto hasSequentialClockPinEvidence(idb::LibCell* lib_cell, idb::IdbPin* idb_pin) -> bool
{
  return isLibertyClockPin(lib_cell, idb_pin) || isSequentialCheckClockPin(lib_cell, idb_pin)
         || isSequentialClockToOutputArc(lib_cell, idb_pin) || (idb_pin != nullptr && idb_pin->is_flip_flop_clk());
}

auto classifySequentialInst(idb::LibCell* lib_cell, idb::IdbPin* primary_clock_pin) -> CtsInstClassification
{
  if (primary_clock_pin != nullptr && hasTransparentDataArcForClockPin(lib_cell, primary_clock_pin)) {
    return {.type = InstType::kLatch,
            .role = "latch_sink",
            .reason = "liberty_latch_transparent_data_arc",
            .input_pin_name = {},
            .output_pin_name = {}};
  }
  if (primary_clock_pin != nullptr && hasSequentialClockPinEvidence(lib_cell, primary_clock_pin)) {
    return {.type = InstType::kFlipFlop,
            .role = "sequential_sink",
            .reason = "liberty_sequential_clock_pin",
            .input_pin_name = {},
            .output_pin_name = {}};
  }
  return {.type = InstType::kFlipFlop,
          .role = "sequential_sink",
          .reason = "liberty_sequential_cell",
          .input_pin_name = {},
          .output_pin_name = {}};
}

auto countDirectClockSinks(const Wrapper& wrapper, idb::IdbNet* idb_net) -> std::size_t
{
  std::size_t sink_count = 0U;
  if (idb_net == nullptr || idb_net->get_instance_pin_list() == nullptr) {
    return sink_count;
  }
  for (auto* idb_pin : idb_net->get_instance_pin_list()->get_pin_list()) {
    if (idb_pin == nullptr || idb_pin->is_io_pin() || !isInputLike(idb_pin)) {
      continue;
    }
    auto* inst = idb_pin->get_instance();
    auto* lib_cell = findLibCell(wrapper, inst);
    auto* cell_master = inst == nullptr ? nullptr : inst->get_cell_master();
    if ((lib_cell != nullptr && lib_cell->isSequentialCell() && !lib_cell->isICG() && hasSequentialClockPinEvidence(lib_cell, idb_pin))
        || (cell_master != nullptr && cell_master->is_block() && isLibertyClockPin(lib_cell, idb_pin)) || idb_pin->is_flip_flop_clk()) {
      ++sink_count;
    }
  }
  return sink_count;
}

auto hasClockSinkOutput(const Wrapper& wrapper, idb::LibCell* lib_cell, idb::IdbInstance* idb_inst, idb::IdbPin* clock_input_pin) -> bool
{
  for (auto* output_pin : collectOutputPins(idb_inst)) {
    if (!outputFunctionUsesInput(lib_cell, output_pin, clock_input_pin)) {
      continue;
    }
    auto* output_net = output_pin == nullptr ? nullptr : output_pin->get_net();
    if (countDirectClockSinks(wrapper, output_net) > 0U) {
      return true;
    }
  }
  return false;
}

auto makeBoundaryClassification(const std::string& role, const std::string& reason) -> CtsInstClassification
{
  return CtsInstClassification{
      .type = role == "clock_logic_boundary" ? InstType::kClockLogic : InstType::kBoundaryLoad,
      .role = role,
      .reason = reason,
      .input_pin_name = {},
      .output_pin_name = {},
  };
}

auto classifyCtsInstFromIdbInst(const Wrapper& wrapper, idb::IdbInstance* idb_inst) -> CtsInstClassification
{
  if (idb_inst == nullptr) {
    return {.type = InstType::kUnknown, .role = "unknown_boundary", .reason = "null_idb_inst", .input_pin_name = {}, .output_pin_name = {}};
  }
  auto* cell_master = idb_inst->get_cell_master();
  if (cell_master != nullptr && cell_master->is_block()) {
    return {.type = InstType::kMacroBlock,
            .role = "macro_clock_sink",
            .reason = "idb_macro_block",
            .input_pin_name = {},
            .output_pin_name = {}};
  }

  auto* lib_cell = findLibCell(wrapper, idb_inst);
  const auto clock_input_pins = collectClockInputPins(idb_inst);
  auto* primary_clock_pin = clock_input_pins.empty() ? nullptr : clock_input_pins.front();

  if (lib_cell != nullptr && lib_cell->isICG()) {
    return {.type = InstType::kClockGate,
            .role = "integrated_clock_gate",
            .reason = primary_clock_pin == nullptr || isClockGateClockPin(lib_cell, primary_clock_pin) ? "liberty_clock_gate"
                                                                                                       : "liberty_clock_gate_cell",
            .input_pin_name = {},
            .output_pin_name = {}};
  }

  if (lib_cell != nullptr && lib_cell->isSequentialCell()) {
    return classifySequentialInst(lib_cell, primary_clock_pin);
  }

  if (lib_cell != nullptr && (lib_cell->isBuffer() || lib_cell->isInverter())) {
    idb::LibPort* input_port = nullptr;
    idb::LibPort* output_port = nullptr;
    lib_cell->bufferPorts(input_port, output_port);
    auto* input_pin = resolveInstPinByLibPort(idb_inst, input_port);
    auto* output_pin = resolveInstPinByLibPort(idb_inst, output_port);
    if (input_pin != nullptr && output_pin != nullptr) {
      return {.type = lib_cell->isBuffer() ? InstType::kBuffer : InstType::kInverter,
              .role = lib_cell->isBuffer() ? "clock_buffer" : "clock_inverter",
              .reason = lib_cell->isBuffer() ? "liberty_buffer" : "liberty_inverter",
              .input_pin_name = termName(input_pin),
              .output_pin_name = termName(output_pin)};
    }
  }

  if (clock_input_pins.size() > 1U) {
    return {
        .type = InstType::kMux, .role = "clock_mux", .reason = "multi_clock_input_boundary", .input_pin_name = {}, .output_pin_name = {}};
  }

  if (primary_clock_pin != nullptr && hasClockSinkOutput(wrapper, lib_cell, idb_inst, primary_clock_pin)) {
    return makeBoundaryClassification("clock_logic_boundary", "clock_dependent_output_feeds_clock_sinks");
  }

  if (idb_inst->is_flip_flop()) {
    return {.type = InstType::kFlipFlop, .role = "sequential_sink", .reason = "idb_flip_flop", .input_pin_name = {}, .output_pin_name = {}};
  }

  if (idb_inst->is_clock_instance()) {
    return makeBoundaryClassification("clock_load_boundary", "clock_net_boundary_load");
  }
  return {
      .type = InstType::kUnknown, .role = "non_clock_unknown", .reason = "non_clock_unknown", .input_pin_name = {}, .output_pin_name = {}};
}

auto ctsPinFullName(idb::IdbPin* idb_pin, Inst* cts_inst) -> std::string
{
  const auto pin_name = termName(idb_pin);
  if (pin_name.empty()) {
    return {};
  }
  return cts_inst == nullptr ? pin_name : cts_inst->get_name() + "/" + pin_name;
}

}  // namespace

class Wrapper::CtsClockReader
{
 public:
  CtsClockReader(Wrapper& wrapper, Design& design, SchemaWriter& reporter) : _wrapper(&wrapper), _design(&design), _reporter(&reporter) {}

  auto readClocks(const std::vector<std::pair<std::string, std::string>>& clock_net_pairs) -> bool
  {
    auto* idb_design = findIdbDesign();
    if (idb_design == nullptr) {
      LOG_ERROR << "CTS clock read failed: iDB design is null.";
      return false;
    }

    auto* idb_net_list = idb_design->get_net_list();
    if (idb_net_list == nullptr) {
      LOG_ERROR << "CTS clock read failed: iDB net list is null.";
      return false;
    }

    clearClockReadData();
    for (const auto& [clock_name, clock_net_name] : clock_net_pairs) {
      auto* idb_net = findSdcClockNetOrError(clock_name, clock_net_name, idb_net_list);
      if (idb_net == nullptr || buildClockFromIdbNet(clock_name, clock_net_name, idb_net) == nullptr) {
        clearClockReadData();
        return false;
      }
    }
    emitInstClassificationSummary(_classification_count_by_key);
    return true;
  }

  auto readTraceClockTargets(const std::vector<ClockTraceClockTarget>& clock_targets) -> bool
  {
    auto* idb_design = findIdbDesign();
    if (idb_design == nullptr) {
      LOG_ERROR << "CTS clock read failed: iDB design is null.";
      return false;
    }

    auto* idb_net_list = idb_design->get_net_list();
    if (idb_net_list == nullptr) {
      LOG_ERROR << "CTS clock read failed: iDB net list is null.";
      return false;
    }

    clearClockReadData();
    for (const auto& clock_target : clock_targets) {
      auto* idb_net = findSdcClockNetOrError(clock_target.clock_name, clock_target.clock_net_name, idb_net_list);
      if (idb_net == nullptr) {
        clearClockReadData();
        return false;
      }
      Clock* clock = nullptr;
      if (clock_target.preclustered_sink_reuse) {
        clock = buildPreclusteredClockFromTraceTarget(clock_target, idb_net);
      } else {
        clock = buildClockFromIdbNet(clock_target.clock_name, clock_target.clock_net_name, idb_net);
      }
      if (clock == nullptr) {
        clearClockReadData();
        return false;
      }
    }
    emitInstClassificationSummary(_classification_count_by_key);
    return true;
  }

 private:
  auto clearClockReadData() -> void
  {
    _wrapper->_cts2idb_inst_map.clear();
    _wrapper->_idb2cts_inst_map.clear();
    _wrapper->_cts2idb_net_map.clear();
    _wrapper->_idb2cts_net_map.clear();
    _wrapper->_cts2idb_pin_map.clear();
    _wrapper->_idb2cts_pin_map.clear();
    _design->clearClocks();
    _design->clearTopologyObjects();
    _classification_count_by_key.clear();
  }

  auto recordInstClassification(const CtsInstClassification& classification) -> void
  {
    ++_classification_count_by_key[classification.role + ":" + classification.reason];
  }

  auto emitInstClassificationSummary(const std::map<std::string, std::size_t>& classification_count_by_key) const -> void
  {
    if (classification_count_by_key.empty()) {
      return;
    }
    TableRows rows;
    rows.reserve(classification_count_by_key.size());
    for (const auto& [key, count] : classification_count_by_key) {
      const auto separator_pos = key.find(':');
      const auto role = separator_pos == std::string::npos ? key : key.substr(0U, separator_pos);
      const auto reason = separator_pos == std::string::npos ? std::string{} : key.substr(separator_pos + 1U);
      rows.push_back({role, reason, std::to_string(count)});
    }
    EmitTable(*_reporter, "CTS Inst Classification Summary", {"Role", "Reason", "Count"}, rows);
  }

  auto findIdbDesign() -> idb::IdbDesign*
  {
    auto* idb_design = _wrapper->_idb_design;
    if (idb_design == nullptr && _wrapper->_idb != nullptr && _wrapper->_idb->get_def_service() != nullptr) {
      idb_design = _wrapper->_idb->get_def_service()->get_design();
      _wrapper->_idb_design = idb_design;
    }
    return idb_design;
  }

  auto findSdcClockNetOrError(const std::string& clock_name, const std::string& clock_net_name, idb::IdbNetList* idb_net_list) const
      -> idb::IdbNet*
  {
    auto* idb_net = idb_net_list->find_net(clock_net_name);
    if (idb_net == nullptr) {
      EmitDiagnostic(*_reporter, DiagnosticLevel::kError, "Wrapper", "failed to resolve SDC-declared clock net in iDB.",
                     {{"clock", clock_name}, {"net", clock_net_name}, {"reason", "unresolved_sdc_clock_source"}});
      LOG_ERROR << "CTS clock read failed for clock \"" << clock_name << "\": SDC-declared net \"" << clock_net_name
                << "\" is not found in iDB.";
      return nullptr;
    }
    return idb_net;
  }

  auto buildClockFromIdbNet(const std::string& clock_name, const std::string& clock_net_name, idb::IdbNet* idb_net) -> Clock*
  {
    if (idb_net == nullptr) {
      return nullptr;
    }

    auto* clock = _design->makeClock(clock_name, clock_net_name);
    if (clock == nullptr) {
      LOG_ERROR << "CTS clock read failed for clock \"" << clock_name << "\": failed to create CTS clock object.";
      return nullptr;
    }
    clock->set_clock_name(clock_name);
    clock->set_clock_net_name(clock_net_name);
    clock->set_clock_source(nullptr);
    clock->set_clock_source_net(nullptr);
    clock->set_preclustered_sink_reuse(false);
    clock->clear_preclustered_anchor_input_net_names();
    clock->clear_loads();
    clock->clearMembership();

    auto* cts_net = buildNetFromIdbNet(idb_net);
    if (cts_net == nullptr) {
      return nullptr;
    }
    cts_net->set_driver(nullptr);
    cts_net->set_loads({});
    clock->set_clock_source_net(cts_net);

    const auto idb_net_pins = collectIdbClockNetPins(idb_net);
    if (idb_net_pins.driver == nullptr) {
      LOG_ERROR << "CTS clock read failed for clock \"" << clock_name << "\": iDB net \"" << clock_net_name
                << "\" has no resolvable driver pin.";
      return nullptr;
    }

    std::vector<idb::IdbPin*> idb_pins;
    idb_pins.reserve(idb_net_pins.loads.size() + 1U);
    idb_pins.push_back(idb_net_pins.driver);
    std::ranges::copy(idb_net_pins.loads, std::back_inserter(idb_pins));

    std::unordered_map<idb::IdbInstance*, Inst*> cts_inst_by_idb;
    cts_inst_by_idb.reserve(idb_pins.size());
    std::vector<Pin*> cts_loads;
    cts_loads.reserve(idb_net_pins.loads.size());
    for (auto* idb_pin : idb_pins) {
      if (idb_pin == nullptr) {
        continue;
      }

      Inst* cts_inst = nullptr;
      if (auto* idb_inst = idb_pin->get_instance(); idb_inst != nullptr) {
        if (cts_inst_by_idb.contains(idb_inst)) {
          cts_inst = cts_inst_by_idb.at(idb_inst);
        } else {
          cts_inst = buildInstFromIdbInst(idb_inst);
          if (cts_inst == nullptr) {
            return nullptr;
          }
          cts_inst_by_idb[idb_inst] = cts_inst;
        }
      } else if (!idb_pin->is_io_pin()) {
        LOG_ERROR << "CTS clock read failed for clock \"" << clock_name << "\": instance pin \"" << idb_pin->get_pin_name()
                  << "\" has no iDB inst.";
        return nullptr;
      }

      auto* cts_pin = buildPinFromIdbPin(idb_pin, cts_inst);
      if (cts_pin == nullptr) {
        return nullptr;
      }
      cts_pin->set_net(cts_net);
      if (!_design->indexPin(cts_pin)) {
        LOG_ERROR << "CTS clock read failed for clock \"" << clock_name << "\": failed to index CTS pin \""
                  << Design::getPinFullName(cts_pin) << "\".";
        return nullptr;
      }

      if (cts_inst != nullptr) {
        if (idb_pin == idb_net_pins.driver) {
          cts_inst->insertDriverPin(cts_pin);
        } else {
          cts_inst->add_pin(cts_pin);
        }
      }
      if (idb_pin == idb_net_pins.driver) {
        clock->set_clock_source(cts_pin);
        cts_net->set_driver(cts_pin);
      } else {
        cts_loads.push_back(cts_pin);
      }
    }
    clock->set_loads(cts_loads);
    cts_net->set_loads(cts_loads);
    return clock;
  }

  auto buildPreclusteredClockFromTraceTarget(const ClockTraceClockTarget& clock_target, idb::IdbNet* source_idb_net) -> Clock*
  {
    if (source_idb_net == nullptr || clock_target.preclustered_sink_anchors.empty()) {
      return nullptr;
    }

    auto* clock = _design->makeClock(clock_target.clock_name, clock_target.clock_net_name);
    if (clock == nullptr) {
      LOG_ERROR << "CTS clock read failed for clock \"" << clock_target.clock_name << "\": failed to create CTS clock object.";
      return nullptr;
    }
    clock->set_clock_name(clock_target.clock_name);
    clock->set_clock_net_name(clock_target.clock_net_name);
    clock->set_clock_source(nullptr);
    clock->set_clock_source_net(nullptr);
    clock->set_preclustered_sink_reuse(true);
    clock->clear_preclustered_anchor_input_net_names();
    clock->clear_loads();
    clock->clearMembership();

    auto* cts_net = buildNetFromIdbNet(source_idb_net);
    if (cts_net == nullptr) {
      return nullptr;
    }
    cts_net->set_driver(nullptr);
    cts_net->set_loads({});
    clock->set_clock_source_net(cts_net);

    const auto source_pins = collectIdbClockNetPins(source_idb_net);
    if (source_pins.driver == nullptr) {
      LOG_ERROR << "CTS clock read failed for clock \"" << clock_target.clock_name << "\": source iDB net \"" << clock_target.clock_net_name
                << "\" has no resolvable driver pin.";
      return nullptr;
    }

    Inst* source_cts_inst = nullptr;
    if (auto* source_idb_inst = source_pins.driver->get_instance(); source_idb_inst != nullptr) {
      source_cts_inst = buildInstFromIdbInst(source_idb_inst);
      if (source_cts_inst == nullptr) {
        return nullptr;
      }
    }
    auto* source_pin = buildOrFindPinFromIdbPin(source_pins.driver, source_cts_inst);
    if (source_pin == nullptr) {
      return nullptr;
    }
    source_pin->set_net(cts_net);
    if (source_cts_inst != nullptr) {
      source_cts_inst->insertDriverPin(source_pin);
    }
    clock->set_clock_source(source_pin);
    cts_net->set_driver(source_pin);

    std::vector<Pin*> cts_loads;
    cts_loads.reserve(clock_target.preclustered_sink_anchors.size());
    for (const auto& anchor : clock_target.preclustered_sink_anchors) {
      auto* idb_inst = findIdbInstOrError(clock_target.clock_name, anchor.driver_inst_name);
      if (idb_inst == nullptr) {
        return nullptr;
      }
      auto* cts_inst = buildInstFromIdbInst(idb_inst);
      if (cts_inst == nullptr) {
        return nullptr;
      }

      auto* output_idb_pin = findIdbInstPinOrError(clock_target.clock_name, idb_inst, anchor.output_pin_name);
      auto* input_idb_pin = findIdbInstPinOrError(clock_target.clock_name, idb_inst, anchor.input_pin_name);
      if (output_idb_pin == nullptr || input_idb_pin == nullptr) {
        return nullptr;
      }

      auto* output_pin = buildOrFindPinFromIdbPin(output_idb_pin, cts_inst);
      auto* input_pin = buildOrFindPinFromIdbPin(input_idb_pin, cts_inst);
      if (output_pin == nullptr || input_pin == nullptr) {
        return nullptr;
      }
      output_pin->set_net(nullptr);
      input_pin->set_net(cts_net);
      cts_inst->insertDriverPin(output_pin);
      cts_inst->add_pin(input_pin);
      cts_loads.push_back(input_pin);
      clock->add_preclustered_anchor_input_net_name(anchor.input_net_name);
    }

    clock->set_loads(cts_loads);
    cts_net->set_loads(cts_loads);
    LOG_INFO << "CTS clock read: clock \"" << clock_target.clock_name << "\" reuses " << clock_target.preclustered_sink_anchors.size()
             << " preclustered sink buffer anchor(s).";
    return clock;
  }

  auto buildInstFromIdbInst(idb::IdbInstance* idb_inst) -> Inst*
  {
    if (idb_inst == nullptr) {
      return nullptr;
    }
    auto* cell_master = idb_inst->get_cell_master();
    auto* coord = idb_inst->get_coordinate();
    if (cell_master == nullptr || coord == nullptr) {
      LOG_ERROR << "CTS clock read failed: iDB inst \"" << idb_inst->get_name() << "\" is missing required cell master or coordinate.";
      return nullptr;
    }

    const auto inst_name = idb_inst->get_name();
    auto* cts_inst = _design->makeInst(inst_name);
    if (cts_inst == nullptr) {
      LOG_ERROR << "CTS clock read failed: failed to create CTS inst \"" << inst_name << "\".";
      return nullptr;
    }
    cts_inst->set_name(inst_name);
    cts_inst->set_cell_master(cell_master->get_name());
    const auto classification = classifyCtsInstFromIdbInst(*_wrapper, idb_inst);
    cts_inst->set_type(classification.type);
    recordInstClassification(classification);
    cts_inst->set_location(Wrapper::idbToCts(*coord));
    bindIdbInst(idb_inst, cts_inst);
    return cts_inst;
  }

  auto buildOrFindPinFromIdbPin(idb::IdbPin* idb_pin, Inst* cts_inst) -> Pin*
  {
    if (idb_pin == nullptr) {
      return nullptr;
    }
    if (_wrapper->_idb2cts_pin_map.contains(idb_pin)) {
      return _wrapper->_idb2cts_pin_map.at(idb_pin);
    }
    const auto pin_full_name = ctsPinFullName(idb_pin, cts_inst);
    if (!pin_full_name.empty()) {
      auto* existing_pin = _design->findPin(pin_full_name);
      if (existing_pin != nullptr) {
        bindIdbPin(idb_pin, existing_pin);
        return existing_pin;
      }
    }
    auto* cts_pin = buildPinFromIdbPin(idb_pin, cts_inst);
    if (cts_pin == nullptr) {
      return nullptr;
    }
    if (!_design->indexPin(cts_pin)) {
      LOG_ERROR << "CTS clock read failed: failed to index CTS pin \"" << Design::getPinFullName(cts_pin) << "\".";
      return nullptr;
    }
    return cts_pin;
  }

  auto buildPinFromIdbPin(idb::IdbPin* idb_pin, Inst* cts_inst) -> Pin*
  {
    if (idb_pin == nullptr) {
      return nullptr;
    }
    auto* idb_term = idb_pin->get_term();
    auto* avg_coord = idb_pin->get_average_coordinate();
    if (idb_term == nullptr || avg_coord == nullptr) {
      LOG_ERROR << "CTS clock read failed: iDB pin \"" << idb_pin->get_pin_name() << "\" is missing required term or average coordinate.";
      return nullptr;
    }

    const auto pin_name = idb_term->get_name();
    const auto pin_full_name = cts_inst == nullptr ? pin_name : cts_inst->get_name() + "/" + pin_name;
    if (_design->findPin(pin_full_name) != nullptr) {
      LOG_ERROR << "CTS clock read failed: duplicate CTS pin \"" << pin_full_name << "\".";
      return nullptr;
    }

    auto* cts_pin = _design->makePin(pin_name);
    if (cts_pin == nullptr) {
      LOG_ERROR << "CTS clock read failed: failed to create CTS pin \"" << pin_full_name << "\".";
      return nullptr;
    }
    cts_pin->set_name(pin_name);
    cts_pin->set_type(convertIdbPinType(idb_term->get_type(), idb_term->get_direction()));
    cts_pin->set_location(Wrapper::idbToCts(*avg_coord));
    cts_pin->set_inst(cts_inst);
    cts_pin->set_io(idb_pin->is_io_pin());
    bindIdbPin(idb_pin, cts_pin);
    return cts_pin;
  }

  auto findIdbInstOrError(const std::string& clock_name, const std::string& inst_name) -> idb::IdbInstance*
  {
    auto* idb_design = findIdbDesign();
    auto* inst_list = idb_design == nullptr ? nullptr : idb_design->get_instance_list();
    auto* idb_inst = inst_list == nullptr ? nullptr : inst_list->find_instance(inst_name);
    if (idb_inst == nullptr) {
      LOG_ERROR << "CTS clock read failed for clock \"" << clock_name << "\": preclustered sink anchor inst \"" << inst_name
                << "\" is not found in iDB.";
      return nullptr;
    }
    return idb_inst;
  }

  static auto findIdbInstPinOrError(const std::string& clock_name, idb::IdbInstance* idb_inst, const std::string& pin_name) -> idb::IdbPin*
  {
    auto* idb_pin = idb_inst == nullptr ? nullptr : idb_inst->get_pin_by_term(pin_name);
    if (idb_pin == nullptr && idb_inst != nullptr && idb_inst->get_pin_list() != nullptr) {
      for (auto* candidate_pin : idb_inst->get_pin_list()->get_pin_list()) {
        if (candidate_pin != nullptr && candidate_pin->get_pin_name() == pin_name) {
          idb_pin = candidate_pin;
          break;
        }
      }
    }
    if (idb_pin == nullptr) {
      LOG_ERROR << "CTS clock read failed for clock \"" << clock_name << "\": preclustered sink anchor pin \"" << pin_name
                << "\" is not found in iDB inst \"" << (idb_inst == nullptr ? "" : idb_inst->get_name()) << "\".";
    }
    return idb_pin;
  }

  auto buildNetFromIdbNet(idb::IdbNet* idb_net) -> Net*
  {
    if (idb_net == nullptr) {
      return nullptr;
    }
    auto* cts_net = _design->makeNet(idb_net->get_net_name());
    if (cts_net == nullptr) {
      LOG_ERROR << "CTS clock read failed: failed to create CTS net \"" << idb_net->get_net_name() << "\".";
      return nullptr;
    }
    cts_net->set_name(idb_net->get_net_name());
    bindIdbNet(idb_net, cts_net);
    return cts_net;
  }

  auto bindIdbPin(idb::IdbPin* idb_pin, Pin* cts_pin) -> void { _wrapper->crossRef(idb_pin, cts_pin); }
  auto bindIdbInst(idb::IdbInstance* idb_inst, Inst* cts_inst) -> void { _wrapper->crossRef(idb_inst, cts_inst); }
  auto bindIdbNet(idb::IdbNet* idb_net, Net* cts_net) -> void { _wrapper->crossRef(idb_net, cts_net); }

  Wrapper* _wrapper = nullptr;
  Design* _design = nullptr;
  SchemaWriter* _reporter = nullptr;
  std::map<std::string, std::size_t> _classification_count_by_key;
};

auto Wrapper::readClocks(Design& design, SchemaWriter& reporter, const std::vector<std::pair<std::string, std::string>>& clock_net_pairs)
    -> bool
{
  return CtsClockReader(*this, design, reporter).readClocks(clock_net_pairs);
}

auto Wrapper::readTraceClockTargets(Design& design, SchemaWriter& reporter, const std::vector<ClockTraceClockTarget>& clock_targets) -> bool
{
  return CtsClockReader(*this, design, reporter).readTraceClockTargets(clock_targets);
}

}  // namespace icts
