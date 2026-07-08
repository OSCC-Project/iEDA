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
 * @file ClockSizingAcceptedEdit.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief Committed design clock sizing edit helpers for CTS post-synthesis optimization.
 */

#include "optimization/clock_sizing_edit/ClockSizingAcceptedEdit.hh"

#include <glog/logging.h>

#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "Log.hh"
#include "design/ClockLayout.hh"
#include "design/Design.hh"
#include "design/Inst.hh"
#include "design/Pin.hh"
#include "io/Wrapper.hh"
#include "optimization/model/ClockSizingOptimizationData.hh"

namespace icts::clock_sizing_optimization {

namespace {

auto FindSingleBufferInputPin(Inst* inst) -> Pin*
{
  if (inst == nullptr) {
    return nullptr;
  }
  const auto* driver_pin = inst->findDriverPin();
  for (auto* pin : inst->get_pins()) {
    if (pin == nullptr || pin == driver_pin) {
      continue;
    }
    if (pin->get_type() == PinType::kIn || pin->get_type() == PinType::kClock) {
      return pin;
    }
  }
  for (auto* pin : inst->get_pins()) {
    if (pin != nullptr && pin != driver_pin) {
      return pin;
    }
  }
  return nullptr;
}

auto CanRenamePin(Design& design, Pin* pin, const std::string& local_name) -> bool
{
  if (pin == nullptr || local_name.empty()) {
    return false;
  }
  if (pin->get_name() == local_name) {
    return true;
  }
  const auto* inst = pin->get_inst();
  const std::string full_name = inst == nullptr ? local_name : inst->get_name() + "/" + local_name;
  auto* existing_pin = design.findPin(full_name);
  return existing_pin == nullptr || existing_pin == pin;
}

auto RenamePin(Design& design, Pin* pin, const std::string& local_name) -> bool
{
  if (pin == nullptr || local_name.empty()) {
    return false;
  }
  if (pin->get_name() == local_name) {
    return true;
  }
  return design.renamePin(pin, local_name);
}

auto ResolveBufferPorts(Wrapper& wrapper, const std::string& cell_master) -> std::optional<std::pair<std::string, std::string>>
{
  auto [input_pin_name, output_pin_name] = wrapper.queryBufferPorts(cell_master);
  if (input_pin_name.empty() || output_pin_name.empty() || input_pin_name == output_pin_name) {
    return std::nullopt;
  }
  return std::make_pair(std::move(input_pin_name), std::move(output_pin_name));
}

auto UpdateClockLayoutInstMaster(ClockLayout& clock_layout, const std::string& inst_name, const std::string& cell_master) -> void
{
  for (auto& layout_clock : clock_layout.get_clocks()) {
    for (auto& layout_inst : layout_clock.insts) {
      if (layout_inst.inst_name == inst_name) {
        layout_inst.cell_master = cell_master;
      }
    }
  }
}

}  // namespace

auto ApplyClockSizingAcceptedEdits(Design& design, Wrapper& wrapper, const std::vector<ClockSizingAcceptedEdit>& accepted_edits,
                                   const std::vector<ClockSizingBuffer>& buffers, ClockLayout& clock_layout) -> bool
{
  std::map<std::string, std::string> final_master_by_inst;
  std::map<std::string, std::string> expected_master_by_inst;
  for (const auto& buffer : buffers) {
    if (buffer.inst != nullptr) {
      expected_master_by_inst[buffer.inst_name] = buffer.inst->get_cell_master();
    }
  }
  for (const auto& accepted_edit : accepted_edits) {
    auto expected_iter = expected_master_by_inst.find(accepted_edit.inst_name);
    if (expected_iter == expected_master_by_inst.end()) {
      LOG_ERROR << "Optimization: cannot apply clock sizing edit for unresolved inst \"" << accepted_edit.inst_name << "\".";
      return false;
    }
    if (expected_iter->second != accepted_edit.from_master) {
      LOG_ERROR << "Optimization: cannot apply clock sizing edit for inst \"" << accepted_edit.inst_name
                << "\" because current master is \"" << expected_iter->second << "\" but solver expected \"" << accepted_edit.from_master
                << "\".";
      return false;
    }
    expected_iter->second = accepted_edit.to_master;
    final_master_by_inst[accepted_edit.inst_name] = accepted_edit.to_master;
  }

  for (const auto& [inst_name, final_master] : final_master_by_inst) {
    auto* inst = design.findInst(inst_name);
    auto* input_pin = FindSingleBufferInputPin(inst);
    auto* output_pin = inst == nullptr ? nullptr : inst->findDriverPin();
    const auto ports = ResolveBufferPorts(wrapper, final_master);
    if (inst == nullptr || input_pin == nullptr || output_pin == nullptr || !ports.has_value()
        || !CanRenamePin(design, input_pin, ports->first) || !CanRenamePin(design, output_pin, ports->second)) {
      LOG_ERROR << "Optimization: cannot apply final master \"" << final_master << "\" to buffer inst \"" << inst_name
                << "\" because its pin pair cannot be updated.";
      return false;
    }
  }

  for (const auto& [inst_name, final_master] : final_master_by_inst) {
    auto* inst = design.findInst(inst_name);
    auto* input_pin = FindSingleBufferInputPin(inst);
    auto* output_pin = inst->findDriverPin();
    const auto ports = ResolveBufferPorts(wrapper, final_master);
    if (!ports.has_value()) {
      return false;
    }
    const std::string old_input_name = input_pin->get_name();
    if (!RenamePin(design, input_pin, ports->first)) {
      return false;
    }
    if (!RenamePin(design, output_pin, ports->second)) {
      LOG_FATAL_IF(!RenamePin(design, input_pin, old_input_name)) << "Optimization: failed to restore buffer input-pin name.";
      return false;
    }
    inst->set_cell_master(final_master);
    inst->set_type(InstType::kBuffer);
    input_pin->set_type(PinType::kIn);
    output_pin->set_type(PinType::kOut);
    inst->insertDriverPin(output_pin);
    UpdateClockLayoutInstMaster(clock_layout, inst->get_name(), final_master);
  }
  return true;
}

}  // namespace icts::clock_sizing_optimization
