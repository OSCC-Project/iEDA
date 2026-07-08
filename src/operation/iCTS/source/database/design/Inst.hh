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
 * @file Inst.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-07
 * @brief Minimal element representing a cell pin/port in CTS
 */

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "spatial/Point.hh"
namespace icts {
class Pin;

enum class InstType
{
  kBuffer,        // clock buffer
  kFlipFlop,      // edge-triggered sequential sink
  kLatch,         // level-sensitive sequential sink
  kInverter,      // clock inverter
  kClockGate,     // clock gate (with enable pin)
  kMux,           // clock mux / multi-clock-input boundary
  kClockLogic,    // clock-derived combinational logic boundary
  kBoundaryLoad,  // non-propagating clock-net load
  kMacroBlock,    // hard macro / block-class cell
  kUnknown
};

class Inst
{
 public:
  Inst(const std::string& _name, const std::string& cell_master, InstType type, const Point<int>& location)
      : _name(_name), _cell_master(cell_master), _type(type), _location(location)
  {
  }
  Inst(const Inst& rhs) = delete;
  Inst(Inst&& rhs) = delete;
  ~Inst() = default;

  auto operator=(const Inst& rhs) -> Inst& = delete;
  auto operator=(Inst&& rhs) -> Inst& = delete;

  // Getter
  auto get_name() const -> const std::string& { return _name; }
  auto get_cell_master() const -> const std::string& { return _cell_master; }
  auto get_type() const -> InstType { return _type; }
  auto get_location() const -> const Point<int>& { return _location; }

  // Setter
  auto set_name(const std::string& name) -> void { _name = name; }
  auto set_cell_master(const std::string& cell_master) -> void { _cell_master = cell_master; }
  auto set_type(InstType type) -> void { _type = type; }
  auto set_location(const Point<int>& location) -> void { _location = location; }

  auto get_pins() const -> const std::vector<Pin*>& { return _pins; }
  auto get_pins() -> std::vector<Pin*>& { return _pins; }
  auto findDriverPin() const -> Pin* { return _pins.empty() ? nullptr : _pins.front(); }
  auto set_pins(const std::vector<Pin*>& pins) -> void { _pins = pins; }
  auto add_pin(Pin* pin) -> void
  {
    if (pin == nullptr || std::ranges::find(_pins, pin) != _pins.end()) {
      return;
    }
    _pins.push_back(pin);
  }
  auto insertDriverPin(Pin* pin) -> void
  {
    if (pin == nullptr) {
      return;
    }
    if (!_pins.empty() && _pins.front() == pin) {
      return;
    }
    std::erase(_pins, pin);
    _pins.insert(_pins.begin(), pin);
  }

  // Boolean functions
  auto is_buffer() const -> bool { return _type == InstType::kBuffer; }
  auto is_flipflop() const -> bool { return _type == InstType::kFlipFlop; }
  auto is_latch() const -> bool { return _type == InstType::kLatch; }
  auto is_inverter() const -> bool { return _type == InstType::kInverter; }
  auto is_clock_gate() const -> bool { return _type == InstType::kClockGate; }
  auto is_mux() const -> bool { return _type == InstType::kMux; }
  auto is_clock_logic() const -> bool { return _type == InstType::kClockLogic; }
  auto is_boundary_load() const -> bool { return _type == InstType::kBoundaryLoad; }
  auto is_macro_block() const -> bool { return _type == InstType::kMacroBlock; }
  auto is_clock_propagation_cell() const -> bool { return is_buffer() || is_inverter(); }
  auto is_sequential_sink() const -> bool { return is_flipflop() || is_latch(); }
  auto is_clock_boundary() const -> bool { return is_clock_gate() || is_mux() || is_clock_logic() || is_boundary_load(); }

 private:
  std::string _name = "";
  std::string _cell_master = "";
  InstType _type = InstType::kUnknown;
  Point<int> _location = Point<int>(-1, -1);
  std::vector<Pin*> _pins;
};

}  // namespace icts
