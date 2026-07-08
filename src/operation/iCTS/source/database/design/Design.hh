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
 * @file Design.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-11
 * @brief Aggregated design database for iCTS
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ClockDAG.hh"
#include "Inst.hh"
#include "Net.hh"
#include "Pin.hh"

namespace icts {

class Clock;
class SchemaWriter;

class Design
{
 public:
  Design();
  ~Design();

  // Delete copy and move constructors
  Design(const Design& rhs) = delete;
  Design(Design&& rhs) = delete;
  auto operator=(const Design& rhs) -> Design& = delete;
  auto operator=(Design&& rhs) -> Design& = delete;

  // Reset design data
  auto reset() -> void;

  // Getter
  auto get_clocks() const -> std::vector<Clock*>;
  auto get_insts() const -> std::vector<Inst*>;
  auto get_pins() const -> std::vector<Pin*>;
  auto get_nets() const -> std::vector<Net*>;

  // Adder
  auto makeClock(const std::string& clock_name, const std::string& clock_net_name) -> Clock*;
  auto findClock(const std::string& clock_name, const std::string& clock_net_name) const -> Clock*;
  auto makeInst(const std::string& name) -> Inst*;
  auto commitInst(std::unique_ptr<Inst> inst) -> Inst*;
  auto findInst(const std::string& name) const -> Inst*;
  auto makePin(const std::string& name) -> Pin*;
  auto commitPin(std::unique_ptr<Pin> pin) -> Pin*;
  auto indexPin(Pin* pin) -> bool;
  auto findPin(const std::string& pin_full_name) const -> Pin*;
  auto renamePin(Pin* pin, const std::string& name) -> bool;
  auto makeNet(const std::string& name) -> Net*;
  auto commitNet(std::unique_ptr<Net> net) -> Net*;
  auto findNet(const std::string& name) const -> Net*;
  auto clearClocks() -> void;
  auto clearTopologyObjects() -> void;
  auto removeClockMembershipObjects(Clock& clock) -> void;
  auto get_clock_dag() const -> const ClockDAG& { return _clock_dag; }
  auto rebuildClockDAG() -> bool;
  auto clearClockDAG() -> void;
  auto emitClockDistributionSummary(SchemaWriter& reporter, const std::string& title = "Clock Distribution Overview") const -> void;
  static auto getPinFullName(const Pin* pin) -> std::string;

 private:
  auto removeInst(Inst* inst) -> void;
  auto removePin(Pin* pin) -> void;
  auto removeNet(Net* net) -> void;

  std::vector<std::unique_ptr<Clock>> _clocks;
  std::vector<std::unique_ptr<Inst>> _insts;
  std::vector<std::unique_ptr<Pin>> _pins;
  std::vector<std::unique_ptr<Net>> _nets;
  std::unordered_map<std::string, Inst*> _inst_by_name;
  std::unordered_map<const Inst*, std::string> _inst_name_by_inst;
  std::unordered_map<std::string, Pin*> _pin_by_full_name;
  std::unordered_map<const Pin*, std::string> _pin_full_name_by_pin;
  std::unordered_map<std::string, Net*> _net_by_name;
  std::unordered_map<const Net*, std::string> _net_name_by_net;
  ClockDAG _clock_dag;
};

}  // namespace icts
