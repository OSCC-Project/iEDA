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
 * @file BufferInsertion.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief Provides temporary CTS object creation and clock-net side-effect restoration helpers.
 */

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "Net.hh"
#include "Pin.hh"
#include "Point.hh"
#include "synthesis/topology/Topology.hh"

namespace icts {
class Design;
class Inst;
}  // namespace icts

namespace icts::topology {

struct BufferCreation
{
  Inst* inst = nullptr;
  Pin* input_pin = nullptr;
  Pin* output_pin = nullptr;
};

struct TopologyNetConnectionInput
{
  Net* net = nullptr;
  Pin* driver = nullptr;
  std::vector<Pin*> sinks;
};

auto MakeObjectName(const std::string& prefix, const std::string& suffix) -> std::string;
auto HasValidLocation(const Point<int>& location) -> bool;
auto FindRenderableLocation(const Pin* pin) -> Point<int>;
auto CollectValidLoads(const Net& net) -> std::vector<Pin*>;
auto ConnectNet(const TopologyNetConnectionInput& input) -> void;
auto ReconnectExistingNet(const TopologyNetConnectionInput& input) -> void;
auto CreateBufferInstance(Topology::Build& result, const std::string& inst_name, const std::string& cell_master,
                          const std::string& input_pin_name, const std::string& output_pin_name, const Point<int>& location)
    -> BufferCreation;
auto CreateNet(Topology::Build& result, const std::string& net_name, Pin* driver, const std::vector<Pin*>& sinks) -> Net*;

class RootNetSideEffectGuard
{
 public:
  RootNetSideEffectGuard(Design& design, Net& root_net, Pin* root_driver);
  RootNetSideEffectGuard(const RootNetSideEffectGuard&) = delete;
  auto operator=(const RootNetSideEffectGuard&) -> RootNetSideEffectGuard& = delete;

  auto restore() -> void;

 private:
  auto appendPinNet(Pin* pin) -> void;

  Design& _design;
  Net& _root_net;
  Pin* _root_driver = nullptr;
  std::vector<Pin*> _original_root_loads;
  Inst* _root_driver_inst = nullptr;
  std::string _root_driver_cell_master;
  std::vector<std::pair<Pin*, Net*>> _pin_nets;
  std::vector<std::pair<Pin*, std::string>> _root_pin_names;
};

class SourceNetSideEffectGuard
{
 public:
  SourceNetSideEffectGuard(Net& source_net, Pin* clock_source, const std::vector<Pin*>& root_inputs);
  SourceNetSideEffectGuard(const SourceNetSideEffectGuard&) = delete;
  auto operator=(const SourceNetSideEffectGuard&) -> SourceNetSideEffectGuard& = delete;

  auto restore() -> void;

 private:
  auto appendPinNet(Pin* pin) -> void;

  Net& _source_net;
  Pin* _original_driver = nullptr;
  std::vector<Pin*> _original_loads;
  std::vector<std::pair<Pin*, Net*>> _pin_nets;
};

}  // namespace icts::topology
