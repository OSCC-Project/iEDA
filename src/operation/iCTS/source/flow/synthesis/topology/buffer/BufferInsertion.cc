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
 * @file BufferInsertion.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief Implements temporary CTS object creation and clock-net side-effect restoration helpers.
 */

#include "synthesis/topology/buffer/BufferInsertion.hh"

#include <glog/logging.h>

#include <algorithm>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "Log.hh"
#include "design/Design.hh"
#include "design/Inst.hh"
#include "design/Net.hh"
#include "design/Pin.hh"

namespace icts::topology {
namespace {

auto isDesignPin(const Design& design, const Pin* pin) -> bool
{
  if (pin == nullptr) {
    return false;
  }

  const auto pins = design.get_pins();
  return std::ranges::find_if(pins, [pin](const auto* candidate) -> bool { return candidate == pin; }) != pins.end();
}

}  // namespace

auto MakeObjectName(const std::string& prefix, const std::string& suffix) -> std::string
{
  if (prefix.empty()) {
    return "cts_" + suffix;
  }
  return prefix + "_" + suffix;
}

auto HasValidLocation(const Point<int>& location) -> bool
{
  return location.get_x() >= 0 && location.get_y() >= 0;
}

auto FindRenderableLocation(const Pin* pin) -> Point<int>
{
  if (pin == nullptr) {
    return Point<int>(-1, -1);
  }

  const auto pin_location = pin->get_location();
  if (HasValidLocation(pin_location)) {
    return pin_location;
  }
  if (pin->get_inst() != nullptr && HasValidLocation(pin->get_inst()->get_location())) {
    return pin->get_inst()->get_location();
  }
  return Point<int>(-1, -1);
}

auto CollectValidLoads(const Net& net) -> std::vector<Pin*>
{
  std::vector<Pin*> loads;
  loads.reserve(net.get_loads().size());
  for (auto* load : net.get_loads()) {
    if (load != nullptr) {
      loads.push_back(load);
    }
  }
  return loads;
}

auto ConnectNet(const TopologyNetConnectionInput& input) -> void
{
  LOG_FATAL_IF(input.net == nullptr) << "Topology buffer insertion: net connection target is null.";
  auto& net = *input.net;
  net.set_driver(input.driver);
  if (input.driver != nullptr) {
    input.driver->set_net(&net);
  }

  net.set_loads({});
  for (auto* sink : input.sinks) {
    if (sink == nullptr) {
      continue;
    }
    net.add_load(sink);
    sink->set_net(&net);
  }
}

auto ReconnectExistingNet(const TopologyNetConnectionInput& input) -> void
{
  LOG_FATAL_IF(input.net == nullptr) << "Topology buffer insertion: net reconnection target is null.";
  auto& net = *input.net;
  auto* old_driver = net.get_driver();
  if (old_driver != nullptr && old_driver != input.driver && old_driver->get_net() == &net) {
    old_driver->set_net(nullptr);
  }
  for (auto* old_sink : net.get_loads()) {
    if (old_sink != nullptr && old_sink->get_net() == &net) {
      old_sink->set_net(nullptr);
    }
  }
  ConnectNet(input);
}

auto CreateBufferInstance(Topology::Build& result, const std::string& inst_name, const std::string& cell_master,
                          const std::string& input_pin_name, const std::string& output_pin_name, const Point<int>& location)
    -> BufferCreation
{
  auto inst = std::make_unique<Inst>(inst_name, cell_master, InstType::kBuffer, location);
  auto* inst_ptr = inst.get();

  auto input_pin = std::make_unique<Pin>(input_pin_name, PinType::kIn, location, inst_ptr, nullptr, false);
  auto* input_pin_ptr = input_pin.get();
  result.output.inserted_pins.push_back(std::move(input_pin));

  auto output_pin = std::make_unique<Pin>(output_pin_name, PinType::kOut, location, inst_ptr, nullptr, false);
  auto* output_pin_ptr = output_pin.get();
  result.output.inserted_pins.push_back(std::move(output_pin));

  inst_ptr->add_pin(input_pin_ptr);
  inst_ptr->insertDriverPin(output_pin_ptr);

  result.output.inserted_insts.push_back(std::move(inst));
  return BufferCreation{
      .inst = inst_ptr,
      .input_pin = input_pin_ptr,
      .output_pin = output_pin_ptr,
  };
}

auto CreateNet(Topology::Build& result, const std::string& net_name, Pin* driver, const std::vector<Pin*>& sinks) -> Net*
{
  auto net = std::make_unique<Net>(net_name);
  auto* net_ptr = net.get();
  ConnectNet(TopologyNetConnectionInput{
      .net = net_ptr,
      .driver = driver,
      .sinks = sinks,
  });
  result.output.inserted_nets.push_back(std::move(net));
  return net_ptr;
}

RootNetSideEffectGuard::RootNetSideEffectGuard(Design& design, Net& root_net, Pin* root_driver)
    : _design(design), _root_net(root_net), _root_driver(root_driver), _original_root_loads(root_net.get_loads())
{
  appendPinNet(_root_driver);
  for (auto* sink : _original_root_loads) {
    appendPinNet(sink);
  }

  if (_root_driver != nullptr) {
    _root_driver_inst = _root_driver->get_inst();
  }
  if (_root_driver_inst != nullptr) {
    _root_driver_cell_master = _root_driver_inst->get_cell_master();
    for (auto* pin : _root_driver_inst->get_pins()) {
      if (pin == nullptr) {
        continue;
      }
      _root_pin_names.emplace_back(pin, pin->get_name());
      appendPinNet(pin);
    }
  } else if (_root_driver != nullptr) {
    _root_pin_names.emplace_back(_root_driver, _root_driver->get_name());
  }
}

auto RootNetSideEffectGuard::restore() -> void
{
  if (_root_driver_inst != nullptr) {
    _root_driver_inst->set_cell_master(_root_driver_cell_master);
  }
  for (const auto& [pin, name] : _root_pin_names) {
    if (pin == nullptr || pin->get_name() == name) {
      continue;
    }
    if (isDesignPin(_design, pin)) {
      (void) _design.renamePin(pin, name);
    } else {
      pin->set_name(name);
    }
  }

  ConnectNet(TopologyNetConnectionInput{
      .net = &_root_net,
      .driver = _root_driver,
      .sinks = _original_root_loads,
  });
  for (const auto& [pin, net] : _pin_nets) {
    if (pin != nullptr) {
      pin->set_net(net);
    }
  }
}

auto RootNetSideEffectGuard::appendPinNet(Pin* pin) -> void
{
  if (pin == nullptr) {
    return;
  }
  const auto iter = std::ranges::find_if(_pin_nets, [pin](const auto& pin_net) -> bool { return pin_net.first == pin; });
  if (iter == _pin_nets.end()) {
    _pin_nets.emplace_back(pin, pin->get_net());
  }
}

SourceNetSideEffectGuard::SourceNetSideEffectGuard(Net& source_net, Pin* clock_source, const std::vector<Pin*>& root_inputs)
    : _source_net(source_net), _original_driver(source_net.get_driver()), _original_loads(source_net.get_loads())
{
  appendPinNet(clock_source);
  for (auto* load : _original_loads) {
    appendPinNet(load);
  }
  for (auto* root_input : root_inputs) {
    appendPinNet(root_input);
  }
}

auto SourceNetSideEffectGuard::restore() -> void
{
  ReconnectExistingNet(TopologyNetConnectionInput{
      .net = &_source_net,
      .driver = _original_driver,
      .sinks = _original_loads,
  });
  for (const auto& [pin, net] : _pin_nets) {
    if (pin != nullptr) {
      pin->set_net(net);
    }
  }
}

auto SourceNetSideEffectGuard::appendPinNet(Pin* pin) -> void
{
  if (pin == nullptr) {
    return;
  }
  const auto iter = std::ranges::find_if(_pin_nets, [pin](const auto& pin_net) -> bool { return pin_net.first == pin; });
  if (iter == _pin_nets.end()) {
    _pin_nets.emplace_back(pin, pin->get_net());
  }
}

}  // namespace icts::topology
