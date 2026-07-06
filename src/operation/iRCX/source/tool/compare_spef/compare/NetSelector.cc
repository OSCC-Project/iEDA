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
#include "compare/NetSelector.hh"

#include <algorithm>

namespace ircx {
namespace compare_spef {
namespace {

auto containsPin(const Net& net, const std::string& pin_name) -> bool
{
  return std::any_of(net.pins.begin(), net.pins.end(), [&](const Pin& pin) { return pin.direction != "N" && pin.name == pin_name; });
}

}  // namespace

NetSelector::NetSelector(const Config& config) : _config(config), _configured_net_names(configuredNetNames(config))
{
}

auto NetSelector::selected(const Net& net) const -> bool
{
  if (_configured_net_names.empty() && !hasPathFilter()) {
    return true;
  }
  if (_configured_net_names.contains(net.name)) {
    return true;
  }
  return matchesPathFilter(net);
}

auto NetSelector::hasPathFilter() const -> bool
{
  return !_config.from_pin.empty() || !_config.to_pin.empty() || !_config.from_pins.empty() || !_config.to_pins.empty()
         || !_config.from_to_pins.empty();
}

auto NetSelector::configuredNetNames(const Config& config) -> std::unordered_set<std::string>
{
  std::unordered_set<std::string> names;
  if (!config.net_name.empty()) {
    names.insert(config.net_name);
  }
  for (const auto& net_name : config.net_names) {
    names.insert(net_name);
  }
  return names;
}

auto NetSelector::matchesPathFilter(const Net& net) const -> bool
{
  if (!_config.from_pin.empty() && containsPin(net, _config.from_pin)) {
    return true;
  }
  if (!_config.to_pin.empty() && containsPin(net, _config.to_pin)) {
    return true;
  }
  for (const auto& pin : _config.from_pins) {
    if (containsPin(net, pin)) {
      return true;
    }
  }
  for (const auto& pin : _config.to_pins) {
    if (containsPin(net, pin)) {
      return true;
    }
  }
  for (const auto& [from_pin, to_pin] : _config.from_to_pins) {
    if (containsPin(net, from_pin) && containsPin(net, to_pin)) {
      return true;
    }
  }
  return false;
}

}  // namespace compare_spef
}  // namespace ircx
