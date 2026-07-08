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
 * @file SaltPinBuilder.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-10
 * @brief Shared ClockRoutingTerminal to salt::Pin conversion helper.
 */

#include "SaltPinBuilder.hh"

#include <cstddef>
#include <memory>
#include <vector>

#include "Point.hh"
#include "RoutingTerminal.hh"
#include "salt/base/net.h"

namespace icts {

auto BuildSaltPins(const ClockRoutingTerminal& driver_terminal, const std::vector<ClockRoutingTerminal>& load_terminals)
    -> std::vector<std::shared_ptr<salt::Pin>>
{
  std::vector<std::shared_ptr<salt::Pin>> salt_pins;
  salt_pins.reserve(load_terminals.size() + 1);
  salt_pins.push_back(
      std::make_shared<salt::Pin>(driver_terminal.location.get_x(), driver_terminal.location.get_y(), 0, driver_terminal.pin_cap));
  for (std::size_t i = 0; i < load_terminals.size(); ++i) {
    const auto& terminal = load_terminals.at(i);
    salt_pins.push_back(
        std::make_shared<salt::Pin>(terminal.location.get_x(), terminal.location.get_y(), static_cast<int>(i + 1), terminal.pin_cap));
  }
  return salt_pins;
}

}  // namespace icts
