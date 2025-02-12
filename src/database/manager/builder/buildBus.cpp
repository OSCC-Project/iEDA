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
#include "builder.h"

namespace idb {
void IdbBuilder::buildBus()
{
  IdbDesign* design = this->get_def_service()->get_design();
  IdbBusList* bus_list = design->get_bus_list();
  IdbNetList* net_list_ptr = design->get_net_list();
  IdbBusBitChars* bus_bit_chars = design->get_bus_bit_chars();

  if (bus_bit_chars == nullptr) {
    return;
  }

  // busNet busInstancePin busIo
  for (auto* net : net_list_ptr->get_net_list()) {
    // parse bus net
    auto net_bus_info = IdbBus::parseBusName(net->get_net_name(), *bus_bit_chars);
    if (net_bus_info) {
      bus_list->addOrUpdate(net_bus_info.value(), [net, &net_bus_info](IdbBus& bus) {
        bus.set_type(IdbBus::kBusNet);
        bus.addNet(net, net_bus_info->second);
      });
    }

    // parse io pin bus
    auto* io_pins = net->get_io_pins();
    for (auto* io_pin : io_pins->get_pin_list()) {
      auto io_bus_info = IdbBus::parseBusName(io_pin->get_pin_name(), *bus_bit_chars);
      if (io_bus_info) {
        bus_list->addOrUpdate(io_bus_info.value(), [io_pin, &io_bus_info](IdbBus& bus) {
          bus.set_type(IdbBus::kBusIo);
          bus.addPin(io_pin, io_bus_info->second);
        });
      }
    }
  }

  auto* instance_list = design->get_instance_list();
  if (instance_list) {
    for (auto* inst : instance_list->get_instance_list()) {
      for (auto* pin : inst->get_pin_list()->get_pin_list()) {
        auto pin_bus_info = IdbBus::parseBusName(pin->get_instance()->get_name() + "/" + pin->get_pin_name(), *bus_bit_chars);
        if (pin_bus_info) {
          bus_list->addOrUpdate(pin_bus_info.value(), [pin, &pin_bus_info](IdbBus& bus) {
            bus.set_type(IdbBus::kBusInstancePin);
            bus.addPin(pin, pin_bus_info->second);
          });
        }
      }
    }
  }
}
}  // namespace idb