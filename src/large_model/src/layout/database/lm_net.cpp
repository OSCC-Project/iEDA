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
#pragma once
/**
 * @project		large model
 * @date		06/11/2024
 * @version		0.1
 * @description
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "lm_net.h"

#include "Log.hh"

namespace ilm {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
LmNet* LmGraph::get_net(int net_id)
{
  auto it = _net_map.find(net_id);
  if (it != _net_map.end()) {
    return &it->second;
  }

  return nullptr;
}

LmNet* LmGraph::addNet(int net_id)
{
  if (get_net(net_id) == nullptr) {
    LmNet lm_net(net_id);
    auto [it, success] = _net_map.insert(std::make_pair(net_id, lm_net));
  }

  return get_net(net_id);
}

void LmGraph::add_net_wire(int net_id, LmNetWire wire)
{
  auto& [start, end] = wire.get_connected_nodes();
  if (start == nullptr || end == nullptr) {
    LOG_INFO << "wire error";
  }

  auto it = _net_map.find(net_id);
  if (it != _net_map.end()) {
    it->second.addWire(wire);
  } else {
    LmNet lm_net(net_id);
    lm_net.addWire(wire);
    auto result = _net_map.insert(std::make_pair(net_id, lm_net));
  }
}

}  // namespace ilm
