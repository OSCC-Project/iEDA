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
 * @file PowerRouter.hh
 * @author Jianrong Su
 * @brief
 * @version 0.1
 * @date 2025-03-28
 */

#pragma once

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "PNPGridManager.hh"

namespace idb {
class IdbSpecialNet;
}

namespace ipnp {

class PowerRouter
{
 public:
  PowerRouter() = default;
  ~PowerRouter() = default;

  void addPowerNets(PNPGridManager pnp_network);

 private:
  void addPowerStripesToCore(idb::IdbSpecialNet* power_net, PNPGridManager pnp_network);
  void addPowerStripesToDie(idb::IdbSpecialNet* power_net, PNPGridManager pnp_network);
  void addPowerFollowPin(idb::IdbSpecialNet* power_net);
  void addPowerPort(PNPGridManager pnp_network, std::string pin_name, std::string layer_name);

  void addVSSNet(PNPGridManager pnp_network);
  void addVDDNet(PNPGridManager pnp_network);
};

}  // namespace ipnp