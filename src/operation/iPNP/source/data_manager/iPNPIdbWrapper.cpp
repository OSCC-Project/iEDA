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
 * @file iPNPIdbWrapper.cpp
 * @author Jianrong Su
 * @brief
 * @version 1.0
 * @date 2025-06-23
 */

#include "iPNPIdbWrapper.hh"

#include <sys/stat.h>
#include <unistd.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "PowerRouter.hh"
#include "PowerVia.hh"

namespace ipnp {

void iPNPIdbWrapper::saveToIdb(GridManager pnp_network)
{
  auto idb_design = dmInst->get_idb_design();
  // clear VDD
  auto* vdd_net = idb_design->get_special_net_list()->find_net("VDD");
  auto* vdd_wire_list = vdd_net->get_wire_list();
  vdd_wire_list->reset();

  // clear VSS
  auto* vss_net = idb_design->get_special_net_list()->find_net("VSS");
  auto* vss_wire_list = vss_net->get_wire_list();
  vss_wire_list->reset();

  // add power/ground network to idb
  PowerRouter* power_router = new PowerRouter();
  power_router->addPowerNets(idb_design, pnp_network);

  // add via to idb
  PowerVia* power_via = new PowerVia();
  idb_design = power_via->connectAllPowerLayers(pnp_network, idb_design);

  delete power_router;
  delete power_via;

  LOG_INFO << "Added iPNP net.";
}

void iPNPIdbWrapper::connect_M2_M1_Layer()
{
  PowerVia* power_via = new PowerVia();

  auto idb_design = dmInst->get_idb_design();
  power_via->connectM2M1Layer(idb_design);

  delete power_via;
}

}  // namespace ipnp
