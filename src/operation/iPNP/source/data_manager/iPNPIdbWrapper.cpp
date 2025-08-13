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
  // clear VDD
  auto* vdd_net = _idb_design->get_special_net_list()->find_net("VDD");
  auto* vdd_wire_list = vdd_net->get_wire_list();
  vdd_wire_list->reset();

  // clear VSS
  auto* vss_net = _idb_design->get_special_net_list()->find_net("VSS");
  auto* vss_wire_list = vss_net->get_wire_list();
  vss_wire_list->reset();

  // add power/ground network to idb
  PowerRouter* power_router = new PowerRouter();
  power_router->addPowerNets(_idb_design, pnp_network);

  // add via to idb
  PowerVia* power_via = new PowerVia();
  power_via->connectAllPowerLayers(pnp_network, _idb_design);

  delete power_router;
  delete power_via;

  LOG_INFO << "Added iPNP net.";
}

void iPNPIdbWrapper::writeIdbToDef(std::string def_file_path)
{
  if (!_idb_design) {
    LOG_ERROR << "Error: IDB design is null in writeIdbToDef";
    return;
  }

  auto* db_builder = get_idb_builder();
  if (!db_builder) {
    LOG_ERROR << "Error: Failed to create IdbBuilder";
    return;
  }

  auto* def_service = db_builder->get_def_service();
  if (!def_service) {
    LOG_ERROR << "Error: DEF service is null";
    delete db_builder;
    return;
  }

  auto* layout = _idb_design->get_layout();
  if (!layout) {
    LOG_ERROR << "Error: Layout is null";
    delete db_builder;
    return;
  }

  def_service->set_layout(layout);

  bool success = db_builder->saveDef(def_file_path);
  if (!success) {
    LOG_INFO << "Successfully wrote DEF file to: " << def_file_path;
  } else {
    LOG_ERROR << "Error: Failed to save DEF file to: " << def_file_path;
  }
}

void iPNPIdbWrapper::connect_M2_M1_Layer()
{
  PowerVia* power_via = new PowerVia();
  power_via->connectM2M1Layer(_idb_design);

  delete power_via;
}

}  // namespace ipnp
