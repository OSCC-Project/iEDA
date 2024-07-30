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
 * @author Xinhao li
 * @brief
 * @version 0.1
 * @date 2024-07-15
 */

#include "iPNPIdbWrapper.hh"

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace ipnp {

void iPNPIdbWrapper::saveToIdb(GridManager pnp_network)
{
  PowerRouter* power_router = new PowerRouter();
  _idb_design->get_special_net_list()->add_net(power_router->createNet(pnp_network, ipnp::PowerType::VDD));
  _idb_design->get_special_net_list()->add_net(power_router->createNet(pnp_network, ipnp::PowerType::VSS));

  std::cout << "[iPNP info]: Added iPNP net." << std::endl;
}

void iPNPIdbWrapper::writeIdbToDef(std::string def_file_path)
{
  auto* db_builder = new idb::IdbBuilder();
  db_builder->get_def_service()->set_layout(_idb_design->get_layout());
  db_builder->saveDef(def_file_path);
}

}  // namespace ipnp
