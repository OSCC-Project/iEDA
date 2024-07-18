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

#include "GridManager.hh"
#include "iPNPCommon.hh"
#include "idm.h"

namespace ipnp {

void iPNPIdbWrapper::readFromIdb(std::string input_def)
{
  auto idb_design = dmInst->get_idb_design();
  idb::IdbSpecialNetList* net_list = idb_design->get_special_net_list();
  idb::IdbSpecialNet* power_net;
  /**
   * @todo net_list --> get power_net
   */
  if (power_net == nullptr) {
    /**
     * @todo
     */
  }

  /**
   * @todo iDB --> GridManager
   * @todo _input_db_pdn = xxxxx
   */
}

void iPNPIdbWrapper::writeToIdb(const GridManager pnp_network)
{
  auto idb_design = dmInst->get_idb_design();
  idb::IdbSpecialNetList* net_list = idb_design->get_special_net_list();

  std::string net_name = "iPNP_PDN";
  std::cout << "[iPDN info]: add net = " << net_name << std::endl;
  idb::IdbSpecialNet* power_net = new idb::IdbSpecialNet();
  power_net->set_net_name(net_name);

  bool is_power;  // VDD or VSS
  idb::IdbConnectType power_type = is_power ? idb::IdbConnectType::kPower : idb::IdbConnectType::kGround;
  power_net->set_connect_type(power_type);

  /**
   * @todo pnp_network --> iDB
   * @code
   * power_net->set(pnp_network);
   * @endcode
   */
  idb_design->get_special_net_list()->add_net(power_net);

  auto idb_builder = dmInst->get_idb_builder();
  idb_builder->saveDef("def file path", idb::DefWriteType::kChip);  // kChip?
}

}  // namespace ipnp
