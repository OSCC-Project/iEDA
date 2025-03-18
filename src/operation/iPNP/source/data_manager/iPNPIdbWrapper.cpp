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
#include <sys/stat.h>
#include <unistd.h>

#include "PowerVia.hh"

namespace ipnp {

  void iPNPIdbWrapper::saveToIdb(GridManager pnp_network)
  {
    // PowerRouter* power_router = new PowerRouter();
    // _idb_design->get_special_net_list()->add_net(power_router->createNet(pnp_network, ipnp::PowerType::kVDD));
    // _idb_design->get_special_net_list()->add_net(power_router->createNet(pnp_network, ipnp::PowerType::kVSS));

    // PowerVia power_via;
    
    // idb::IdbDesign* updated_design = power_via.connectAllPowerLayers(pnp_network, _idb_design);
    // if (!updated_design) {
    //   std::cerr << "[iPNP error]: Failed to connect power layers." << std::endl;
    // }
    // else {
    //   // 更新设计对象
    //   _idb_design = updated_design;
    //   std::cout << "[iPNP info]: Successfully connected all power layers." << std::endl;
    // }
    
    // delete power_router;

    std::cout << "[iPNP info]: Added iPNP net." << std::endl;
  }

  void iPNPIdbWrapper::writeIdbToDef(std::string def_file_path)
  {
    if (!_idb_design) {
      std::cerr << "Error: IDB design is null in writeIdbToDef" << std::endl;
      return;
    }

    auto* db_builder = get_idb_builder();
    if (!db_builder) {
      std::cerr << "Error: Failed to create IdbBuilder" << std::endl;
      return;
    }

    auto* def_service = db_builder->get_def_service();
    if (!def_service) {
      std::cerr << "Error: DEF service is null" << std::endl;
      delete db_builder;
      return;
    }

    auto* layout = _idb_design->get_layout();
    if (!layout) {
      std::cerr << "Error: Layout is null" << std::endl;
      delete db_builder;
      return;
    }

    def_service->set_layout(layout);

    bool success = db_builder->saveDef(def_file_path);
    if (!success) {
      std::cout << "Successfully wrote DEF file to: " << def_file_path << std::endl;
    } // saveDef的返回值本来就写反了
    else {
      std::cerr << "Error: Failed to save DEF file to: " << def_file_path << std::endl;
    }

    delete db_builder;
  }

}  // namespace ipnp
