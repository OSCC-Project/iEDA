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

namespace ipnp {

void iPNPIdbWrapper::saveToIdb(GridManager pnp_network)
{
  PowerRouter* power_router = new PowerRouter();
  _idb_design->get_special_net_list()->add_net(power_router->createNet(pnp_network, ipnp::PowerType::kVDD));
  _idb_design->get_special_net_list()->add_net(power_router->createNet(pnp_network, ipnp::PowerType::kVSS));

  std::cout << "[iPNP info]: Added iPNP net." << std::endl;
}

void iPNPIdbWrapper::writeIdbToDef(std::string def_file_path)
{
  if (!_idb_design) {
    std::cerr << "Error: IDB design is null in writeIdbToDef" << std::endl;
    return;
  }

  try {
    auto* db_builder = new idb::IdbBuilder();
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
    
    // Check if the directory exists and is writable
    std::string dir_path = def_file_path.substr(0, def_file_path.find_last_of("/\\"));
    if (!dir_path.empty()) {
      struct stat info;
      if (stat(dir_path.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR)) {
        std::cerr << "Error: Directory does not exist: " << dir_path << std::endl;
        delete db_builder;
        return;
      }
      
      // Check if directory is writable
      if (access(dir_path.c_str(), W_OK) != 0) {
        std::cerr << "Error: No write permission for directory: " << dir_path << std::endl;
        delete db_builder;
        return;
      }
    }
    
    bool success = db_builder->saveDef(def_file_path);
    if (!success) {
      std::cerr << "Error: Failed to save DEF file to: " << def_file_path << std::endl;
    } else {
      std::cout << "Successfully wrote DEF file to: " << def_file_path << std::endl;
    }
    
    delete db_builder;
  } catch (const std::exception& e) {
    std::cerr << "Exception in writeIdbToDef: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "Unknown exception in writeIdbToDef" << std::endl;
  }
}

}  // namespace ipnp
