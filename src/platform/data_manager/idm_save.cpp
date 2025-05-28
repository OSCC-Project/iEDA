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
 * @File Name: dm_save.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "idm.h"

namespace idm {

bool DataManager::save(std::string name, std::string def_path)
{
  std::string full_path = def_path + "/" + name + ".def";
  std::cout << full_path << std::endl;

  if (_idb_builder == nullptr) {
    return false;
  }

  if (def_path.empty()) {
    def_path = _config.get_output_path();
  }

  if (def_path.empty()) {
    return false;
  }

  full_path = def_path + "/" + name + ".def";

  if (!saveDef(full_path)) {
    return false;
  }

  return true;
}

bool DataManager::saveDef(string def_path)
{
  if (_idb_builder == nullptr || _idb_lef_service == nullptr || _layout == nullptr) {
    return false;
  }
  return _idb_builder->saveDef(def_path);
}

bool DataManager::saveLef(string lef_path)
{
  if (_idb_builder == nullptr || _idb_lef_service == nullptr || _layout == nullptr) {
    return false;
  }
  return _idb_builder->saveLef(lef_path);
}

void DataManager::saveVerilog(string verilog_path, std::set<std::string>&& exclude_cell_names /*={}*/,
                              bool is_add_space_for_escape_name /*=false*/)
{
  if (_idb_builder == nullptr || _idb_lef_service == nullptr || _layout == nullptr) {
    std::cout << "idb_builder error.\n";
  }
  return _idb_builder->saveVerilog(verilog_path, exclude_cell_names, is_add_space_for_escape_name);
}

bool DataManager::saveGDSII(string path)
{
  if (_idb_builder == nullptr || _idb_lef_service == nullptr || _layout == nullptr) {
    return false;
  }
  return _idb_builder->saveGDSII(path);
}
bool DataManager::saveJSON(string path, string options)
{
  if (_idb_builder == nullptr || _idb_lef_service == nullptr || _layout == nullptr) {
    return false;
  }
  return _idb_builder->saveJSON(path, options);
}

}  // namespace idm
