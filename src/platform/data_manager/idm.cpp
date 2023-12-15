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
 * @File Name: data_manager.cpp
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

DataManager* DataManager::_instance = nullptr;

bool DataManager::initConfig(string config_path)
{
  return _config.initConfig(config_path);
}

bool DataManager::init(string config_path)
{
  if (_idb_builder == nullptr) {
    _idb_builder = new IdbBuilder();
  }

  if (!initConfig(config_path)) {
    return false;
  }

  if (!initLef(_config.get_lef_paths())) {
    return false;
  }

  if (!initDef(_config.get_def_path())) {
    return false;
  }

  return true;
}

bool DataManager::readLef(string config_path)
{
  if (_idb_builder == nullptr) {
    _idb_builder = new IdbBuilder();
  }

  if (!initConfig(config_path)) {
    return false;
  }

  // tech lef
  if (!initLef(std::vector<std::string>{_config.get_tech_lef_path()}, true)) {
    return false;
  }
  // lef
  if (!initLef(_config.get_lef_paths())) {
    return false;
  }

  return true;
}

bool DataManager::readLef(vector<string> lef_paths, bool b_techlef)
{
  if (_idb_builder == nullptr) {
    _idb_builder = new IdbBuilder();
  }

  if (!initLef(lef_paths, b_techlef)) {
    return false;
  }

  return true;
}

bool DataManager::readDef(string path)
{
  if (_idb_builder == nullptr || _idb_lef_service == nullptr || _layout == nullptr) {
    return false;
  }

  if (!initDef(path)) {
    return false;
  }

  return true;
}

bool DataManager::readVerilog(string path, string top_module)
{
  if (_idb_builder == nullptr || _idb_lef_service == nullptr || _layout == nullptr) {
    return false;
  }

  if (!initVerilog(path, top_module)) {
    return false;
  }

  return true;
}

int DataManager::get_routing_layer_1st()
{
  string routing_layer_1st = _config.get_routing_layer_1st();
  auto layout = get_idb_layout();
  auto layer_list = layout->get_layers();

  auto layer_find = layer_list->find_layer(routing_layer_1st);

  return layer_find != nullptr ? layer_find->get_id() : 0;
}

}  // namespace idm
