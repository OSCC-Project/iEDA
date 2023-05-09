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

#include "Config.hpp"
#include "Database.hpp"
#include "PAConfig.hpp"
#include "PADatabase.hpp"
#include "PANet.hpp"

namespace irt {

class PADataManager
{
 public:
  PADataManager() = default;
  PADataManager(const PADataManager& other) = delete;
  PADataManager(PADataManager&& other) = delete;
  ~PADataManager() = default;
  PADataManager& operator=(const PADataManager& other) = delete;
  PADataManager& operator=(PADataManager&& other) = delete;
  // function
  void input(Config& config, Database& database);
  PAConfig& getConfig() { return _pa_config; }
  PADatabase& getDatabase() { return _pa_database; }
  std::vector<PANet> convertToPANetList(std::vector<Net>& net_list);
  PANet convertToPANet(Net& net);

 private:
  PAConfig _pa_config;
  PADatabase _pa_database;
  // function
  void wrapConfig(Config& config);
  void wrapDatabase(Database& database);
  void wrapDie(Database& database);
  void wrapCellAxis(Database& database);
  void wrapRoutingLayerList(Database& database);
  void wrapLayerViaMasterList(Database& database);
  void wrapRoutingBlockageList(Database& database);
  void buildConfig();
  void buildDatabase();
};

}  // namespace irt
