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
#include "DRConfig.hpp"
#include "DRDatabase.hpp"
#include "DRNet.hpp"
#include "Database.hpp"

namespace irt {

class DRDataManager
{
 public:
  DRDataManager() = default;
  DRDataManager(const DRDataManager& other) = delete;
  DRDataManager(DRDataManager&& other) = delete;
  ~DRDataManager() = default;
  DRDataManager& operator=(const DRDataManager& other) = delete;
  DRDataManager& operator=(DRDataManager&& other) = delete;
  // function
  void input(Config& config, Database& database);
  DRConfig& getConfig() { return _dr_config; }
  DRDatabase& getDatabase() { return _dr_database; }
  std::vector<DRNet> convertToDRNetList(std::vector<Net>& net_list);
  DRNet convertToDRNet(Net& net);

 private:
  DRConfig _dr_config;
  DRDatabase _dr_database;
  // function
  void wrapConfig(Config& config);
  void wrapDatabase(Database& database);
  void wrapMicronDBU(Database& database);
  void wrapGCellAxis(Database& database);
  void wrapDie(Database& database);
  void wrapRoutingLayerList(Database& database);
  void wrapCutLayerList(Database& database);
  void wrapLayerViaMasterList(Database& database);
  void wrapRoutingBlockageList(Database& database);
  void buildConfig();
  void buildDatabase();
};

}  // namespace irt
