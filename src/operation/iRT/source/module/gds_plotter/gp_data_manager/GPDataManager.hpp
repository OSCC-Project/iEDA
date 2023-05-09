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
#include "GPConfig.hpp"
#include "GPDatabase.hpp"

namespace irt {

class GPDataManager
{
 public:
  GPDataManager() = default;
  GPDataManager(const GPDataManager& other) = delete;
  GPDataManager(GPDataManager&& other) = delete;
  ~GPDataManager() = default;
  GPDataManager& operator=(const GPDataManager& other) = delete;
  GPDataManager& operator=(GPDataManager&& other) = delete;
  // function
  void input(Config& config, Database& database);
  GPConfig& getConfig() { return _gp_config; }
  GPDatabase& getDatabase() { return _gp_database; }

 private:
  GPConfig _gp_config;
  GPDatabase _gp_database;
  // function
  void wrapConfig(Config& config);
  void wrapDatabase(Database& database);
  void wrapGCellAxis(Database& database);
  void wrapDie(Database& database);
  void wrapViaLib(Database& database);
  void wrapLayerList(Database& database);
  void wrapBlockageList(Database& database);
  void buildConfig();
  void buildDatabase();
  void buildGDSLayerMap();
  void buildLypFile();
};

}  // namespace irt
