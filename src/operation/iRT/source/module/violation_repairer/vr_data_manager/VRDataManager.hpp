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
#include "VRConfig.hpp"
#include "VRDatabase.hpp"
#include "VRNet.hpp"

namespace irt {

class VRDataManager
{
 public:
  VRDataManager() = default;
  VRDataManager(const VRDataManager& other) = delete;
  VRDataManager(VRDataManager&& other) = delete;
  ~VRDataManager() = default;
  VRDataManager& operator=(const VRDataManager& other) = delete;
  VRDataManager& operator=(VRDataManager&& other) = delete;
  // function
  void input(Config& config, Database& database);
  std::vector<VRNet> convertToVRNetList(std::vector<Net>& net_list);
  VRNet convertToVRNet(Net& net);
  VRConfig& getConfig() { return _vr_config; }
  VRDatabase& getDatabase() { return _vr_database; }

 private:
  VRConfig _vr_config;
  VRDatabase _vr_database;
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
