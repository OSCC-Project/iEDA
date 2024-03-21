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
#include "EGRConfig.hpp"
#include "EGRDatabase.hpp"
#include "EGRHelper.hpp"
#include "EGRStat.hpp"
#include "builder.h"
#include "def_service.h"
#include "lef_service.h"

namespace irt {

class EGRDataManager
{
 public:
  EGRDataManager() = default;
  EGRDataManager(const EGRDataManager& other) = delete;
  EGRDataManager(EGRDataManager&& other) = delete;
  ~EGRDataManager() = default;
  EGRDataManager& operator=(const EGRDataManager& other) = delete;
  EGRDataManager& operator=(EGRDataManager&& other) = delete;
  // function
  void input(std::map<std::string, std::any>& config_map, idb::IdbBuilder* idb_builder);
  EGRConfig& getConfig() { return _egr_config; }
  EGRDatabase& getDatabase() { return _egr_database; }
  EGRHelper& getEGRHelper() { return _egr_helper; }
  EGRStat& getEGRStat() { return _egr_stat; }

 private:
  EGRConfig _egr_config;
  EGRDatabase _egr_database;
  EGRHelper _egr_helper;
  EGRStat _egr_stat;
  // Helper

  // function
  void wrapConfig(std::map<std::string, std::any>& config_map);
  void wrapDatabase(idb::IdbBuilder* idb_builder);
  void wrapDesignName(idb::IdbBuilder* idb_builder);
  void wrapMicronDBU(idb::IdbBuilder* idb_builder);
  void wrapDie(idb::IdbBuilder* idb_builder);
  void wrapLayerList(idb::IdbBuilder* idb_builder);
  void wrapTrackAxis(RoutingLayer& routing_layer, idb::IdbLayerRouting* idb_layer);
  void wrapLayerViaMasterList(idb::IdbBuilder* idb_builder);
  void wrapObstacleList(idb::IdbBuilder* idb_builder);
  void wrapArtificialObstacle(idb::IdbBuilder* idb_builder);
  void wrapInstanceObstacle(idb::IdbBuilder* idb_builder);
  void wrapSpecialNetObstacle(idb::IdbBuilder* idb_builder);
  void wrapNetList(idb::IdbBuilder* idb_builder);
  bool checkSkipping(idb::IdbNet* idb_net);
  void wrapPinList(EGRNet& egr_net, idb::IdbNet* idb_net);
  void wrapPinShapeList(EGRPin& egr_pin, idb::IdbPin* idb_pin);
  void processEmptyShapePin(EGRNet& net);
  void wrapDrivingPin(EGRNet& egr_net, idb::IdbNet* idb_net);
  void updateHelper(idb::IdbBuilder* idb_builder);
  void buildConfig();
  void buildSkipNetNameSet();
  void buildCellSize();
  void buildBottomTopLayerIdx();
  void buildEGRStrategy();
  void buildDatabase();
  void buildLayerList();
  void buildLayerViaMasterList();
  void transLayerViaMasterList();
  void makeLayerViaMasterList();
  void buildDie();
  void buildObstacleList();
  void buildNetList();
  void buildPinList(EGRNet& egr_net);
  void buildDrivingPin(EGRNet& egr_net);
  void buildLayerResourceMap();
  void initLayerResourceMapSize();
  void addResourceMapSupply();
  void addResourceMapDemand();
  void legalizeResourceMapDemand();
  void buildHVLayerIdxList();
  Direction getRTDirectionByDB(idb::IdbLayerDirection idb_direction);
  int32_t getEGRRoutingLayerIndexByDB(int32_t db_layer_idx);
  int32_t getEGRCutLayerIndexByDB(int32_t db_layer_idx);
  PlanarRect getGridRect(PlanarRect& real_rect);
  void printConfig();
  void printDatabase();
};

}  // namespace irt
