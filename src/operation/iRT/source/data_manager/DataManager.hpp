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
#include "Helper.hpp"
#include "Logger.hpp"
#include "SortStatus.hpp"
#include "SortType.hpp"
#include "builder.h"
#include "def_service.h"
#include "lef_service.h"

namespace irt {

class DataManager
{
 public:
  DataManager() = default;
  DataManager(const DataManager& other) = delete;
  DataManager(DataManager&& other) = delete;
  ~DataManager() = default;
  DataManager& operator=(const DataManager& other) = delete;
  DataManager& operator=(DataManager&& other) = delete;
  // function
  void input(std::map<std::string, std::any>& config_map, idb::IdbBuilder* idb_builder);
  void save(Stage stage);
  void load(Stage stage);
  void output(idb::IdbBuilder* idb_builder);
  Config& getConfig() { return _config; }
  Database& getDatabase() { return _database; }
  Helper& getHelper() { return _helper; }

 private:
  // self
  Config _config;
  Database _database;
  Helper _helper;
  // function
  void wrapConfig(std::map<std::string, std::any>& config_map);
  void wrapDatabase(idb::IdbBuilder* idb_builder);
  void wrapMicronDBU(idb::IdbBuilder* idb_builder);
  void wrapDie(idb::IdbBuilder* idb_builder);
  void wrapLayerList(idb::IdbBuilder* idb_builder);
  void wrapTrackAxis(RoutingLayer& routing_layer, idb::IdbLayerRouting* idb_layer);
  void wrapSpacingTable(RoutingLayer& routing_layer, idb::IdbLayerRouting* idb_layer);
  void wrapLayerViaMasterList(idb::IdbBuilder* idb_builder);
  void wrapBlockageList(idb::IdbBuilder* idb_builder);
  void wrapArtificialBlockage(idb::IdbBuilder* idb_builder);
  void wrapInstanceBlockage(idb::IdbBuilder* idb_builder);
  void wrapSpecialNetBlockage(idb::IdbBuilder* idb_builder);
  void wrapNetList(idb::IdbBuilder* idb_builder);
  bool checkSkipping(idb::IdbNet* idb_net);
  void wrapPinList(Net& net, idb::IdbNet* idb_net);
  void wrapPinShapeList(Pin& pin, idb::IdbPin* idb_pin);
  void wrapDrivingPin(Net& net, idb::IdbNet* idb_net);
  void updateHelper(idb::IdbBuilder* idb_builder);
  void buildConfig();
  void buildDatabase();
  void buildGCellAxis();
  void makeGCellAxis();
  irt_int getProposedInterval();
  std::vector<irt_int> makeGCellScaleList(Direction direction, irt_int ref_pitch);
  std::vector<GCellGrid> makeGCellGridList(std::vector<irt_int>& gcell_scale_list);
  void checkGCellAxis();
  void buildDie();
  void makeDie();
  void checkDie();
  void buildLayerList();
  void transLayerList();
  void makeLayerList();
  void checkLayerList();
  void buildLayerViaMasterList();
  void transLayerViaMasterList();
  void makeLayerViaMasterList();
  bool sortByMultiLevel(ViaMaster& via_master1, ViaMaster& via_master2);
  SortStatus sortByWidthASC(ViaMaster& via_master1, ViaMaster& via_master2);
  SortStatus sortByLayerDirectionPriority(ViaMaster& via_master1, ViaMaster& via_master2);
  SortStatus sortByLengthASC(ViaMaster& via_master1, ViaMaster& via_master2);
  SortStatus sortBySymmetryPriority(ViaMaster& via_master1, ViaMaster& via_master2);
  void buildBlockageList();
  void transBlockageList();
  void makeBlockageList();
  void checkBlockageList();
  void buildNetList();
  void buildPinList(Net& net);
  void transPinList(Net& net);
  void makePinList(Net& net);
  void checkPinList(Net& net);
  void buildDrivingPin(Net& net);
  void updateHelper();
  void printConfig();
  void printDatabase();
  void saveStageResult(Stage stage);
  void saveHeadInfo(nlohmann::json& all_json, std::string& current_stage);
  std::tuple<std::string, std::string,std::set<std::string>, std::string>  getHeadInfo(const std::string& stage);
  std::string getBaseName(std::string file_path);
  void saveStageNetResult(nlohmann::json& net_json, Net& net);
  void savePinAccessorResult(nlohmann::json& net_json, Net& net);
  void saveBasicInfo(nlohmann::json& net_json, Net& net);
  void saveResourceAllocatorResult(nlohmann::json& net_json, Net& net);
  void saveGlobalRouterResult(nlohmann::json& net_json, Net& net);
  void saveResultTree(nlohmann::json& net_json, Net& net, MTree<RTNode>& node_tree);
  void saveTrackAssignerResult(nlohmann::json& net_json, Net& net);
  void saveDetailedRouterResult(nlohmann::json& net_json, Net& net);
  void saveViolationRepairerResult(nlohmann::json& net_json, Net& net);
  void loadStageResult(Stage stage);
  void checkHeadInfo(nlohmann::json& all_json, std::string current_stage);
  void loadStageNetResult(nlohmann::json& net_json, Net& net);
  void loadBasicInfo(nlohmann::json& net_json, Net& net);
  void loadPinAccessorResult(nlohmann::json& net_json, Net& net);
  void loadResourceAllocatorResult(nlohmann::json& net_json, Net& net);
  void loadGlobalRouterResult(nlohmann::json& net_json, Net& net);
  void loadResultTree(nlohmann::json& net_json, Net& net, MTree<RTNode>& node_tree);
  void loadTrackAssignerResult(nlohmann::json& net_json, Net& net);
  void loadDetailedRouterResult(nlohmann::json& net_json, Net& net);
  void loadViolationRepairerResult(nlohmann::json& net_json, Net& net);
  void outputGCellGrid(idb::IdbBuilder* idb_builder);
  void outputNetList(idb::IdbBuilder* idb_builder);
  void convertToIDBNet(idb::IdbBuilder* idb_builder, Net& net, idb::IdbNet* idb_net);
  void convertToIDBWire(idb::IdbLayers* idb_layer_list, WireNode& wire_node, idb::IdbRegularWireSegment* idb_segment);
  void convertToIDBVia(idb::IdbVias* lef_via_list, idb::IdbVias* def_via_list, ViaNode& via_node, idb::IdbRegularWireSegment* idb_segment);
  void saveDef(idb::IdbBuilder* idb_builder);
  Direction getRTDirectionByDB(idb::IdbLayerDirection idb_direction);
  ConnectType getRTConnectTypeByDB(idb::IdbConnectType idb_connect_type);
};

}  // namespace irt
