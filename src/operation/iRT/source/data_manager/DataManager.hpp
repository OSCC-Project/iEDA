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

#include "ChangeType.hpp"
#include "Config.hpp"
#include "Database.hpp"
#include "Helper.hpp"
#include "Logger.hpp"
#include "NetShape.hpp"
#include "Reporter.hpp"
#include "SortStatus.hpp"

namespace irt {

#define DM_INST (irt::DataManager::getInst())

class DataManager
{
 public:
  static void initInst();
  static DataManager& getInst();
  static void destroyInst();
  // function
  void input(std::map<std::string, std::any>& config_map, idb::IdbBuilder* idb_builder);
  void output(idb::IdbBuilder* idb_builder);

#if 1  // 有关GCellMap操作
  void updateFixedRectToGCellMap(ChangeType change_type, irt_int net_idx, EXTLayerRect* ext_layer_rect, bool is_routing);
  void updateAccessPointToGCellMap(ChangeType change_type, irt_int net_idx, AccessPoint* access_point);
  void updateNetResultToGCellMap(ChangeType change_type, irt_int net_idx, Segment<LayerCoord>* segment);
  void updateViolationToGCellMap(ChangeType change_type, Violation* violation);
  void updatePatchToGCellMap(ChangeType change_type, irt_int net_idx, EXTLayerRect* ext_layer_rect);
  std::map<bool, std::map<irt_int, std::map<irt_int, std::set<EXTLayerRect*>>>> getTypeLayerNetFixedRectMap(EXTPlanarRect& region);
  std::map<irt_int, std::set<AccessPoint*>> getNetAccessPointMap(EXTPlanarRect& region);
  std::map<irt_int, std::set<Segment<LayerCoord>*>> getNetResultMap(EXTPlanarRect& region);
  std::set<Violation*> getViolationSet(EXTPlanarRect& region);
  std::map<irt_int, std::set<EXTLayerRect*>> getNetPatchMap(EXTPlanarRect& region);
#endif

#if 1  // 获得NetShapeList
  std::vector<NetShape> getNetShapeList(irt_int net_idx, std::vector<Segment<LayerCoord>>& segment_list);
  std::vector<NetShape> getNetShapeList(irt_int net_idx, Segment<LayerCoord>& segment);
  std::vector<NetShape> getNetShapeList(irt_int net_idx, MTree<LayerCoord>& coord_tree);
  std::vector<NetShape> getNetShapeList(irt_int net_idx, LayerCoord& first_coord, LayerCoord& second_coord);
#endif

#if 1  // 获得IdbWireSegment
  idb::IdbLayerShape* getIDBLayerShapeByFixedRect(EXTLayerRect* fixed_rect, bool is_routing);
  idb::IdbRegularWireSegment* getIDBSegmentByNetResult(irt_int net_idx, Segment<LayerCoord>& segment);
  idb::IdbRegularWireSegment* getIDBSegmentByNetPatch(irt_int net_idx, EXTLayerRect& ext_layer_rect);
#endif

  Config& getConfig() { return _config; }
  Database& getDatabase() { return _database; }
  Helper& getHelper() { return _helper; }
  Reporter& getReporter() { return _reporter; }

 private:
  static DataManager* _dm_instance;
  // config & database & helper
  Config _config;
  Database _database;
  Helper _helper;
  Reporter _reporter;

  DataManager() = default;
  DataManager(const DataManager& other) = delete;
  DataManager(DataManager&& other) = delete;
  ~DataManager() = default;
  DataManager& operator=(const DataManager& other) = delete;
  DataManager& operator=(DataManager&& other) = delete;

#if 1  // input
  void wrapConfig(std::map<std::string, std::any>& config_map);
  void wrapDatabase(idb::IdbBuilder* idb_builder);
  void wrapMicronDBU(idb::IdbBuilder* idb_builder);
  void wrapDie(idb::IdbBuilder* idb_builder);
  void wrapRow(idb::IdbBuilder* idb_builder);
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
  void processEmptyShapePin(Net& net);
  void wrapDrivingPin(Net& net, idb::IdbNet* idb_net);
  void updateHelper(idb::IdbBuilder* idb_builder);
  Direction getRTDirectionByDB(idb::IdbLayerDirection idb_direction);
  ConnectType getRTConnectTypeByDB(idb::IdbConnectType idb_connect_type);
  void buildConfig();
  void buildDatabase();
  void buildGCellAxis();
  void makeGCellAxis();
  irt_int getRecommendedPitch();
  std::vector<ScaleGrid> makeGCellGridList(Direction direction, irt_int recommended_pitch);
  std::vector<ScaleGrid> makeGCellGridList(std::vector<irt_int>& gcell_scale_list);
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
  void buildGCellMap();
  void updateHelper();
  void printConfig();
  void printDatabase();
  void writePYScript();
#endif

#if 1  // output
  void outputGCellGrid(idb::IdbBuilder* idb_builder);
  void outputNetList(idb::IdbBuilder* idb_builder);
#endif

#if 1  // 获得IdbWireSegment
  idb::IdbRegularWireSegment* getIDBWire(irt_int net_idx, Segment<LayerCoord>& segment);
  idb::IdbRegularWireSegment* getIDBVia(irt_int net_idx, Segment<LayerCoord>& segment);
#endif
};

}  // namespace irt
