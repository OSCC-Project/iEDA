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
#include "Logger.hpp"
#include "Monitor.hpp"
#include "NetShape.hpp"
#include "SortStatus.hpp"

namespace irt {

#define RTDM (irt::DataManager::getInst())

class DataManager
{
 public:
  static void initInst();
  static DataManager& getInst();
  static void destroyInst();
  // function
  void input(std::map<std::string, std::any>& config_map);
  void output();

#if 1  // 更新GCellMap
  void updateFixedRectToGCellMap(ChangeType change_type, int32_t net_idx, EXTLayerRect* ext_layer_rect, bool is_routing);
  void updateNetAccessPointToGCellMap(ChangeType change_type, int32_t net_idx, AccessPoint* access_point);
  void updateNetPinAccessResultToGCellMap(ChangeType change_type, int32_t net_idx, int32_t pin_idx, Segment<LayerCoord>* segment);
  void updateNetPinAccessPatchToGCellMap(ChangeType change_type, int32_t net_idx, int32_t pin_idx, EXTLayerRect* ext_layer_rect);
  void updateNetGlobalResultToGCellMap(ChangeType change_type, int32_t net_idx, Segment<LayerCoord>* segment);
  void updateNetDetailedResultToGCellMap(ChangeType change_type, int32_t net_idx, Segment<LayerCoord>* segment);
  void updateNetDetailedPatchToGCellMap(ChangeType change_type, int32_t net_idx, EXTLayerRect* ext_layer_rect);
  void updateViolationToGCellMap(ChangeType change_type, Violation* violation);
  std::map<bool, std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>> getTypeLayerNetFixedRectMap(EXTPlanarRect& region);
  std::map<int32_t, std::set<AccessPoint*, CmpAccessPoint>> getNetAccessPointMap(EXTPlanarRect& region);
  std::map<int32_t, std::map<int32_t, std::set<Segment<LayerCoord>*>>> getNetPinAccessResultMap(EXTPlanarRect& region);
  std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>> getNetPinAccessPatchMap(EXTPlanarRect& region);
  std::map<int32_t, std::set<Segment<LayerCoord>*>> getNetGlobalResultMap(EXTPlanarRect& region);
  std::map<int32_t, std::set<Segment<LayerCoord>*>> getNetDetailedResultMap(EXTPlanarRect& region);
  std::map<int32_t, std::set<EXTLayerRect*>> getNetDetailedPatchMap(EXTPlanarRect& region);
  std::set<Violation*, CmpViolation> getViolationSet(EXTPlanarRect& region);
#endif

#if 1  // 获得NetShapeList
  std::vector<NetShape> getNetGlobalShapeList(int32_t net_idx, std::vector<Segment<LayerCoord>>& segment_list);
  std::vector<NetShape> getNetGlobalShapeList(int32_t net_idx, Segment<LayerCoord>& segment);
  std::vector<NetShape> getNetGlobalShapeList(int32_t net_idx, MTree<LayerCoord>& coord_tree);
  std::vector<NetShape> getNetGlobalShapeList(int32_t net_idx, LayerCoord& first_coord, LayerCoord& second_coord);
  std::vector<NetShape> getNetDetailedShapeList(int32_t net_idx, std::vector<Segment<LayerCoord>>& segment_list);
  std::vector<NetShape> getNetDetailedShapeList(int32_t net_idx, Segment<LayerCoord>& segment);
  std::vector<NetShape> getNetDetailedShapeList(int32_t net_idx, MTree<LayerCoord>& coord_tree);
  std::vector<NetShape> getNetDetailedShapeList(int32_t net_idx, LayerCoord& first_coord, LayerCoord& second_coord);
#endif

#if 1  // 获得唯一的结果
  int32_t getOnlyOffset();
  int32_t getOnlyPitch();
#endif

  Config& getConfig() { return _config; }
  Database& getDatabase() { return _database; }

 private:
  static DataManager* _dm_instance;
  // config & database
  Config _config;
  Database _database;

  DataManager() = default;
  DataManager(const DataManager& other) = delete;
  DataManager(DataManager&& other) = delete;
  ~DataManager() = default;
  DataManager& operator=(const DataManager& other) = delete;
  DataManager& operator=(DataManager&& other) = delete;

#if 1  // build
  void buildConfig();
  void buildDatabase();
  void buildLayerList();
  void transLayerList();
  void makeLayerList();
  void makeRoutingLayerList();
  void makeCutLayerList();
  void checkLayerList();
  void buildLayerInfo();
  void buildGCellAxis();
  void makeGCellAxis();
  std::vector<ScaleGrid> makeGCellGridList(Direction direction);
  void checkGCellAxis();
  void buildDie();
  void makeDie();
  void checkDie();
  void buildLayerViaMasterList();
  void transLayerViaMasterList();
  void makeLayerViaMasterList();
  void buildLayerViaMasterInfo();
  void buildObstacleList();
  void transObstacleList();
  void makeObstacleList();
  void checkObstacleList();
  void buildNetList();
  void buildPinList(Net& net);
  void transPinList(Net& net);
  void makePinList(Net& net);
  void checkPinList(Net& net);
  void buildDetectionDistance();
  void buildGCellMap();
  void initGCellMap();
  void updateGCellMap();
  int32_t getBucketIdx(int32_t scale_start, int32_t scale_end, int32_t bucket_start, int32_t bucket_end, int32_t bucket_length);
  void buildFixRectMap();
  void printConfig();
  void printDatabase();
  void outputScript();
  void outputJson();
  std::string outputEnvJson();
#endif

#if 1  // destroy
  void destroyGCellMap();
#endif
};

}  // namespace irt
