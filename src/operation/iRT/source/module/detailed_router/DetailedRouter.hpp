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
#include "DRDataManager.hpp"
#include "DRModel.hpp"
#include "DRNode.hpp"
#include "Database.hpp"
#include "Net.hpp"
#include "RTU.hpp"
#include "SortStatus.hpp"

namespace irt {

#define DR_INST (irt::DetailedRouter::getInst())

class DetailedRouter
{
 public:
  static void initInst(Config& config, Database& database);
  static DetailedRouter& getInst();
  static void destroyInst();
  // function
  void route(std::vector<Net>& net_list);

 private:
  // self
  static DetailedRouter* _dr_instance;
  // config & database
  DRDataManager _dr_data_manager;

  DetailedRouter(Config& config, Database& database) { init(config, database); }
  DetailedRouter(const DetailedRouter& other) = delete;
  DetailedRouter(DetailedRouter&& other) = delete;
  ~DetailedRouter() = default;
  DetailedRouter& operator=(const DetailedRouter& other) = delete;
  DetailedRouter& operator=(DetailedRouter&& other) = delete;
  // function
  void init(Config& config, Database& database);
  void routeDRNetList(std::vector<DRNet>& dr_net_list);

#if 1  // build dr_model
  DRModel initDRModel(std::vector<DRNet>& dr_net_list);
  void buildDRModel(DRModel& dr_model);
  void addBlockageList(DRModel& dr_model);
  void addNetRegionList(DRModel& dr_model);
  void buildDRTaskList(DRModel& dr_model);
  std::map<TNode<RTNode>*, DRTask> makeDRNodeTaskMap(DRNet& dr_net);
  std::map<TNode<RTNode>*, std::vector<TNode<RTNode>*>> getDRTAListMap(DRNet& dr_net);
  DRGroup makeDRGroup(TNode<RTNode>* dr_node_node, TNode<RTNode>* ta_node_node);
  void buildBoundingBox(DRBox& dr_box, DRTask& dr_task);
  void buildDRTaskPriority(DRModel& dr_model);
#endif

#if 1  // route dr_model
  void routeDRModel(DRModel& dr_model);
#endif

#if 1  // build dr_box
  void buildDRBox(DRBox& dr_box);
  void initLayerGraphList(DRBox& dr_box);
  void buildScaleOrientList(DRBox& dr_box);
  void buildBasicLayerGraph(DRBox& dr_box);
  void buildCrossLayerGraph(DRBox& dr_box);
  std::vector<LayerCoord> addCoordToGraph(DRBox& dr_box, LayerCoord& added_coord);
  void buildLayerNeighbor(std::vector<LayerCoord>& new_coord_list, LayerCoord& added_coord);
  void buildPlanarNeighbor(std::vector<LayerCoord>& new_coord_list, DRNodeGraph& node_graph, LayerCoord& added_coord, Direction direction);
  void addNeighborToGraph(DRBox& dr_box, LayerCoord& first_coord, LayerCoord& second_coord);
  void buildLayerNodeList(DRBox& dr_box);
  void buildOBSTaskMap(DRBox& dr_box);
  std::map<DRNode*, std::set<Orientation>> getNodeOrientationMap(DRBox& dr_box, LayerRect& blockage);
  std::vector<Segment<DRNode*>> getNodeSegmentList(DRBox& dr_box, LayerRect& blockage);
  void buildCostTaskMap(DRBox& dr_box);
#endif

#if 1  // check dr_box
  void checkDRBox(DRBox& dr_box);
#endif

#if 1  // sort dr_box
  void sortDRBox(DRBox& dr_box);
  bool sortByMultiLevel(DRTask& task1, DRTask& task2);
  SortStatus sortByClockPriority(DRTask& task1, DRTask& task2);
  SortStatus sortByRoutingAreaASC(DRTask& task1, DRTask& task2);
  SortStatus sortByLengthWidthRatioDESC(DRTask& task1, DRTask& task2);
  SortStatus sortByPinNumDESC(DRTask& task1, DRTask& task2);
#endif

#if 1  // route dr_box
  void routeDRBox(DRBox& dr_box);
  void routeDRTask(DRBox& dr_box, DRTask& dr_task);
  void initRoutingInfo(DRBox& dr_box, DRTask& dr_task);
  bool isConnectedAllEnd(DRBox& dr_box);
  void routeSinglePath(DRBox& dr_box);
  void initPathHead(DRBox& dr_box);
  bool searchEnded(DRBox& dr_box);
  void expandSearching(DRBox& dr_box);
  bool passCheckingSegment(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  bool replaceParentNode(DRBox& dr_box, DRNode* parent_node, DRNode* child_node);
  void resetPathHead(DRBox& dr_box);
  void rerouteByEnlarging(DRBox& dr_box);
  bool isRoutingFailed(DRBox& dr_box);
  void resetSinglePath(DRBox& dr_box);
  void rerouteByforcing(DRBox& dr_box);
  void updatePathResult(DRBox& dr_box);
  void resetStartAndEnd(DRBox& dr_box);
  void updateNetResult(DRBox& dr_box, DRTask& dr_task);
  void resetSingleNet(DRBox& dr_box);
  void pushToOpenList(DRBox& dr_box, DRNode* curr_node);
  DRNode* popFromOpenList(DRBox& dr_box);
  double getKnowCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  double getJointCost(DRBox& dr_box, DRNode* curr_node, Orientation orientation);
  double getEstimateCostToEnd(DRBox& dr_box, DRNode* curr_node);
  double getEstimateCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  Orientation getOrientation(DRNode* start_node, DRNode* end_node);
  double getWireCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  double getViaCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
#endif

#if 1  // plot dr_box
  void plotDRBox(DRBox& dr_box, irt_int curr_task_idx = -1);
#endif

#if 1  // count dr_box
  void countDRBox(DRBox& dr_box);
  std::vector<LayerRect> convertToRectList(std::vector<Segment<LayerCoord>>& segment_list);
#endif

#if 1  // update dr_model
  void updateDRModel(DRModel& dr_model);
  void buildRoutingResult(DRTask& dr_task);
  void updateOriginTAResultTree(DRModel& dr_model);
#endif

#if 1  // report dr_model
  void reportDRModel(DRModel& dr_model);
  void countDRModel(DRModel& dr_model);
  void reportTable(DRModel& dr_model);
#endif
};

}  // namespace irt
