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
#include "DRModel.hpp"
#include "DRNet.hpp"
#include "DRNode.hpp"
#include "DataManager.hpp"
#include "Database.hpp"
#include "Net.hpp"
#include "RTU.hpp"

namespace irt {

#define DR_INST (irt::DetailedRouter::getInst())

class DetailedRouter
{
 public:
  static void initInst();
  static DetailedRouter& getInst();
  static void destroyInst();
  // function
  void route(std::vector<Net>& net_list);

 private:
  // self
  static DetailedRouter* _dr_instance;

  DetailedRouter() = default;
  DetailedRouter(const DetailedRouter& other) = delete;
  DetailedRouter(DetailedRouter&& other) = delete;
  ~DetailedRouter() = default;
  DetailedRouter& operator=(const DetailedRouter& other) = delete;
  DetailedRouter& operator=(DetailedRouter&& other) = delete;
  // function
  void routeNetList(std::vector<Net>& net_list);

#if 1  // build dr_model
  DRModel initDRModel(std::vector<Net>& net_list);
  std::vector<DRNet> convertToDRNetList(std::vector<Net>& net_list);
  DRNet convertToDRNet(Net& net);
  void buildDRModel(DRModel& dr_model);
  void updateNetBlockageMap(DRModel& dr_model);
  void updateNetPanelResultMap(DRModel& dr_model);
  void buildBoxScaleAxis(DRModel& dr_model);
  void buildDRTaskList(DRModel& dr_model);
  std::map<TNode<RTNode>*, DRTask> makeDRNodeTaskMap(GridMap<DRBox>& dr_box_map, DRNet& dr_net);
  DRGroup makeDRGroup(DRBox& dr_box, DRPin& dr_pin);
  DRGroup makeDRGroup(DRBox& dr_box, TNode<RTNode>* ta_node_node);
  void buildBoundingBox(DRBox& dr_box, DRTask& dr_task);
  void buildDRBoxMap(DRModel& dr_model);
  void initLayerNodeMap(DRBox& dr_box);
  void buildNeighborMap(DRBox& dr_box);
  void buildOBSTaskMap(DRBox& dr_box);
  std::map<LayerCoord, std::set<Orientation>, CmpLayerCoordByLayerASC> getGridOrientationMap(DRBox& dr_box,
                                                                                             LayerRect& min_scope_regular_rect);
  std::vector<Segment<LayerCoord>> getRealSegmentList(DRBox& dr_box, LayerRect& min_scope_regular_rect);
  void checkDRBox(DRBox& dr_box);
  void saveDRBox(DRBox& dr_box);
#endif

#if 1  // route dr_model
  void routeDRModel(DRModel& dr_model);
#endif

#if 1  // sort dr_task_list
  void resetDRBox(DRBox& dr_box);
#endif

#if 1  // init dr_task_list
  void sortDRTaskList(DRBox& dr_box);
#endif

#if 1  // route dr_box
  void routeDRBox(DRBox& dr_box);
  void routeDRTask(DRBox& dr_box, DRTask& dr_task);
  void initSingleTask(DRBox& dr_box, DRTask& dr_task);
  bool isConnectedAllEnd(DRBox& dr_box);
  void routeByStrategy(DRBox& dr_box, DRRouteStrategy dr_route_strategy);
  void routeSinglePath(DRBox& dr_box);
  void initPathHead(DRBox& dr_box);
  bool searchEnded(DRBox& dr_box);
  void expandSearching(DRBox& dr_box);
  bool passCheckingSegment(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  bool passCheckingByDynamicDRC(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  std::vector<Segment<LayerCoord>> getRoutingSegmentListByPathHead(DRBox& dr_box);
  bool replaceParentNode(DRBox& dr_box, DRNode* parent_node, DRNode* child_node);
  void resetPathHead(DRBox& dr_box);
  bool isRoutingFailed(DRBox& dr_box);
  void resetSinglePath(DRBox& dr_box);
  void updatePathResult(DRBox& dr_box);
  void updateDirectionSet(DRBox& dr_box);
  void resetStartAndEnd(DRBox& dr_box);
  void updateTaskResult(DRBox& dr_box, DRTask& dr_task);
  void resetSingleTask(DRBox& dr_box);
  void pushToOpenList(DRBox& dr_box, DRNode* curr_node);
  DRNode* popFromOpenList(DRBox& dr_box);
  double getKnowCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  double getJointCost(DRBox& dr_box, DRNode* curr_node, Orientation orientation);
  double getKnowWireCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  double getKnowCornerCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  double getViaCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  double getEstimateCostToEnd(DRBox& dr_box, DRNode* curr_node);
  double getEstimateCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  double getEstimateWireCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  double getEstimateCornerCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
#endif

#if 1  // count dr_box
  void countDRBox(DRBox& dr_box);
#endif

#if 1  // sort dr_task_list
  void updateBestRouteResult(DRBox& dr_box, DRBoxStat& best_stat, std::map<irt_int, std::vector<Segment<LayerCoord>>>& best_route_result);
#endif

#if 1  // sort dr_task_list
  void updateDRRouteResult(DRBox& dr_box, DRBoxStat& best_stat, std::map<irt_int, std::vector<Segment<LayerCoord>>>& best_route_result);
#endif

#if 1  // plot dr_box
  void plotDRBox(DRBox& dr_box, irt_int curr_task_idx = -1);
#endif

#if 1  // update dr_box
  void updateDRBox(DRModel& dr_model, DRBox& dr_box);
#endif

#if 1  // update dr_model
  void updateDRModel(DRModel& dr_model);
  void buildRoutingResult(DRTask& dr_task);
  void updateOriginDRResultTree(DRModel& dr_model);
#endif

#if 1  // report dr_model
  void reportDRModel(DRModel& dr_model);
  void countDRModel(DRModel& dr_model);
  void reportTable(DRModel& dr_model);
#endif
};

}  // namespace irt
