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
#include "Net.hpp"
#include "SortStatus.hpp"
#include "TADataManager.hpp"
#include "TAModel.hpp"
#include "TAPanel.hpp"

namespace irt {

#define TA_INST (irt::TrackAssigner::getInst())

class TrackAssigner
{
 public:
  static void initInst(Config& config, Database& database);
  static TrackAssigner& getInst();
  static void destroyInst();
  // function
  void assign(std::vector<Net>& net_list);

 private:
  // self
  static TrackAssigner* _ta_instance;
  // config & database
  TADataManager _ta_data_manager;

  TrackAssigner(Config& config, Database& database) { init(config, database); }
  TrackAssigner(const TrackAssigner& other) = delete;
  TrackAssigner(TrackAssigner&& other) = delete;
  ~TrackAssigner() = default;
  TrackAssigner& operator=(const TrackAssigner& other) = delete;
  TrackAssigner& operator=(TrackAssigner&& other) = delete;
  // function
  void init(Config& config, Database& database);
  void assignTANetList(std::vector<TANet>& ta_net_list);

#if 1  // build ta_model
  TAModel initTAModel(std::vector<TANet>& ta_net_list);
  void buildTAModel(TAModel& ta_model);
  void buildTATaskList(TAModel& ta_model);
  std::map<TNode<RTNode>*, TATask> makeTANodeTaskMap(TANet& ta_net);
  void makeGroupAndCost(TANet& ta_net, std::map<TNode<RTNode>*, TATask>& ta_node_task_map);
  TAGroup makeTAGroup(TNode<RTNode>* dr_node_node, TNode<RTNode>* ta_node_node, std::vector<LayerCoord>& pin_coord_list);
  std::map<LayerCoord, double, CmpLayerCoordByXASC> makeTACostMap(TNode<RTNode>* ta_node_node,
                                                                  std::map<TNode<RTNode>*, TAGroup>& ta_group_map,
                                                                  std::vector<LayerCoord>& pin_coord_list);
  void expandCoordCostMap(std::map<TNode<RTNode>*, TATask>& ta_node_task_map);
  void buildPanelRegion(TAModel& ta_model);
  void updateNetBlockageMap(TAModel& ta_model);
  void updateNetFenceRegionMap(TAModel& ta_model);
  void buildTATaskPriority(TAModel& ta_model);
#endif

#if 1  // assign ta_model
  void assignTAModel(TAModel& ta_model);
#endif

#if 1  // build ta_panel
  void buildTAPanel(TAPanel& ta_panel);
  void initTANodeMap(TAPanel& ta_panel);
  void buildNeighborMap(TAPanel& ta_panel);
  void buildOBSTaskMap(TAPanel& ta_panel);
  std::map<PlanarCoord, std::set<Orientation>, CmpPlanarCoordByXASC> getGridOrientationMap(TAPanel& ta_panel,
                                                                                           PlanarRect& enlarge_real_rect);
  std::vector<Segment<LayerCoord>> getRealSegmentList(TAPanel& ta_panel, PlanarRect& enlarge_real_rect);
  std::vector<LayerRect> getRealRectList(std::vector<Segment<LayerCoord>> segment_list);
  void buildFenceTaskMap(TAPanel& ta_panel);
#endif

#if 1  // check ta_panel
  void checkTAPanel(TAPanel& ta_panel);
#endif

#if 1  // sort ta_panel
  void sortTAPanel(TAPanel& ta_panel);
  bool sortByMultiLevel(TATask& task1, TATask& task2);
  SortStatus sortByClockPriority(TATask& task1, TATask& task2);
  SortStatus sortByLengthWidthRatioDESC(TATask& task1, TATask& task2);
#endif

#if 1  // assign ta_panel
  void assignTAPanel(TAPanel& ta_panel);
  void routeTATask(TAPanel& ta_panel, TATask& ta_task);
  void initRoutingInfo(TAPanel& ta_panel, TATask& ta_task);
  bool isConnectedAllEnd(TAPanel& ta_panel);
  void routeSinglePath(TAPanel& ta_panel);
  void initPathHead(TAPanel& ta_panel);
  bool searchEnded(TAPanel& ta_panel);
  void expandSearching(TAPanel& ta_panel);
  bool passCheckingSegment(TAPanel& ta_panel, TANode* start_node, TANode* end_node);
  bool replaceParentNode(TAPanel& ta_panel, TANode* parent_node, TANode* child_node);
  void resetPathHead(TAPanel& ta_panel);
  bool isRoutingFailed(TAPanel& ta_panel);
  void resetSinglePath(TAPanel& ta_panel);
  void rerouteByIgnoring(TAPanel& ta_panel, TARouteStrategy ta_route_strategy);
  void updatePathResult(TAPanel& ta_panel);
  void updateOrientationSet(TAPanel& ta_panel);
  void resetStartAndEnd(TAPanel& ta_panel);
  void updateNetResult(TAPanel& ta_panel, TATask& ta_task);
  void updateENVTaskMap(TAPanel& ta_panel, TATask& ta_task);
  void updateDemand(TAPanel& ta_panel, TATask& ta_task);
  void updateResult(TAPanel& ta_panel, TATask& ta_task);
  void resetSingleNet(TAPanel& ta_panel);
  void pushToOpenList(TAPanel& ta_panel, TANode* curr_node);
  TANode* popFromOpenList(TAPanel& ta_panel);
  double getKnowCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node);
  double getJointCost(TAPanel& ta_panel, TANode* curr_node, Orientation orientation);
  double getKnowWireCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node);
  double getKnowCornerCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node);
  double getViaCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node);
  double getEstimateCostToEnd(TAPanel& ta_panel, TANode* curr_node);
  double getEstimateCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node);
  double getEstimateWireCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node);
  double getEstimateCornerCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node);
#endif

#if 1  // plot ta_panel
  void plotTAPanel(TAPanel& ta_panel, irt_int curr_task_idx = -1);
#endif

#if 1  // update ta_panel
  void updateTAPanel(TAModel& ta_model, TAPanel& ta_panel);
#endif

#if 1  // update ta_model
  void updateTAModel(TAModel& ta_model);
  void buildRoutingResult(TATask& ta_task);
  std::vector<TAGroup> getBoundaryTAGroupList(TATask& ta_task);
  void updateOriginTAResultTree(TAModel& ta_model);
#endif

#if 1  // report ta_model
  void reportTAModel(TAModel& ta_model);
  void countTAModel(TAModel& ta_model);
  void reportTable(TAModel& ta_model);
#endif
};

}  // namespace irt
