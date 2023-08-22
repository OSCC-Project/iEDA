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
#include "DRCRect.hpp"
#include "DataManager.hpp"
#include "Database.hpp"
#include "Net.hpp"
#include "TAModel.hpp"
#include "TAPanel.hpp"

namespace irt {

#define TA_INST (irt::TrackAssigner::getInst())

class TrackAssigner
{
 public:
  static void initInst();
  static TrackAssigner& getInst();
  static void destroyInst();
  // function
  void assign(std::vector<Net>& net_list);

 private:
  // self
  static TrackAssigner* _ta_instance;

  TrackAssigner() = default;
  TrackAssigner(const TrackAssigner& other) = delete;
  TrackAssigner(TrackAssigner&& other) = delete;
  ~TrackAssigner() = default;
  TrackAssigner& operator=(const TrackAssigner& other) = delete;
  TrackAssigner& operator=(TrackAssigner&& other) = delete;
  // function
  void assignNetList(std::vector<Net>& net_list);

#if 1  // init
  TAModel init(std::vector<Net>& net_list);
  TAModel initTAModel(std::vector<Net>& net_list);
  std::vector<TANet> convertToTANetList(std::vector<Net>& net_list);
  TANet convertToTANet(Net& net);
  void buildTAModel(TAModel& ta_model);
  void buildSchedule(TAModel& ta_model);
  void shrinkPanelRegion(TAModel& ta_model);
  void buildPanelTrackAxis(TAModel& ta_model);
  void updateNetFixedRectMap(TAModel& ta_model);
  void updateRectToEnv(TAModel& ta_model, ChangeType change_type, TASourceType ta_source_type, TAPanelId ta_panel_id, DRCRect drc_rect);
  void updateNetEnclosureMap(TAModel& ta_model);
  void buildTATaskList(TAModel& ta_model);
  void buildTATask(TAModel& ta_model, TANet& ta_net);
  std::map<TNode<RTNode>*, TATask> makeTANodeTaskMap(TAModel& ta_model, TANet& ta_net);
  TAGroup makeTAGroup(TAModel& ta_model, TNode<RTNode>* dr_node_node, TNode<RTNode>* ta_node_node, std::vector<LayerCoord>& pin_coord_list);
  std::map<LayerCoord, double, CmpLayerCoordByXASC> makeTACostMap(TNode<RTNode>* ta_node_node,
                                                                  std::map<TNode<RTNode>*, TAGroup>& ta_group_map,
                                                                  std::vector<LayerCoord>& pin_coord_list);
  void buildNetTaskMap(TAModel& ta_model);
  void outputTADataset(TAModel& ta_model);
#endif

#if 1  // iterative
  void iterative(TAModel& ta_model);
  void assignTAModel(TAModel& ta_model);
  void iterativeTAPanel(TAModel& ta_model, TAPanelId& ta_panel_id);
  void buildTAPanel(TAModel& ta_model, TAPanel& ta_panel);
  void initTANodeMap(TAPanel& ta_panel);
  void buildNeighborMap(TAPanel& ta_panel);
  void makeRoutingState(TAPanel& ta_panel);
  void buildSourceOrienTaskMap(TAPanel& ta_panel);
  void updateRectToGraph(TAPanel& ta_panel, ChangeType change_type, TASourceType ta_source_type, DRCRect drc_rect);
  std::map<LayerCoord, std::set<Orientation>, CmpLayerCoordByXASC> getGridOrientationMap(TAPanel& ta_panel, const DRCRect& drc_rect);
  std::vector<Segment<LayerCoord>> getSegmentList(TAPanel& ta_panel, LayerRect min_scope_rect);
  std::vector<LayerRect> getRealRectList(std::vector<Segment<LayerCoord>> segment_list);
  void checkTAPanel(TAPanel& ta_panel);
  void saveTAPanel(TAPanel& ta_panel);
  void resetTAPanel(TAModel& ta_model, TAPanel& ta_panel);
  void sortTAPanel(TAModel& ta_model, TAPanel& ta_panel);
  bool sortByMultiLevel(TATask& task1, TATask& task2);
  SortStatus sortByClockPriority(TATask& task1, TATask& task2);
  SortStatus sortByPreferLengthDESC(TATask& task1, TATask& task2);
  void assignTAPanel(TAModel& ta_model, TAPanel& ta_panel);
  void routeTATask(TAModel& ta_model, TAPanel& ta_panel, TATask& ta_task);
  void initSingleTask(TAPanel& ta_panel, TATask& ta_task);
  bool isConnectedAllEnd(TAPanel& ta_panel);
  void routeByStrategy(TAPanel& ta_panel, TARouteStrategy ta_route_strategy);
  void routeSinglePath(TAPanel& ta_panel);
  void initPathHead(TAPanel& ta_panel);
  bool searchEnded(TAPanel& ta_panel);
  void expandSearching(TAPanel& ta_panel);
  bool passChecking(TAPanel& ta_panel, TANode* start_node, TANode* end_node);
  std::vector<Segment<LayerCoord>> getRoutingSegmentListByNode(TANode* node);
  void resetPathHead(TAPanel& ta_panel);
  bool isRoutingFailed(TAPanel& ta_panel);
  void resetSinglePath(TAPanel& ta_panel);
  void updatePathResult(TAPanel& ta_panel);
  void updateDirectionSet(TAPanel& ta_panel);
  void resetStartAndEnd(TAPanel& ta_panel);
  void updateTaskResult(TAModel& ta_model, TAPanel& ta_panel, TATask& ta_task);
  void resetSingleTask(TAPanel& ta_panel);
  void pushToOpenList(TAPanel& ta_panel, TANode* curr_node);
  TANode* popFromOpenList(TAPanel& ta_panel);
  double getKnowCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node);
  double getNodeCost(TAPanel& ta_panel, TANode* curr_node, Orientation orientation);
  double getKnowWireCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node);
  double getKnowCornerCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node);
  double getKnowViaCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node);
  double getEstimateCostToEnd(TAPanel& ta_panel, TANode* curr_node);
  double getEstimateCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node);
  double getEstimateWireCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node);
  double getEstimateCornerCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node);
  double getEstimateViaCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node);
  void processTAPanel(TAModel& ta_model, TAPanel& ta_panel);
  void buildRoutingResult(TATask& ta_task);
  void countTAPanel(TAModel& ta_model, TAPanel& ta_panel);
  void reportTAPanel(TAModel& ta_model, TAPanel& ta_panel);
  void freeTAPanel(TAModel& ta_model, TAPanel& ta_panel);
  bool stopTAPanel(TAModel& ta_model, TAPanel& ta_panel);
  void countTAModel(TAModel& ta_model);
  void reportTAModel(TAModel& ta_model);
  bool stopTAModel(TAModel& ta_model);
#endif

#if 1  // update
  void update(TAModel& ta_model);
#endif

#if 1  // plot ta_panel
  void plotTAPanel(TAPanel& ta_panel, irt_int curr_task_idx = -1);
#endif
};

}  // namespace irt
