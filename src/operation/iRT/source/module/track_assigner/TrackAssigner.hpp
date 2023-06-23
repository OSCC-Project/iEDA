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
#include "DataManager.hpp"
#include "Database.hpp"
#include "Net.hpp"
#include "SortStatus.hpp"
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

#if 1  // build ta_model
  TAModel initTAModel(std::vector<Net>& net_list);
  std::vector<TANet> convertToTANetList(std::vector<Net>& net_list);
  TANet convertToTANet(Net& net);
  void buildTAModel(TAModel& ta_model);
  void updateNetBlockageMap(TAModel& ta_model);
  void buildPanelScaleAxis(TAModel& ta_model);
  void buildTATaskList(TAModel& ta_model);
  std::map<TNode<RTNode>*, TATask> makeTANodeTaskMap(std::vector<std::vector<TAPanel>>& layer_panel_list, TANet& ta_net);
  TAGroup makeTAGroup(TAPanel& ta_panel, TNode<RTNode>* dr_node_node);
  void buildLayerPanelList(TAModel& ta_model);
  void initTANodeMap(TAPanel& ta_panel);
  void buildNeighborMap(TAPanel& ta_panel);
  void buildOBSTaskMap(TAPanel& ta_panel);
  std::map<PlanarCoord, std::set<Orientation>, CmpPlanarCoordByXASC> getGridOrientationMap(TAPanel& ta_panel,
                                                                                           PlanarRect& min_scope_regular_rect);
  std::vector<Segment<LayerCoord>> getRealSegmentList(TAPanel& ta_panel, PlanarRect& min_scope_regular_rect);
  std::vector<LayerRect> getRealRectList(std::vector<Segment<LayerCoord>> segment_list);
  void checkTAPanel(TAPanel& ta_panel);
  void saveTAPanel(TAPanel& ta_panel);
#endif

#if 1  // assign ta_model
  void assignTAModel(TAModel& ta_model);
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
  void updateDirectionSet(TAPanel& ta_panel);
  void resetStartAndEnd(TAPanel& ta_panel);
  void updateNetResult(TAPanel& ta_panel, TATask& ta_task);
  void updateResult(TAPanel& ta_panel, TATask& ta_task);
  void resetSingleNet(TAPanel& ta_panel);
  void pushToOpenList(TAPanel& ta_panel, TANode* curr_node);
  TANode* popFromOpenList(TAPanel& ta_panel);
  double getKnowCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node);
  double getJointCost(TAPanel& ta_panel, TANode* curr_node, Orientation orientation);
  double getWireCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node);
  double getKnowCornerCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node);
  double getViaCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node);
  double getEstimateCostToEnd(TAPanel& ta_panel, TANode* curr_node);
  double getEstimateCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node);
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
  void updateOriginTAResultTree(TAModel& ta_model);
#endif

#if 1  // report ta_model
  void reportTAModel(TAModel& ta_model);
  void countTAModel(TAModel& ta_model);
  void reportTable(TAModel& ta_model);
#endif
};

}  // namespace irt
