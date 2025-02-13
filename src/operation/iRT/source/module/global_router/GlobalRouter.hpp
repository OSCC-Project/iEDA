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
#include "GRModel.hpp"
#include "Monitor.hpp"

namespace irt {

#define RTGR (irt::GlobalRouter::getInst())

class GlobalRouter
{
 public:
  static void initInst();
  static GlobalRouter& getInst();
  static void destroyInst();
  // function
  void route();

 private:
  // self
  static GlobalRouter* _gr_instance;

  GlobalRouter() = default;
  GlobalRouter(const GlobalRouter& other) = delete;
  GlobalRouter(GlobalRouter&& other) = delete;
  ~GlobalRouter() = default;
  GlobalRouter& operator=(const GlobalRouter& other) = delete;
  GlobalRouter& operator=(GlobalRouter&& other) = delete;
  // function
  GRModel initGRModel();
  std::vector<GRNet> convertToGRNetList(std::vector<Net>& net_list);
  GRNet convertToGRNet(Net& net);
  void buildLayerNodeMap(GRModel& gr_model);
  void buildOrientSupply(GRModel& gr_model);
  void resetDemand(GRModel& gr_model);
  void iterativeGRModel(GRModel& gr_model);
  void setGRIterParam(GRModel& gr_model, int32_t iter, GRIterParam& gr_iter_param);
  void initGRBoxMap(GRModel& gr_model);
  void buildBoxSchedule(GRModel& gr_model);
  void splitNetResult(GRModel& gr_model);
  void routeGRBoxMap(GRModel& gr_model);
  void buildNetResult(GRBox& gr_box);
  void initGRTaskList(GRModel& gr_model, GRBox& gr_box);
  void buildOverflow(GRModel& gr_model, GRBox& gr_box);
  bool needRouting(GRModel& gr_model, GRBox& gr_box);
  void buildBoxTrackAxis(GRBox& gr_box);
  void buildLayerNodeMap(GRBox& gr_box);
  void buildGRNodeNeighbor(GRBox& gr_box);
  void buildOrientSupply(GRModel& gr_model, GRBox& gr_box);
  void buildOrientDemand(GRModel& gr_model, GRBox& gr_box);
  void routeGRBox(GRBox& gr_box);
  std::vector<GRTask*> initTaskSchedule(GRBox& gr_box);
  void routeGRTask(GRBox& gr_box, GRTask* gr_task);
  void initSingleTask(GRBox& gr_box, GRTask* gr_task);
  bool isConnectedAllEnd(GRBox& gr_box);
  void routeSinglePath(GRBox& gr_box);
  void initPathHead(GRBox& gr_box);
  bool searchEnded(GRBox& gr_box);
  void expandSearching(GRBox& gr_box);
  void resetPathHead(GRBox& gr_box);
  void updatePathResult(GRBox& gr_box);
  std::vector<Segment<LayerCoord>> getRoutingSegmentListByNode(GRNode* node);
  void resetStartAndEnd(GRBox& gr_box);
  void resetSinglePath(GRBox& gr_box);
  void updateTaskResult(GRBox& gr_box);
  std::vector<Segment<LayerCoord>> getRoutingSegmentList(GRBox& gr_box);
  void resetSingleTask(GRBox& gr_box);
  void pushToOpenList(GRBox& gr_box, GRNode* curr_node);
  GRNode* popFromOpenList(GRBox& gr_box);
  double getKnowCost(GRBox& gr_box, GRNode* start_node, GRNode* end_node);
  double getNodeCost(GRBox& gr_box, GRNode* curr_node, Orientation orientation);
  double getKnowWireCost(GRBox& gr_box, GRNode* start_node, GRNode* end_node);
  double getKnowViaCost(GRBox& gr_box, GRNode* start_node, GRNode* end_node);
  double getEstimateCostToEnd(GRBox& gr_box, GRNode* curr_node);
  double getEstimateCost(GRBox& gr_box, GRNode* start_node, GRNode* end_node);
  double getEstimateWireCost(GRBox& gr_box, GRNode* start_node, GRNode* end_node);
  double getEstimateViaCost(GRBox& gr_box, GRNode* start_node, GRNode* end_node);
  void updateOverflow(GRBox& gr_box);
  void updateBestResult(GRBox& gr_box);
  void updateTaskSchedule(GRBox& gr_box, std::vector<GRTask*>& routing_task_list);
  void selectBestResult(GRBox& gr_box);
  void uploadBestResult(GRBox& gr_box);
  void freeGRBox(GRBox& gr_box);
  int32_t getOverflow(GRModel& gr_model);
  void uploadNetResult(GRModel& gr_model);
  void updateBestResult(GRModel& gr_model);
  bool stopIteration(GRModel& gr_model);
  void selectBestResult(GRModel& gr_model);
  void uploadBestResult(GRModel& gr_model);

#if 1  // update env
  void updateDemandToGraph(GRModel& gr_model, ChangeType change_type, int32_t net_idx, std::set<Segment<LayerCoord>*>& segment_set);
  void updateDemandToGraph(GRBox& gr_box, ChangeType change_type, int32_t net_idx, std::vector<Segment<LayerCoord>>& segment_list);
#endif

#if 1  // exhibit
  void updateSummary(GRModel& gr_model);
  void printSummary(GRModel& gr_model);
  void outputGuide(GRModel& gr_model);
  void outputDemandCSV(GRModel& gr_model);
  void outputOverflowCSV(GRModel& gr_model);
#endif

#if 1  // debug
  void debugCheckGRBox(GRBox& gr_box);
#endif
};

}  // namespace irt
