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
#include "Monitor.hpp"
#include "SRModel.hpp"

namespace irt {

#define RTSR (irt::SpaceRouter::getInst())

class SpaceRouter
{
 public:
  static void initInst();
  static SpaceRouter& getInst();
  static void destroyInst();
  // function
  void route();

 private:
  // self
  static SpaceRouter* _sr_instance;

  SpaceRouter() = default;
  SpaceRouter(const SpaceRouter& other) = delete;
  SpaceRouter(SpaceRouter&& other) = delete;
  ~SpaceRouter() = default;
  SpaceRouter& operator=(const SpaceRouter& other) = delete;
  SpaceRouter& operator=(SpaceRouter&& other) = delete;
  // function
  SRModel initSRModel();
  std::vector<SRNet> convertToSRNetList(std::vector<Net>& net_list);
  SRNet convertToSRNet(Net& net);
  void buildLayerNodeMap(SRModel& sr_model);
  void buildOrientSupply(SRModel& sr_model);
  void reviseNodeDemand(SRModel& sr_model);
  void routeSRModel(SRModel& sr_model);
  void initRoutingState(SRModel& sr_model);
  void setSRIterParam(SRModel& sr_model, int32_t iter, SRIterParam& sr_iter_param);
  void initSRBoxMap(SRModel& sr_model);
  void resetRoutingState(SRModel& sr_model);
  void buildBoxSchedule(SRModel& sr_model);
  void splitNetResult(SRModel& sr_model);
  void routeSRBoxMap(SRModel& sr_model);
  void buildNetResult(SRBox& sr_box);
  void initSRTaskList(SRModel& sr_model, SRBox& sr_box);
  void buildOverflow(SRModel& sr_model, SRBox& sr_box);
  bool needRouting(SRModel& sr_model, SRBox& sr_box);
  void buildBoxTrackAxis(SRBox& sr_box);
  void buildLayerNodeMap(SRModel& sr_model, SRBox& sr_box);
  void buildSRNodeNeighbor(SRBox& sr_box);
  void buildOrientSupply(SRModel& sr_model, SRBox& sr_box);
  void buildOrientDemand(SRModel& sr_model, SRBox& sr_box);
  void routeSRBox(SRBox& sr_box);
  std::vector<SRTask*> initTaskSchedule(SRBox& sr_box);
  void routeSRTask(SRBox& sr_box, SRTask* sr_task);
  void initSingleTask(SRBox& sr_box, SRTask* sr_task);
  bool isConnectedAllEnd(SRBox& sr_box);
  void routeSinglePath(SRBox& sr_box);
  void initPathHead(SRBox& sr_box);
  bool searchEnded(SRBox& sr_box);
  void expandSearching(SRBox& sr_box);
  void resetPathHead(SRBox& sr_box);
  void updatePathResult(SRBox& sr_box);
  std::vector<Segment<LayerCoord>> getRoutingSegmentListByNode(SRNode* node);
  void resetStartAndEnd(SRBox& sr_box);
  void resetSinglePath(SRBox& sr_box);
  void updateTaskResult(SRBox& sr_box);
  std::vector<Segment<LayerCoord>> getRoutingSegmentList(SRBox& sr_box);
  void resetSingleTask(SRBox& sr_box);
  void pushToOpenList(SRBox& sr_box, SRNode* curr_node);
  SRNode* popFromOpenList(SRBox& sr_box);
  double getKnownCost(SRBox& sr_box, SRNode* start_node, SRNode* end_node);
  double getNodeCost(SRBox& sr_box, SRNode* curr_node, Direction direction);
  double getKnownWireCost(SRBox& sr_box, SRNode* start_node, SRNode* end_node);
  double getKnownViaCost(SRBox& sr_box, SRNode* start_node, SRNode* end_node);
  double getEstimateCostToEnd(SRBox& sr_box, SRNode* curr_node);
  double getEstimateCost(SRBox& sr_box, SRNode* start_node, SRNode* end_node);
  double getEstimateWireCost(SRBox& sr_box, SRNode* start_node, SRNode* end_node);
  double getEstimateViaCost(SRBox& sr_box, SRNode* start_node, SRNode* end_node);
  void updateOverflow(SRBox& sr_box);
  void updateBestResult(SRBox& sr_box);
  void updateTaskSchedule(SRBox& sr_box, std::vector<SRTask*>& routing_task_list);
  void selectBestResult(SRBox& sr_box);
  void uploadBestResult(SRBox& sr_box);
  void freeSRBox(SRBox& sr_box);
  double getOverflow(SRModel& sr_model);
  void uploadNetResult(SRModel& sr_model);
  void updateBestResult(SRModel& sr_model);
  bool stopIteration(SRModel& sr_model, std::vector<SRIterParam>& sr_iter_param_list);
  void selectBestResult(SRModel& sr_model);
  void uploadBestResult(SRModel& sr_model);

#if 1  // update env
  void updateDemandToGraph(SRModel& sr_model, ChangeType change_type, int32_t net_idx, std::set<Segment<LayerCoord>*>& segment_set);
  void updateDemandToGraph(SRBox& sr_box, ChangeType change_type, int32_t net_idx, std::vector<Segment<LayerCoord>>& segment_list);
#endif

#if 1  // exhibit
  void updateSummary(SRModel& sr_model);
  void printSummary(SRModel& sr_model);
  void outputGuide(SRModel& sr_model);
  void outputNetCSV(SRModel& sr_model);
  void outputOverflowCSV(SRModel& sr_model);
  void outputJson(SRModel& sr_model);
  std::string outputNetJson(SRModel& sr_model);
  std::string outputOverflowJson(SRModel& sr_model);
  std::string outputSummaryJson(SRModel& sr_model);
#endif

#if 1  // debug
  void debugPlotSRModel(SRModel& sr_model, std::string flag);
  void debugCheckSRBox(SRBox& sr_box);
  void debugPlotSRBox(SRBox& sr_box, std::string flag);
#endif
};

}  // namespace irt
