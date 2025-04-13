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
#include "DataManager.hpp"
#include "Database.hpp"
#include "HRBoxId.hpp"
#include "HRIterParam.hpp"
#include "HRModel.hpp"
#include "HRNet.hpp"
#include "HRNode.hpp"
#include "Net.hpp"
#include "RTHeader.hpp"

namespace irt {

#define RTHR (irt::HybridRouter::getInst())

class HybridRouter
{
 public:
  static void initInst();
  static HybridRouter& getInst();
  static void destroyInst();
  // function
  void route();

 private:
  // self
  static HybridRouter* _hr_instance;

  HybridRouter() = default;
  HybridRouter(const HybridRouter& other) = delete;
  HybridRouter(HybridRouter&& other) = delete;
  ~HybridRouter() = default;
  HybridRouter& operator=(const HybridRouter& other) = delete;
  HybridRouter& operator=(HybridRouter&& other) = delete;
  // function
  HRModel initHRModel();
  std::vector<HRNet> convertToHRNetList(std::vector<Net>& net_list);
  HRNet convertToHRNet(Net& net);
  void updateAccessPoint(HRModel& hr_model);
  void initNetFinalResultMap(HRModel& hr_model);
  void buildNetFinalResultMap(HRModel& hr_model);
  void clearIgnoredViolation(HRModel& hr_model);
  void uploadViolation(HRModel& hr_model);
  std::vector<Violation> getViolationList(HRModel& hr_model);
  void iterativeHRModel(HRModel& hr_model);
  void initRoutingState(HRModel& hr_model);
  void setHRIterParam(HRModel& hr_model, int32_t iter, HRIterParam& hr_iter_param);
  void resetRoutingState(HRModel& hr_model);
  void initHRBoxMap(HRModel& hr_model);
  void buildBoxSchedule(HRModel& hr_model);
  void splitNetResult(HRModel& hr_model);
  void routeHRBoxMap(HRModel& hr_model);
  void buildFixedRect(HRBox& hr_box);
  void buildNetResult(HRBox& hr_box);
  void initHRTaskList(HRModel& hr_model, HRBox& hr_box);
  void buildViolation(HRBox& hr_box);
  bool needRouting(HRBox& hr_box);
  void buildBoxTrackAxis(HRBox& hr_box);
  void buildLayerNodeMap(HRBox& hr_box);
  void buildHRNodeNeighbor(HRBox& hr_box);
  void buildOrientNetMap(HRBox& hr_box);
  void exemptPinShape(HRBox& hr_box);
  void routeHRBox(HRBox& hr_box);
  std::vector<HRTask*> initTaskSchedule(HRBox& hr_box);
  void routeHRTask(HRBox& hr_box, HRTask* hr_task);
  void initSingleTask(HRBox& hr_box, HRTask* hr_task);
  bool isConnectedAllEnd(HRBox& hr_box);
  void routeSinglePath(HRBox& hr_box);
  void initPathHead(HRBox& hr_box);
  bool searchEnded(HRBox& hr_box);
  void expandSearching(HRBox& hr_box);
  void resetPathHead(HRBox& hr_box);
  void updatePathResult(HRBox& hr_box);
  std::vector<Segment<LayerCoord>> getRoutingSegmentListByNode(HRNode* node);
  void resetStartAndEnd(HRBox& hr_box);
  void resetSinglePath(HRBox& hr_box);
  void updateTaskResult(HRBox& hr_box);
  std::vector<Segment<LayerCoord>> getRoutingSegmentList(HRBox& hr_box);
  void resetSingleTask(HRBox& hr_box);
  void pushToOpenList(HRBox& hr_box, HRNode* curr_node);
  HRNode* popFromOpenList(HRBox& hr_box);
  double getKnowCost(HRBox& hr_box, HRNode* start_node, HRNode* end_node);
  double getNodeCost(HRBox& hr_box, HRNode* curr_node, Orientation orientation);
  double getKnowWireCost(HRBox& hr_box, HRNode* start_node, HRNode* end_node);
  double getKnowViaCost(HRBox& hr_box, HRNode* start_node, HRNode* end_node);
  double getEstimateCostToEnd(HRBox& hr_box, HRNode* curr_node);
  double getEstimateCost(HRBox& hr_box, HRNode* start_node, HRNode* end_node);
  double getEstimateWireCost(HRBox& hr_box, HRNode* start_node, HRNode* end_node);
  double getEstimateViaCost(HRBox& hr_box, HRNode* start_node, HRNode* end_node);
  void updateViolationList(HRBox& hr_box);
  std::vector<Violation> getViolationList(HRBox& hr_box);
  void updateBestResult(HRBox& hr_box);
  void updateTaskSchedule(HRBox& hr_box, std::vector<HRTask*>& routing_task_list);
  void selectBestResult(HRBox& hr_box);
  void uploadBestResult(HRBox& hr_box);
  void freeHRBox(HRBox& hr_box);
  int32_t getViolationNum(HRModel& hr_model);
  void uploadNetResult(HRModel& hr_model);
  void updateBestResult(HRModel& hr_model);
  bool stopIteration(HRModel& hr_model);
  void selectBestResult(HRModel& hr_model);
  void uploadBestResult(HRModel& hr_model);

#if 1  // update env
  void updateFixedRectToGraph(HRBox& hr_box, ChangeType change_type, int32_t net_idx, EXTLayerRect* fixed_rect, bool is_routing);
  void updateFixedRectToGraph(HRBox& hr_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment);
  void updateRoutedRectToGraph(HRBox& hr_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment);
  void addViolationToGraph(HRBox& hr_box, Violation& violation);
  void addViolationToGraph(HRBox& hr_box, LayerRect& searched_rect, std::vector<Segment<LayerCoord>>& overlap_segment_list);
  std::map<HRNode*, std::set<Orientation>> getNodeOrientationMap(HRBox& hr_box, NetShape& net_shape);
  std::map<HRNode*, std::set<Orientation>> getRoutingNodeOrientationMap(HRBox& hr_box, NetShape& net_shape);
  std::map<HRNode*, std::set<Orientation>> getCutNodeOrientationMap(HRBox& hr_box, NetShape& net_shape);
#endif

#if 1  // exhibit
  void updateSummary(HRModel& hr_model);
  void printSummary(HRModel& hr_model);
  void outputNetCSV(HRModel& hr_model);
  void outputViolationCSV(HRModel& hr_model);
#endif

#if 1  // debug
  void debugPlotHRModel(HRModel& hr_model, std::string flag);
  void debugCheckHRBox(HRBox& hr_box);
  void debugPlotHRBox(HRBox& hr_box, std::string flag);
#endif
};

}  // namespace irt
