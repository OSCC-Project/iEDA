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
#include "DRBoxId.hpp"
#include "DRIterParam.hpp"
#include "DRModel.hpp"
#include "DRNet.hpp"
#include "DRNode.hpp"
#include "DRPatch.hpp"
#include "DataManager.hpp"
#include "Database.hpp"
#include "Net.hpp"
#include "RTHeader.hpp"

namespace irt {

#define RTDR (irt::DetailedRouter::getInst())

class DetailedRouter
{
 public:
  static void initInst();
  static DetailedRouter& getInst();
  static void destroyInst();
  // function
  void route();

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
  DRModel initDRModel();
  std::vector<DRNet> convertToDRNetList(std::vector<Net>& net_list);
  DRNet convertToDRNet(Net& net);
  void routeDRModel(DRModel& dr_model);
  void initRoutingState(DRModel& dr_model);
  void setDRIterParam(DRModel& dr_model, int32_t iter, DRIterParam& dr_iter_param);
  void initDRBoxMap(DRModel& dr_model);
  void resetRoutingState(DRModel& dr_model);
  void buildBoxSchedule(DRModel& dr_model);
  void splitNetResult(DRModel& dr_model);
  void routeDRBoxMap(DRModel& dr_model);
  void buildFixedRect(DRBox& dr_box);
  void buildAccessPoint(DRBox& dr_box);
  void buildNetResult(DRBox& dr_box);
  void buildNetPatch(DRBox& dr_box);
  void initDRTaskList(DRModel& dr_model, DRBox& dr_box);
  void buildRouteViolation(DRBox& dr_box);
  bool needRouting(DRBox& dr_box);
  void buildBoxTrackAxis(DRBox& dr_box);
  void buildLayerNodeMap(DRBox& dr_box);
  void buildLayerShadowMap(DRBox& dr_box);
  void buildDRNodeNeighbor(DRBox& dr_box);
  void buildOrientNetMap(DRBox& dr_box);
  void buildNetShadowMap(DRBox& dr_box);
  void exemptPinShape(DRBox& dr_box);
  void routeDRBox(DRBox& dr_box);
  std::vector<DRTask*> initTaskSchedule(DRBox& dr_box);
  void updateGraph(DRBox& dr_box, DRTask* dr_task);
  void routeDRTask(DRBox& dr_box, DRTask* dr_task);
  void initSingleRouteTask(DRBox& dr_box, DRTask* dr_task);
  bool isConnectedAllEnd(DRBox& dr_box);
  void routeSinglePath(DRBox& dr_box);
  void initPathHead(DRBox& dr_box);
  bool searchEnded(DRBox& dr_box);
  void expandSearching(DRBox& dr_box);
  void resetPathHead(DRBox& dr_box);
  void updatePathResult(DRBox& dr_box);
  std::vector<Segment<LayerCoord>> getRoutingSegmentListByNode(DRNode* node);
  void resetStartAndEnd(DRBox& dr_box);
  void resetSinglePath(DRBox& dr_box);
  void updateTaskResult(DRBox& dr_box);
  std::vector<Segment<LayerCoord>> getRoutingSegmentList(DRBox& dr_box);
  void resetSingleRouteTask(DRBox& dr_box);
  void pushToOpenList(DRBox& dr_box, DRNode* curr_node);
  DRNode* popFromOpenList(DRBox& dr_box);
  double getKnownCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  double getNodeCost(DRBox& dr_box, DRNode* curr_node, Orientation orientation);
  double getKnownWireCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  double getKnownViaCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  double getKnownSelfCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  double getEstimateCostToEnd(DRBox& dr_box, DRNode* curr_node);
  double getEstimateCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  double getEstimateWireCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  double getEstimateViaCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  void patchDRTask(DRBox& dr_box, DRTask* dr_task);
  void initSinglePatchTask(DRBox& dr_box, DRTask* dr_task);
  std::vector<Violation> getPatchViolationList(DRBox& dr_box);
  bool searchViolation(DRBox& dr_box);
  bool isValidPatchViolation(DRBox& dr_box, Violation& violation);
  std::vector<PlanarRect> getViolationOverlapRect(DRBox& dr_box, Violation& violation);
  void addViolationToShadow(DRBox& dr_box);
  void patchSingleViolation(DRBox& dr_box);
  std::vector<DRPatch> getCandidatePatchList(DRBox& dr_box);
  void buildSingleViolation(DRBox& dr_box, DRPatch& dr_patch);
  void updateSingleViolation(DRBox& dr_box);
  void updateTriedFixViolation(DRBox& dr_box);
  void resetSingleViolation(DRBox& dr_box);
  void clearViolationShadow(DRBox& dr_box);
  void updateTaskPatch(DRBox& dr_box);
  void resetSinglePatchTask(DRBox& dr_box);
  void updateRouteViolationList(DRBox& dr_box);
  std::vector<Violation> getRouteViolationList(DRBox& dr_box);
  void updateBestResult(DRBox& dr_box);
  void updateTaskSchedule(DRBox& dr_box, std::vector<DRTask*>& routing_task_list);
  void selectBestResult(DRBox& dr_box);
  void uploadBestResult(DRBox& dr_box);
  void freeDRBox(DRBox& dr_box);
  int32_t getRouteViolationNum(DRModel& dr_model);
  void uploadNetResult(DRModel& dr_model);
  void uploadNetPatch(DRModel& dr_model);
  void uploadViolation(DRModel& dr_model);
  std::vector<Violation> getRouteViolationList(DRModel& dr_model);
  void updateBestResult(DRModel& dr_model);
  bool stopIteration(DRModel& dr_model);
  void selectBestResult(DRModel& dr_model);
  void uploadBestResult(DRModel& dr_model);

#if 1  // update env
  void updateFixedRectToGraph(DRBox& dr_box, ChangeType change_type, int32_t net_idx, EXTLayerRect* fixed_rect, bool is_routing);
  void updateFixedRectToGraph(DRBox& dr_box, ChangeType change_type, int32_t net_idx, LayerRect& real_rect, bool is_routing);
  void updateFixedRectToGraph(DRBox& dr_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>* segment);
  void updateRoutedRectToGraph(DRBox& dr_box, ChangeType change_type, int32_t net_idx, LayerRect& real_rect, bool is_routing);
  void updateRoutedRectToGraph(DRBox& dr_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment);
  void updateRoutedRectToGraph(DRBox& dr_box, ChangeType change_type, int32_t net_idx, EXTLayerRect& routed_rect, bool is_routing);
  void addRouteViolationToGraph(DRBox& dr_box, Violation& violation);
  void addRouteViolationToGraph(DRBox& dr_box, LayerRect& searched_rect, std::vector<Segment<LayerCoord>>& overlap_segment_list);
  std::map<DRNode*, std::set<Orientation>> getNodeOrientationMap(DRBox& dr_box, NetShape& net_shape);
  std::map<DRNode*, std::set<Orientation>> getRoutingNodeOrientationMap(DRBox& dr_box, NetShape& net_shape);
  std::map<DRNode*, std::set<Orientation>> getCutNodeOrientationMap(DRBox& dr_box, NetShape& net_shape);
  void updateFixedRectToShadow(DRBox& dr_box, ChangeType change_type, int32_t net_idx, EXTLayerRect* fixed_rect, bool is_routing);
  void updateFixedRectToShadow(DRBox& dr_box, ChangeType change_type, int32_t net_idx, LayerRect& real_rect, bool is_routing);
  void updateFixedRectToShadow(DRBox& dr_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>* segment);
  void updateRoutedRectToShadow(DRBox& dr_box, ChangeType change_type, int32_t net_idx, LayerRect& real_rect, bool is_routing);
  void updateRoutedRectToShadow(DRBox& dr_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment);
  void updateRoutedRectToShadow(DRBox& dr_box, ChangeType change_type, int32_t net_idx, EXTLayerRect& routed_rect, bool is_routing);
  void addPatchViolationToShadow(DRBox& dr_box, Violation& violation);
  std::vector<PlanarRect> getShadowShape(DRBox& dr_box, NetShape& net_shape);
  std::vector<PlanarRect> getRoutingShadowShapeList(DRBox& dr_box, NetShape& net_shape);
#endif

#if 1  // get env
  double getFixedRectCost(DRBox& dr_box, int32_t net_idx, EXTLayerRect& patch);
  double getRoutedRectCost(DRBox& dr_box, int32_t net_idx, EXTLayerRect& patch);
  double getViolationCost(DRBox& dr_box, int32_t net_idx, EXTLayerRect& patch);
#endif

#if 1  // exhibit
  void updateSummary(DRModel& dr_model);
  void printSummary(DRModel& dr_model);
  void outputNetCSV(DRModel& dr_model);
  void outputViolationCSV(DRModel& dr_model);
  void outputNetJson(DRModel& dr_model);
  void outputViolationJson(DRModel& dr_model);
  void outputSummaryJson(DRModel& dr_model);
#endif

#if 1  // debug
  void debugPlotDRModel(DRModel& dr_model, std::string flag);
  void debugCheckDRBox(DRBox& dr_box);
  void debugPlotDRBox(DRBox& dr_box, std::string flag);
#endif
};

}  // namespace irt
