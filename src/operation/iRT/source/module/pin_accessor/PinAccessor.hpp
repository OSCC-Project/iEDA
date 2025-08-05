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
#include "Net.hpp"
#include "PABoxId.hpp"
#include "PAComParam.hpp"
#include "PAIterParam.hpp"
#include "PAModel.hpp"
#include "PANet.hpp"
#include "PANode.hpp"
#include "RTHeader.hpp"

namespace irt {

#define RTPA (irt::PinAccessor::getInst())

class PinAccessor
{
 public:
  static void initInst();
  static PinAccessor& getInst();
  static void destroyInst();
  // function
  void access();

 private:
  // self
  static PinAccessor* _pa_instance;

  PinAccessor() = default;
  PinAccessor(const PinAccessor& other) = delete;
  PinAccessor(PinAccessor&& other) = delete;
  ~PinAccessor() = default;
  PinAccessor& operator=(const PinAccessor& other) = delete;
  PinAccessor& operator=(PinAccessor&& other) = delete;
  // function
  PAModel initPAModel();
  std::vector<PANet> convertToPANetList(std::vector<Net>& net_list);
  PANet convertToPANet(Net& net);
  void setPAComParam(PAModel& pa_model);
  void buildBlockTrimRectMap(PAModel& pa_model);
  void initAccessPointList(PAModel& pa_model);
  std::vector<LayerRect> getLegalShapeList(PAModel& pa_model, int32_t net_idx, PAPin* pa_pin);
  std::vector<PlanarRect> getPlanarLegalRectList(PAModel& pa_model, int32_t curr_net_idx, PAPin* pa_pin, std::vector<EXTLayerRect>& pin_shape_list);
  std::vector<AccessPoint> getAccessPointList(PAModel& pa_model, int32_t pin_idx, std::vector<LayerRect>& legal_shape_list);
  void uniformSampleCoordList(PAModel& pa_model, std::vector<LayerCoord>& layer_coord_list);
  void uploadAccessPointList(PAModel& pa_model);
  void routePAModel(PAModel& pa_model);
  void initRoutingState(PAModel& pa_model);
  void setPAIterParam(PAModel& pa_model, int32_t iter, PAIterParam& pa_iter_param);
  void initPABoxMap(PAModel& pa_model);
  void resetRoutingState(PAModel& pa_model);
  void buildBoxSchedule(PAModel& pa_model);
  void routePABoxMap(PAModel& pa_model);
  void buildFixedRect(PABox& pa_box);
  void buildAccessPoint(PABox& pa_box);
  void buildAccessResult(PABox& pa_box);
  void buildAccessPatch(PABox& pa_box);
  void initPATaskList(PAModel& pa_model, PABox& pa_box);
  void buildRouteViolation(PABox& pa_box);
  bool needRouting(PABox& pa_box);
  void buildBoxTrackAxis(PABox& pa_box);
  void buildLayerNodeMap(PABox& pa_box);
  void buildLayerShadowMap(PABox& pa_box);
  void buildPANodeNeighbor(PABox& pa_box);
  void buildOrientNetMap(PABox& pa_box);
  void buildNetShadowMap(PABox& pa_box);
  void exemptPinShape(PABox& pa_box);
  void routePABox(PABox& pa_box);
  std::vector<PATask*> initTaskSchedule(PABox& pa_box);
  void updateGraph(PABox& pa_box, PATask* pa_task);
  void routePATask(PABox& pa_box, PATask* pa_task);
  void initSingleRouteTask(PABox& pa_box, PATask* pa_task);
  bool isConnectedAllEnd(PABox& pa_box);
  void routeSinglePath(PABox& pa_box);
  void initPathHead(PABox& pa_box);
  bool searchEnded(PABox& pa_box);
  void expandSearching(PABox& pa_box);
  void resetPathHead(PABox& pa_box);
  void updatePathResult(PABox& pa_box);
  std::vector<Segment<LayerCoord>> getRoutingSegmentListByNode(PANode* node);
  void resetStartAndEnd(PABox& pa_box);
  void resetSinglePath(PABox& pa_box);
  void updateTaskResult(PABox& pa_box);
  std::vector<Segment<LayerCoord>> getRoutingSegmentList(PABox& pa_box);
  void resetSingleRouteTask(PABox& pa_box);
  void pushToOpenList(PABox& pa_box, PANode* curr_node);
  PANode* popFromOpenList(PABox& pa_box);
  double getKnownCost(PABox& pa_box, PANode* start_node, PANode* end_node);
  double getNodeCost(PABox& pa_box, PANode* curr_node, Orientation orientation);
  double getKnownWireCost(PABox& pa_box, PANode* start_node, PANode* end_node);
  double getKnownViaCost(PABox& pa_box, PANode* start_node, PANode* end_node);
  double getKnownSelfCost(PABox& pa_box, PANode* start_node, PANode* end_node);
  double getEstimateCostToEnd(PABox& pa_box, PANode* curr_node);
  double getEstimateCost(PABox& pa_box, PANode* start_node, PANode* end_node);
  double getEstimateWireCost(PABox& pa_box, PANode* start_node, PANode* end_node);
  double getEstimateViaCost(PABox& pa_box, PANode* start_node, PANode* end_node);
  void patchPATask(PABox& pa_box, PATask* pa_task);
  void initSinglePatchTask(PABox& pa_box, PATask* pa_task);
  std::vector<Violation> getPatchViolationList(PABox& pa_box, const std::set<ViolationType>& check_type_set, const std::vector<LayerRect>& check_region_list);
  bool searchViolation(PABox& pa_box);
  bool isValidPatchViolation(PABox& pa_box, Violation& violation);
  std::vector<PlanarRect> getViolationOverlapRect(PABox& pa_box, Violation& violation);
  void addViolationToShadow(PABox& pa_box);
  void patchSingleViolation(PABox& pa_box);
  std::vector<PAPatch> getCandidatePatchList(PABox& pa_box);
  bool getSolvedStatus(PABox& pa_box, std::vector<Violation>& origin_patch_violation_list, std::vector<Violation>& curr_patch_violation_list);
  void resetSingleViolation(PABox& pa_box);
  void clearViolationShadow(PABox& pa_box);
  void updateTaskPatch(PABox& pa_box);
  void resetSinglePatchTask(PABox& pa_box);
  void updateRouteViolationList(PABox& pa_box);
  std::vector<Violation> getRouteViolationList(PABox& pa_box);
  void updateAccessPoint(PABox& pa_box);
  void updateBestResult(PABox& pa_box);
  void updateTaskSchedule(PABox& pa_box, std::vector<PATask*>& routing_task_list);
  void selectBestResult(PABox& pa_box);
  void uploadBestResult(PABox& pa_box);
  void freePABox(PABox& pa_box);
  int32_t getRouteViolationNum(PAModel& pa_model);
  void uploadViolation(PAModel& pa_model);
  std::vector<Violation> getRouteViolationList(PAModel& pa_model);
  void updateBestResult(PAModel& pa_model);
  bool stopIteration(PAModel& pa_model);
  void selectBestResult(PAModel& pa_model);
  void uploadBestResult(PAModel& pa_model);
  void uploadAccessPoint(PAModel& pa_model);
  void uploadAccessResult(PAModel& pa_model);
  void uploadAccessPatch(PAModel& pa_model);

#if 1  // update env
  void updateFixedRectToGraph(PABox& pa_box, ChangeType change_type, int32_t net_idx, EXTLayerRect* fixed_rect, bool is_routing);
  void updateFixedRectToGraph(PABox& pa_box, ChangeType change_type, int32_t net_idx, LayerRect& real_rect, bool is_routing);
  void updateFixedRectToGraph(PABox& pa_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>* segment);
  void updateRoutedRectToGraph(PABox& pa_box, ChangeType change_type, int32_t net_idx, LayerRect& real_rect, bool is_routing);
  void updateRoutedRectToGraph(PABox& pa_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment);
  void updateRoutedRectToGraph(PABox& pa_box, ChangeType change_type, int32_t net_idx, EXTLayerRect& routed_rect, bool is_routing);
  void addRouteViolationToGraph(PABox& pa_box, Violation& violation);
  void addRouteViolationToGraph(PABox& pa_box, LayerRect& searched_rect, std::vector<Segment<LayerCoord>>& overlap_segment_list);
  std::map<PANode*, std::set<Orientation>> getNodeOrientationMap(PABox& pa_box, NetShape& net_shape);
  std::map<PANode*, std::set<Orientation>> getRoutingNodeOrientationMap(PABox& pa_box, NetShape& net_shape);
  std::map<PANode*, std::set<Orientation>> getCutNodeOrientationMap(PABox& pa_box, NetShape& net_shape);
  void updateFixedRectToShadow(PABox& pa_box, ChangeType change_type, int32_t net_idx, EXTLayerRect* fixed_rect, bool is_routing);
  void updateFixedRectToShadow(PABox& pa_box, ChangeType change_type, int32_t net_idx, LayerRect& real_rect, bool is_routing);
  void updateFixedRectToShadow(PABox& pa_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>* segment);
  void updateRoutedRectToShadow(PABox& pa_box, ChangeType change_type, int32_t net_idx, LayerRect& real_rect, bool is_routing);
  void updateRoutedRectToShadow(PABox& pa_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment);
  void updateRoutedRectToShadow(PABox& pa_box, ChangeType change_type, int32_t net_idx, EXTLayerRect& routed_rect, bool is_routing);
  void addPatchViolationToShadow(PABox& pa_box, Violation& violation);
  std::vector<PlanarRect> getShadowShape(PABox& pa_box, NetShape& net_shape);
  std::vector<PlanarRect> getRoutingShadowShapeList(PABox& pa_box, NetShape& net_shape);
#endif

#if 1  // get env
  double getFixedRectCost(PABox& pa_box, int32_t net_idx, EXTLayerRect& patch);
  double getRoutedRectCost(PABox& pa_box, int32_t net_idx, EXTLayerRect& patch);
  double getViolationCost(PABox& pa_box, int32_t net_idx, EXTLayerRect& patch);
#endif

#if 1  // exhibit
  void updateSummary(PAModel& pa_model);
  void printSummary(PAModel& pa_model);
  void outputNetCSV(PAModel& pa_model);
  void outputViolationCSV(PAModel& pa_model);
  void outputJson(PAModel& pa_model);
  std::string outputNetJson(PAModel& pa_model);
  std::string outputViolationJson(PAModel& pa_model);
  std::string outputSummaryJson(PAModel& pa_model);
#endif

#if 1  // debug
  void debugPlotPAModel(PAModel& pa_model, std::string flag);
  void debugCheckPABox(PABox& pa_box);
  void debugPlotPABox(PABox& pa_box, std::string flag);
#endif
};

}  // namespace irt
