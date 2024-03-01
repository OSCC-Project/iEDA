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
#include "DRModel.hpp"
#include "DRNet.hpp"
#include "DRNode.hpp"
#include "DRParameter.hpp"
#include "DataManager.hpp"
#include "Database.hpp"
#include "Net.hpp"
#include "RTHeader.hpp"

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
  DRModel initDRModel(std::vector<Net>& net_list);
  std::vector<DRNet> convertToDRNetList(std::vector<Net>& net_list);
  DRNet convertToDRNet(Net& net);
  void iterativeDRModel(DRModel& dr_model);
  void setDRParameter(DRModel& dr_model, DRParameter& dr_parameter);
  void initDRBoxMap(DRModel& dr_model);
  void splitNetResult(DRModel& dr_model);
  void splitNetResult(DRBox& dr_box);
  void buildBoxSchedule(DRModel& dr_model);
  void routeDRBoxMap(DRModel& dr_model);
  void initDRTaskList(DRModel& dr_model, DRBox& dr_box);
  std::map<int32_t, std::set<LayerCoord, CmpLayerCoordByLayerASC>> getNetConnectPointMap(DRBox& dr_box);
  std::map<int32_t, std::set<LayerCoord, CmpLayerCoordByLayerASC>> getNetBoundaryPointMap(DRBox& dr_box);
  void buildBoundingBox(DRTask* dr_task);
  void buildDRTaskList(DRBox& dr_box);
  void buildFixedRectList(DRBox& dr_box);
  void buildViolationList(DRBox& dr_box);
  void buildBoxTrackAxis(DRBox& dr_box);
  void initLayerNodeMap(DRBox& dr_box);
  void initDRNodeValid(DRBox& dr_box);
  void buildDRNodeNeighbor(DRBox& dr_box);
  void buildOrienNetMap(DRBox& dr_box);
  void routeDRBox(DRBox& dr_box);
  std::vector<DRTask*> initTaskSchedule(DRBox& dr_box);
  std::vector<DRTask*> getTaskScheduleByViolation(DRBox& dr_box);
  void routeDRTask(DRBox& dr_box, DRTask* dr_task);
  void initSingleTask(DRBox& dr_box, DRTask* dr_task);
  bool isConnectedAllEnd(DRBox& dr_box);
  void routeSinglePath(DRBox& dr_box);
  void initPathHead(DRBox& dr_box);
  bool searchEnded(DRBox& dr_box);
  void expandSearching(DRBox& dr_box);
  void resetPathHead(DRBox& dr_box);
  bool isRoutingFailed(DRBox& dr_box);
  void resetSinglePath(DRBox& dr_box);
  void updatePathResult(DRBox& dr_box);
  std::vector<Segment<LayerCoord>> getRoutingSegmentListByNode(DRNode* node);
  void updateDirectionSet(DRBox& dr_box);
  void resetStartAndEnd(DRBox& dr_box);
  void updateTaskResult(DRBox& dr_box);
  std::vector<Segment<LayerCoord>> getRoutingSegmentList(DRBox& dr_box);
  void resetSingleTask(DRBox& dr_box);
  void pushToOpenList(DRBox& dr_box, DRNode* curr_node);
  DRNode* popFromOpenList(DRBox& dr_box);
  double getKnowCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  double getNodeCost(DRBox& dr_box, DRNode* curr_node, Orientation orientation);
  double getKnowWireCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  double getKnowCornerCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  double getKnowViaCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  double getEstimateCostToEnd(DRBox& dr_box, DRNode* curr_node);
  double getEstimateCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  double getEstimateWireCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  double getEstimateCornerCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  double getEstimateViaCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node);
  void applyPatch(DRBox& dr_box, DRTask* dr_task);
  std::vector<EXTLayerRect> getPatchList(DRBox& dr_box, DRTask* dr_task);
  std::vector<EXTLayerRect> getNotchPatchList(DRBox& dr_box, DRTask* dr_task);
  LayerRect getNotchPatch(int32_t layer_idx, std::vector<PlanarCoord>& task_point_list);
  std::vector<EXTLayerRect> getMinAreaPatchList(DRBox& dr_box, DRTask* dr_task);
  void updateViolationList(DRBox& dr_box);
  std::vector<Violation> getViolationList(DRBox& dr_box);
  void updateDRTaskToGcellMap(DRBox& dr_box);
  void updateViolationToGcellMap(DRBox& dr_box);
  void freeDRBox(DRBox& dr_box);
  int32_t getViolationNum();

#if 1  // update env
  void updateFixedRectToGraph(DRBox& dr_box, ChangeType change_type, int32_t net_idx, EXTLayerRect* fixed_rect, bool is_routing);
  void updateNetResultToGraph(DRBox& dr_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment);
  void updatePatchToGraph(DRBox& dr_box, ChangeType change_type, int32_t net_idx, EXTLayerRect& patch);
  std::map<DRNode*, std::set<Orientation>> getNodeOrientationMap(DRBox& dr_box, NetShape& net_shape);
  std::map<DRNode*, std::set<Orientation>> getRoutingNodeOrientationMap(DRBox& dr_box, NetShape& net_shape);
  std::map<DRNode*, std::set<Orientation>> getCutNodeOrientationMap(DRBox& dr_box, NetShape& net_shape);
  void updateViolationToGraph(DRBox& dr_box, ChangeType change_type, Violation& violation);
#endif

#if 1  // debug
  void debugCheckDRBox(DRBox& dr_box);
  void debugPlotDRBox(DRBox& dr_box, int32_t curr_task_idx, std::string flag);
#endif

#if 1  // exhibit
  void reportSummary(DRModel& dr_model, int32_t iter);
  void writeNetCSV(DRModel& dr_model, int32_t iter);
  void writeViolationCSV(DRModel& dr_model, int32_t iter);
#endif
};

}  // namespace irt
