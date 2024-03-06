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
#include "IRModel.hpp"
#include "RTHeader.hpp"
#include "flute3/flute.h"

namespace irt {

#define IR_INST (irt::InitialRouter::getInst())

class InitialRouter
{
 public:
  static void initInst();
  static InitialRouter& getInst();
  static void destroyInst();
  // function
  void route();

 private:
  // self
  static InitialRouter* _ir_instance;

  InitialRouter() { Flute::readLUT(); }
  InitialRouter(const InitialRouter& other) = delete;
  InitialRouter(InitialRouter&& other) = delete;
  ~InitialRouter() { Flute::deleteLUT(); }
  InitialRouter& operator=(const InitialRouter& other) = delete;
  InitialRouter& operator=(InitialRouter&& other) = delete;
  // function
  IRModel initIRModel();
  std::vector<IRNet> convertToIRNetList(std::vector<Net>& net_list);
  IRNet convertToIRNet(Net& net);
  void setIRParameter(IRModel& ir_model);
  void makeGridCoordList(IRModel& ir_model);
  void initLayerNodeMap(IRModel& ir_model);
  void buildIRNodeNeighbor(IRModel& ir_model);
  void buildOrienSupply(IRModel& ir_model);
  void sortIRModel(IRModel& ir_model);
  bool sortByMultiLevel(IRModel& ir_model, int32_t net_idx1, int32_t net_idx2);
  SortStatus sortByClockPriority(IRNet& net1, IRNet& net2);
  SortStatus sortByRoutingAreaASC(IRNet& net1, IRNet& net2);
  SortStatus sortByLengthWidthRatioDESC(IRNet& net1, IRNet& net2);
  SortStatus sortByPinNumDESC(IRNet& net1, IRNet& net2);
  void routeIRModel(IRModel& ir_model);
  void routeIRNet(IRModel& ir_model, IRNet& ir_net);
  void makeIRTaskList(IRModel& ir_model, IRNet& ir_net, std::vector<IRTask>& ir_task_list,
                      std::vector<Segment<LayerCoord>>& routing_segment_list);
  std::vector<Segment<PlanarCoord>> getPlanarTopoListByFlute(std::vector<PlanarCoord>& planar_coord_list);
  void routeIRTask(IRModel& ir_model, IRTask* ir_task);
  void initSingleTask(IRModel& ir_model, IRTask* ir_task);
  bool isConnectedAllEnd(IRModel& ir_model);
  void routeSinglePath(IRModel& ir_model);
  void initPathHead(IRModel& ir_model);
  bool searchEnded(IRModel& ir_model);
  void expandSearching(IRModel& ir_model);
  void resetPathHead(IRModel& ir_model);
  bool isRoutingFailed(IRModel& ir_model);
  void resetSinglePath(IRModel& ir_model);
  void updatePathResult(IRModel& ir_model);
  std::vector<Segment<LayerCoord>> getRoutingSegmentListByNode(IRNode* node);
  void updateDirectionSet(IRModel& ir_model);
  void resetStartAndEnd(IRModel& ir_model);
  void updateTaskResult(IRModel& ir_model);
  std::vector<Segment<LayerCoord>> getRoutingSegmentList(IRModel& ir_model);
  void resetSingleTask(IRModel& ir_model);
  void pushToOpenList(IRModel& ir_model, IRNode* curr_node);
  IRNode* popFromOpenList(IRModel& ir_model);
  double getKnowCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node);
  double getNodeCost(IRModel& ir_model, IRNode* curr_node, Orientation orientation);
  double getKnowWireCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node);
  double getKnowCornerCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node);
  double getKnowViaCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node);
  double getEstimateCostToEnd(IRModel& ir_model, IRNode* curr_node);
  double getEstimateCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node);
  double getEstimateWireCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node);
  double getEstimateCornerCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node);
  double getEstimateViaCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node);
  MTree<LayerCoord> getCoordTree(IRNet& ir_net, std::vector<Segment<LayerCoord>>& routing_segment_list);
  void updateDemand(IRModel& ir_model, IRNet& ir_net, MTree<LayerCoord>& coord_tree);
  void updateIRModel(IRModel& ir_model);

#if 1  // debug
  void debugCheckIRModel(IRModel& ir_model);
  void debugOutputGuide(IRModel& ir_model);
#endif

#if 1  // exhibit
  void updateSummary(IRModel& ir_model);
  void printSummary(IRModel& ir_model);
  void writeDemandCSV(IRModel& ir_model);
  void writeOverflowCSV(IRModel& ir_model);
#endif
};

}  // namespace irt
