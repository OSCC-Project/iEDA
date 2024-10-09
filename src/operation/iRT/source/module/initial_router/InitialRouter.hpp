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

namespace irt {

#define RTIR (irt::InitialRouter::getInst())

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

  InitialRouter() = default;
  InitialRouter(const InitialRouter& other) = delete;
  InitialRouter(InitialRouter&& other) = delete;
  ~InitialRouter() = default;
  InitialRouter& operator=(const InitialRouter& other) = delete;
  InitialRouter& operator=(InitialRouter&& other) = delete;
  // function
  IRModel initIRModel();
  std::vector<IRNet> convertToIRNetList(std::vector<Net>& net_list);
  IRNet convertToIRNet(Net& net);
  void initIRTaskList(IRModel& ir_model);
  void setIRParameter(IRModel& ir_model);
  void buildLayerNodeMap(IRModel& ir_model);
  void buildIRNodeNeighbor(IRModel& ir_model);
  void buildOrientSupply(IRModel& ir_model);
  void buildTopoTree(IRModel& ir_model);
  void routeIRModel(IRModel& ir_model);
  void routeIRNet(IRModel& ir_model, IRNet* ir_net);
  void makeIRTopoList(IRModel& ir_model, IRNet* ir_net, std::vector<IRTopo>& ir_topo_list,
                      std::vector<Segment<LayerCoord>>& routing_segment_list);
  void routeIRTopo(IRModel& ir_model, IRTopo* ir_topo);
  void initSingleTask(IRModel& ir_model, IRTopo* ir_topo);
  bool isConnectedAllEnd(IRModel& ir_model);
  void routeSinglePath(IRModel& ir_model);
  void initPathHead(IRModel& ir_model);
  bool searchEnded(IRModel& ir_model);
  void expandSearching(IRModel& ir_model);
  void resetPathHead(IRModel& ir_model);
  void updatePathResult(IRModel& ir_model);
  std::vector<Segment<LayerCoord>> getRoutingSegmentListByNode(IRNode* node);
  void resetStartAndEnd(IRModel& ir_model);
  void resetSinglePath(IRModel& ir_model);
  void updateTaskResult(IRModel& ir_model);
  std::vector<Segment<LayerCoord>> getRoutingSegmentList(IRModel& ir_model);
  void resetSingleTask(IRModel& ir_model);
  void pushToOpenList(IRModel& ir_model, IRNode* curr_node);
  IRNode* popFromOpenList(IRModel& ir_model);
  double getKnowCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node);
  double getNodeCost(IRModel& ir_model, IRNode* curr_node, Orientation orientation);
  double getKnowWireCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node);
  double getKnowViaCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node);
  double getEstimateCostToEnd(IRModel& ir_model, IRNode* curr_node);
  double getEstimateCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node);
  double getEstimateWireCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node);
  double getEstimateViaCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node);
  MTree<LayerCoord> getCoordTree(IRNet* ir_net, std::vector<Segment<LayerCoord>>& routing_segment_list);
  void updateDemand(IRModel& ir_model, IRNet* ir_net, MTree<LayerCoord>& coord_tree);
  void uploadNetResult(IRNet* ir_net, MTree<LayerCoord>& coord_tree);

#if 1  // exhibit
  void updateSummary(IRModel& ir_model);
  void printSummary(IRModel& ir_model);
  void outputDemandCSV(IRModel& ir_model);
  void outputOverflowCSV(IRModel& ir_model);
#endif

#if 1  // debug
  void debugCheckIRModel(IRModel& ir_model);
  void outputGuide(IRModel& ir_model);
#endif
};

}  // namespace irt
