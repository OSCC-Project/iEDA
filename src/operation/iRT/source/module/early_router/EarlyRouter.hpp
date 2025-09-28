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
#include "ERModel.hpp"
#include "RTHeader.hpp"

namespace irt {

#define RTER (irt::EarlyRouter::getInst())

class EarlyRouter
{
 public:
  static void initInst();
  static EarlyRouter& getInst();
  static void destroyInst();
  // function
  void route();

 private:
  // self
  static EarlyRouter* _er_instance;

  EarlyRouter() = default;
  EarlyRouter(const EarlyRouter& other) = delete;
  EarlyRouter(EarlyRouter&& other) = delete;
  ~EarlyRouter() = default;
  EarlyRouter& operator=(const EarlyRouter& other) = delete;
  EarlyRouter& operator=(EarlyRouter&& other) = delete;
  // function
  ERModel initERModel();
  std::vector<ERNet> convertToERNetList(std::vector<Net>& net_list);
  ERNet convertToERNet(Net& net);
  void setERComParam(ERModel& er_model);
  void generateAccessPoint(ERModel& er_model);
  void initERTaskList(ERModel& er_model);
  void buildPlanarNodeMap(ERModel& er_model);
  void buildPlanarNodeNeighbor(ERModel& er_model);
  void buildPlanarOrientSupply(ERModel& er_model);
  void generateTopoTree(ERModel& er_model);
  void routePlanarNet(ERModel& er_model, ERNet* er_net);
  std::vector<Segment<PlanarCoord>> getPlanarTopoList(ERModel& er_model, ERNet* er_net);
  std::vector<Segment<PlanarCoord>> getRoutingSegmentList(ERModel& er_model, Segment<PlanarCoord>& planar_topo);
  std::vector<Segment<PlanarCoord>> getRoutingSegmentListByStraight(ERModel& er_model, Segment<PlanarCoord>& planar_topo);
  std::vector<Segment<PlanarCoord>> getRoutingSegmentListByLPattern(ERModel& er_model, Segment<PlanarCoord>& planar_topo);
  double getPlanarNodeCost(ERModel& er_model, std::vector<Segment<PlanarCoord>>& routing_segment_list);
  MTree<LayerCoord> getPlanarCoordTree(ERNet* er_net, std::vector<Segment<PlanarCoord>>& planar_routing_segment_list);
  void buildLayerNodeMap(ERModel& er_model);
  void buildLayerNodeNeighbor(ERModel& er_model);
  void buildLayerOrientSupply(ERModel& er_model);
  void generateGlobalTree(ERModel& er_model);
  void routeLayerNet(ERModel& er_model, ERNet* er_net);
  void makeERTopoList(ERModel& er_model, ERNet* er_net, std::vector<ERTopo>& er_topo_list, std::vector<Segment<LayerCoord>>& routing_segment_list);
  void routeERTopo(ERModel& er_model, ERTopo* er_topo);
  void initSingleTask(ERModel& er_model, ERTopo* er_topo);
  bool isConnectedAllEnd(ERModel& er_model);
  void routeSinglePath(ERModel& er_model);
  void initPathHead(ERModel& er_model);
  bool searchEnded(ERModel& er_model);
  void expandSearching(ERModel& er_model);
  void resetPathHead(ERModel& er_model);
  void updatePathResult(ERModel& er_model);
  std::vector<Segment<LayerCoord>> getRoutingSegmentListByNode(ERNode* node);
  void resetStartAndEnd(ERModel& er_model);
  void resetSinglePath(ERModel& er_model);
  void updateTaskResult(ERModel& er_model);
  std::vector<Segment<LayerCoord>> getRoutingSegmentList(ERModel& er_model);
  void resetSingleTask(ERModel& er_model);
  void pushToOpenList(ERModel& er_model, ERNode* curr_node);
  ERNode* popFromOpenList(ERModel& er_model);
  double getKnownCost(ERModel& er_model, ERNode* start_node, ERNode* end_node);
  double getNodeCost(ERModel& er_model, ERNode* curr_node, Orientation orientation);
  double getKnownWireCost(ERModel& er_model, ERNode* start_node, ERNode* end_node);
  double getKnownViaCost(ERModel& er_model, ERNode* start_node, ERNode* end_node);
  double getEstimateCostToEnd(ERModel& er_model, ERNode* curr_node);
  double getEstimateCost(ERModel& er_model, ERNode* start_node, ERNode* end_node);
  double getEstimateWireCost(ERModel& er_model, ERNode* start_node, ERNode* end_node);
  double getEstimateViaCost(ERModel& er_model, ERNode* start_node, ERNode* end_node);
  MTree<LayerCoord> getLayerCoordTree(ERNet* er_net, std::vector<Segment<LayerCoord>>& routing_segment_list);
  void uploadNetResult(ERNet* er_net, MTree<LayerCoord>& coord_tree);

#if 1  // update env
  void updatePlanarDemandToGraph(ERModel& er_model, ChangeType change_type, MTree<LayerCoord>& coord_tree);
  void updateLayerDemandToGraph(ERModel& er_model, ChangeType change_type, MTree<LayerCoord>& coord_tree);
#endif

#if 1  // exhibit
  void updateSummary(ERModel& er_model);
  void printSummary(ERModel& er_model);
  void outputGCellCSV(ERModel& er_model);
  void outputGuide(ERModel& er_model);
  void outputDemandCSV(ERModel& er_model);
  void outputOverflowCSV(ERModel& er_model);
#endif

#if 1  // debug
  void debugCheckPlanarNodeMap(ERModel& er_model);
  void debugCheckLayerNodeMap(ERModel& er_model);
#endif
};

}  // namespace irt
