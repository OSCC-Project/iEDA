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
#include "RTU.hpp"
#include "flute3/flute.h"

namespace irt {

#define GR_INST (irt::GlobalRouter::getInst())

class GlobalRouter
{
 public:
  static void initInst();
  static GlobalRouter& getInst();
  static void destroyInst();
  // function
  void route(std::vector<Net>& net_list);

 private:
  // self
  static GlobalRouter* _gr_instance;

  GlobalRouter() { Flute::readLUT(); }
  GlobalRouter(const GlobalRouter& other) = delete;
  GlobalRouter(GlobalRouter&& other) = delete;
  ~GlobalRouter() = default;
  GlobalRouter& operator=(const GlobalRouter& other) = delete;
  GlobalRouter& operator=(GlobalRouter&& other) = delete;
  // function
  void routeNetList(std::vector<Net>& net_list);

#if 1  // init
  GRModel init(std::vector<Net>& net_list);
  GRModel initGRModel(std::vector<Net>& net_list);
  std::vector<GRNet> convertToGRNetList(std::vector<Net>& net_list);
  GRNet convertToGRNet(Net& net);
  void buildGRModel(GRModel& gr_model);
  void buildNeighborMap(GRModel& gr_model);
  void updateNetRectMap(GRModel& gr_model);
  void addRectToEnv(GRModel& gr_model, irt_int net_idx, LayerRect real_rect);
  void cutBlockageList(GRModel& gr_model);
  void updateWholeDemand(GRModel& gr_model);
  void updateNetDemandMap(GRModel& gr_model);
  void updateNodeSupply(GRModel& gr_model);
  std::vector<PlanarRect> getWireList(GRNode& gr_node, RoutingLayer& routing_layer);
  void buildAccessMap(GRModel& gr_model);
  void checkGRModel(GRModel& gr_model);
#endif

#if 1  // iterative
  void iterative(GRModel& gr_model);
  void sortGRModel(GRModel& gr_model);
  bool sortByMultiLevel(GRNet& net1, GRNet& net2);
  SortStatus sortByRoutingAreaASC(GRNet& net1, GRNet& net2);
  SortStatus sortByLengthWidthRatioDESC(GRNet& net1, GRNet& net2);
  SortStatus sortByPinNumDESC(GRNet& net1, GRNet& net2);
  void resetGRModel(GRModel& gr_model);
  void routeGRModel(GRModel& gr_model);
  void routeGRNet(GRModel& gr_model, GRNet& gr_net);
  void initSingleNet(GRModel& gr_model, GRNet& gr_net);
  std::vector<Segment<PlanarCoord>> getPlanarTopoListByFlute(std::vector<PlanarCoord>& planar_coord_list);
  void initSinglePath(GRModel& gr_model, std::pair<std::vector<GRNode*>, std::vector<GRNode*>>& node_topo);
  void routeByStrategy(GRModel& gr_model, GRRouteStrategy gr_route_strategy);
  void routeSinglePath(GRModel& gr_model);
  void initPathHead(GRModel& gr_model);
  bool searchEnded(GRModel& gr_model);
  void expandSearching(GRModel& gr_model);
  bool passCheckingSegment(GRModel& gr_model, GRNode* start_node, GRNode* end_node);
  bool replaceParentNode(GRModel& gr_model, GRNode* parent_node, GRNode* child_node);
  void resetPathHead(GRModel& gr_model);
  bool isRoutingFailed(GRModel& gr_model);
  void resetSinglePath(GRModel& gr_model);
  void updatePathResult(GRModel& gr_model);
  void updateDirectionSet(GRModel& gr_model);
  void updateNetResult(GRModel& gr_model, GRNet& gr_net);
  void optNodeSegmentListDueFlute(GRModel& gr_model, GRNet& gr_net);
  void resetSingleNet(GRModel& gr_model);
  void pushToOpenList(GRModel& gr_model, GRNode* curr_node);
  GRNode* popFromOpenList(GRModel& gr_model);
  double getKnowCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node);
  double getJointCost(GRModel& gr_model, GRNode* curr_node, Orientation orientation);
  double getKnowWireCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node);
  double getKnowCornerCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node);
  double getViaCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node);
  double getEstimateCostToEnd(GRModel& gr_model, GRNode* curr_node);
  double getEstimateCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node);
  double getEstimateWireCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node);
  double getEstimateCornerCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node);
  void processGRModel(GRModel& gr_model);
  void initRoutingResult(GRNet& gr_net);
  RTNode convertToRTNode(LayerCoord& coord, std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC>& key_coord_pin_map);
  void buildRoutingResult(GRNet& gr_net);
  void buildDRNode(TNode<RTNode>* parent_node, TNode<RTNode>* child_node);
  void buildTANode(TNode<RTNode>* parent_node, TNode<RTNode>* child_node);
  void reportGRModel(GRModel& gr_model);
  void countGRModel(GRModel& gr_model);
  void reportTable(GRModel& gr_model);
#endif

#if 1  // update
  void update(GRModel& gr_model);
#endif

#if 1  // plot gr_model
  void plotCongestionMap(GRModel& gr_model, irt_int iter);
  void plotGRModel(GRModel& gr_model, irt_int curr_net_idx);
#endif
};

}  // namespace irt
