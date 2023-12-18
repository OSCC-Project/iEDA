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
#include "DRCChecker.hpp"
#include "DRCShape.hpp"
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
  void updateBlockageMap(GRModel& gr_model);
  void updateNetShapeMap(GRModel& gr_model);
  void updateReservedViaMap(GRModel& gr_model);
  void updateWholeDemand(GRModel& gr_model);
  void updateNetDemandMap(GRModel& gr_model);
  void updateNodeResourceSupply(GRModel& gr_model);
  std::vector<PlanarRect> getCrossingWireList(PlanarRect& base_rect, RoutingLayer& routing_layer);
  void updateNodeAccessSupply(GRModel& gr_model);
  bool isAccess(GRNode& pre_node, GRNode& curr_node, PlanarRect& wire);
  void makeRoutingState(GRModel& gr_model);
  void checkGRModel(GRModel& gr_model);
  void writePYScript();
#endif

#if 1  // iterative
  void iterative(GRModel& gr_model);
  void resetGRModel(GRModel& gr_model);
  void sortGRModel(GRModel& gr_model);
  bool sortByMultiLevel(GRModel& gr_model, irt_int net_idx1, irt_int net_idx2);
  SortStatus sortByClockPriority(GRNet& net1, GRNet& net2);
  SortStatus sortByRoutingAreaASC(GRNet& net1, GRNet& net2);
  SortStatus sortByLengthWidthRatioDESC(GRNet& net1, GRNet& net2);
  SortStatus sortByPinNumDESC(GRNet& net1, GRNet& net2);
  void updateRipupGrid(GRModel& gr_model);
  void addHistoryCost(GRModel& gr_model);
  void ripupPassedNet(GRModel& gr_model);
  void routeGRModel(GRModel& gr_model);
  void routeGRNet(GRModel& gr_model, GRNet& gr_net);
  void outputGRDataset(GRModel& gr_model, GRNet& gr_net);
  void initSingleNet(GRModel& gr_model, GRNet& gr_net);
  std::vector<Segment<PlanarCoord>> getPlanarTopoListByFlute(std::vector<PlanarCoord>& planar_coord_list);
  void initSingleTask(GRModel& gr_model, GRTask& gr_task);
  bool isConnectedAllEnd(GRModel& gr_model);
  void routeSinglePath(GRModel& gr_model);
  void initPathHead(GRModel& gr_model);
  bool searchEnded(GRModel& gr_model);
  void expandSearching(GRModel& gr_model);
  void resetPathHead(GRModel& gr_model);
  bool isRoutingFailed(GRModel& gr_model);
  void resetSinglePath(GRModel& gr_model);
  void updatePathResult(GRModel& gr_model);
  void updateDirectionSet(GRModel& gr_model);
  void resetStartAndEnd(GRModel& gr_model);
  void resetSingleTask(GRModel& gr_model);
  void updateNetResult(GRModel& gr_model, GRNet& gr_net);
  void updateDemand(GRModel& gr_model, GRNet& gr_net, ChangeType change_type);
  void updateRoutingTree(GRModel& gr_model, GRNet& gr_net);
  void resetSingleNet(GRModel& gr_model);
  void pushToOpenList(GRModel& gr_model, GRNode* curr_node);
  GRNode* popFromOpenList(GRModel& gr_model);
  double getKnowCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node);
  double getNodeCost(GRModel& gr_model, GRNode* curr_node, Orientation orientation);
  double getKnowWireCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node);
  double getKnowCornerCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node);
  double getKnowViaCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node);
  double getEstimateCostToEnd(GRModel& gr_model, GRNode* curr_node);
  double getEstimateCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node);
  double getEstimateWireCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node);
  double getEstimateCornerCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node);
  double getEstimateViaCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node);
  void processGRModel(GRModel& gr_model);
  void buildRoutingResult(GRNet& gr_net);
  GuideSegNode convertToGuideSegNode(LayerCoord& coord, std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC>& key_coord_pin_map);
  void buildDRGuideNode(TNode<GuideSegNode>* parent_node, TNode<GuideSegNode>* child_node);
  void buildTAGuideNode(TNode<GuideSegNode>* parent_node, TNode<GuideSegNode>* child_node);
  void countGRModel(GRModel& gr_model);
  void reportGRModel(GRModel& gr_model);
  bool stopGRModel(GRModel& gr_model);
#endif

#if 1  // update
  void update(GRModel& gr_model);
  void outputGuide(GRModel& gr_model);
#endif

#if 1  // update env
  std::vector<DRCShape> getDRCShapeList(irt_int gr_net_idx, std::vector<Segment<LayerCoord>>& segment_list);
  void updateRectToUnit(GRModel& gr_model, ChangeType change_type, GRSourceType gr_source_type, DRCShape drc_shape);
#endif

#if 1  // plot gr_model
  void outputCongestionMap(GRModel& gr_model);
  void plotGRModel(GRModel& gr_model, irt_int curr_net_idx = -1);
#endif
};

}  // namespace irt
