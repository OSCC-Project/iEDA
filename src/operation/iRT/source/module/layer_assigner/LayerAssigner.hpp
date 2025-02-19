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
#include "LAModel.hpp"
#include "RTHeader.hpp"

namespace irt {

#define RTLA (irt::LayerAssigner::getInst())

class LayerAssigner
{
 public:
  static void initInst();
  static LayerAssigner& getInst();
  static void destroyInst();
  // function
  void route();

 private:
  // self
  static LayerAssigner* _la_instance;

  LayerAssigner() = default;
  LayerAssigner(const LayerAssigner& other) = delete;
  LayerAssigner(LayerAssigner&& other) = delete;
  ~LayerAssigner() = default;
  LayerAssigner& operator=(const LayerAssigner& other) = delete;
  LayerAssigner& operator=(LayerAssigner&& other) = delete;
  // function
  LAModel initLAModel();
  std::vector<LANet> convertToLANetList(std::vector<Net>& net_list);
  LANet convertToLANet(Net& net);
  void initLATaskList(LAModel& la_model);
  void setLAComParam(LAModel& la_model);
  void buildLayerNodeMap(LAModel& la_model);
  void buildLANodeNeighbor(LAModel& la_model);
  void buildOrientSupply(LAModel& la_model);
  void buildTopoTree(LAModel& la_model);
  void routeLAModel(LAModel& la_model);
  void routeLANet(LAModel& la_model, LANet* la_net);
  void makeLATopoList(LAModel& la_model, LANet* la_net, std::vector<LATopo>& la_topo_list, std::vector<Segment<LayerCoord>>& routing_segment_list);
  void routeLATopo(LAModel& la_model, LATopo* la_topo);
  void initSingleTask(LAModel& la_model, LATopo* la_topo);
  bool isConnectedAllEnd(LAModel& la_model);
  void routeSinglePath(LAModel& la_model);
  void initPathHead(LAModel& la_model);
  bool searchEnded(LAModel& la_model);
  void expandSearching(LAModel& la_model);
  void resetPathHead(LAModel& la_model);
  void updatePathResult(LAModel& la_model);
  std::vector<Segment<LayerCoord>> getRoutingSegmentListByNode(LANode* node);
  void resetStartAndEnd(LAModel& la_model);
  void resetSinglePath(LAModel& la_model);
  void updateTaskResult(LAModel& la_model);
  std::vector<Segment<LayerCoord>> getRoutingSegmentList(LAModel& la_model);
  void resetSingleTask(LAModel& la_model);
  void pushToOpenList(LAModel& la_model, LANode* curr_node);
  LANode* popFromOpenList(LAModel& la_model);
  double getKnowCost(LAModel& la_model, LANode* start_node, LANode* end_node);
  double getNodeCost(LAModel& la_model, LANode* curr_node, Orientation orientation);
  double getKnowWireCost(LAModel& la_model, LANode* start_node, LANode* end_node);
  double getKnowViaCost(LAModel& la_model, LANode* start_node, LANode* end_node);
  double getEstimateCostToEnd(LAModel& la_model, LANode* curr_node);
  double getEstimateCost(LAModel& la_model, LANode* start_node, LANode* end_node);
  double getEstimateWireCost(LAModel& la_model, LANode* start_node, LANode* end_node);
  double getEstimateViaCost(LAModel& la_model, LANode* start_node, LANode* end_node);
  MTree<LayerCoord> getCoordTree(LANet* la_net, std::vector<Segment<LayerCoord>>& routing_segment_list);
  void uploadNetResult(LANet* la_net, MTree<LayerCoord>& coord_tree);

#if 1  // update env
  void updateDemandToGraph(LAModel& la_model, ChangeType change_type, MTree<LayerCoord>& coord_tree);
#endif

#if 1  // exhibit
  void updateSummary(LAModel& la_model);
  void printSummary(LAModel& la_model);
  void outputGuide(LAModel& la_model);
  void outputDemandCSV(LAModel& la_model);
  void outputOverflowCSV(LAModel& la_model);
#endif

#if 1  // debug
  void debugCheckLAModel(LAModel& la_model);
#endif
};

}  // namespace irt
