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
#include "ERCandidate.hpp"
#include "ERModel.hpp"
#include "ERPackage.hpp"
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
  void initAccessPointList(ERModel& er_model);
  std::vector<LayerCoord> getAccessCoordList(ERModel& er_model, std::vector<EXTLayerRect>& pin_shape_list);
  void uniformSampleCoordList(ERModel& er_model, std::vector<LayerCoord>& layer_coord_list);
  void buildConflictList(ERModel& er_model);
  std::vector<std::pair<ERPin*, std::set<ERPin*>>> getPinConlictMap(ERModel& er_model);
  bool hasConflict(AccessPoint& curr_access_point, AccessPoint& gcell_access_point);
  bool hasConflict(LayerCoord layer_coord1, LayerCoord layer_coord2);
  void eliminateConflict(ERModel& er_model);
  std::vector<ERConflictPoint> getBestPointList(ERConflictGroup& er_conflict_group);
  void uploadAccessPoint(ERModel& er_model);
  void uploadAccessPatch(ERModel& er_model);
  void buildSupplySchedule(ERModel& er_model);
  void analyzeSupply(ERModel& er_model);
  EXTLayerRect getSearchRect(LayerCoord& first_coord, LayerCoord& second_coord);
  std::vector<LayerRect> getCrossingWireList(EXTLayerRect& search_rect);
  bool isAccess(LayerRect& wire, std::vector<PlanarRect>& obs_rect_list);
  void buildIgnoreNet(ERModel& er_model);
  void analyzeDemandUnit(ERModel& er_model);
  void initERTaskList(ERModel& er_model);
  void buildPlanarNodeMap(ERModel& er_model);
  void buildPlanarNodeNeighbor(ERModel& er_model);
  void buildPlanarOrientSupply(ERModel& er_model);
  void generateTopology(ERModel& er_model);
  void generateERTask(ERModel& er_model, ERNet* er_task);
  void initSinglePlanarTask(ERModel& er_model, ERNet* er_task);
  std::vector<Segment<PlanarCoord>> getPlanarRoutingSegmentList(ERModel& er_model);
  std::vector<Segment<PlanarCoord>> getPlanarTopoList(ERModel& er_model);
  std::vector<std::vector<Segment<PlanarCoord>>> getRoutingSegmentListList(ERModel& er_model, Segment<PlanarCoord>& planar_topo);
  std::vector<std::vector<Segment<PlanarCoord>>> getRoutingSegmentListByStraight(ERModel& er_model, Segment<PlanarCoord>& planar_topo);
  std::vector<std::vector<Segment<PlanarCoord>>> getRoutingSegmentListByLPattern(ERModel& er_model, Segment<PlanarCoord>& planar_topo);
  std::vector<std::vector<Segment<PlanarCoord>>> getRoutingSegmentListByZPattern(ERModel& er_model, Segment<PlanarCoord>& planar_topo);
  std::vector<int32_t> getMidIndexList(int32_t first_idx, int32_t second_idx);
  std::vector<std::vector<Segment<PlanarCoord>>> getRoutingSegmentListByUPattern(ERModel& er_model, Segment<PlanarCoord>& planar_topo);
  std::vector<std::vector<Segment<PlanarCoord>>> getRoutingSegmentListByInner3Bends(ERModel& er_model, Segment<PlanarCoord>& planar_topo);
  std::vector<std::vector<Segment<PlanarCoord>>> getRoutingSegmentListByOuter3Bends(ERModel& er_model, Segment<PlanarCoord>& planar_topo);
  void updateERCandidate(ERModel& er_model, ERCandidate& er_candidate);
  MTree<PlanarCoord> getCoordTree(ERModel& er_model, std::vector<Segment<PlanarCoord>>& routing_segment_list);
  void resetSinglePlanarTask(ERModel& er_model);
  void buildLayerNodeMap(ERModel& er_model);
  void buildLayerNodeNeighbor(ERModel& er_model);
  void buildLayerOrientSupply(ERModel& er_model);
  void assignLayer(ERModel& er_model);
  void assignERTask(ERModel& er_model, ERNet* er_task);
  void initSingleTask(ERModel& er_model, ERNet* er_task);
  bool needRouting(ERModel& er_model);
  void spiltPlaneTree(ERModel& er_model);
  void insertMidPoint(ERModel& er_model, TNode<PlanarCoord>* planar_node, TNode<PlanarCoord>* child_node);
  void buildPillarTree(ERModel& er_model);
  ERPillar convertERPillar(PlanarCoord& planar_coord, std::map<PlanarCoord, std::set<int32_t>, CmpPlanarCoordByXASC>& coord_pin_layer_map);
  void assignPillarTree(ERModel& er_model);
  void assignForward(ERModel& er_model);
  std::vector<int32_t> getCandidateLayerList(ERModel& er_model, ERPackage& er_package);
  double getFullViaCost(ERModel& er_model, std::set<int32_t>& layer_idx_set, int32_t candidate_layer_idx);
  void buildLayerCost(ERModel& er_model, ERPackage& er_package);
  std::pair<int32_t, double> getParentPillarCost(ERModel& er_model, ERPackage& er_package, int32_t candidate_layer_idx);
  double getExtraViaCost(ERModel& er_model, std::set<int32_t>& layer_idx_set, int32_t candidate_layer_idx);
  double getSegmentCost(ERModel& er_model, ERPackage& er_package, int32_t candidate_layer_idx);
  double getChildPillarCost(ERModel& er_model, ERPackage& er_package, int32_t candidate_layer_idx);
  void assignBackward(ERModel& er_model);
  int32_t getBestLayerBySelf(TNode<ERPillar>* pillar_node);
  int32_t getBestLayerByChild(TNode<ERPillar>* parent_pillar_node);
  void buildLayerTree(ERModel& er_model, ERNet* er_task);
  std::vector<Segment<LayerCoord>> getLayerRoutingSegmentList(ERModel& er_model);
  MTree<LayerCoord> getCoordTree(ERModel& er_model, std::vector<Segment<LayerCoord>>& routing_segment_list);
  void resetSingleLayerTask(ERModel& er_model);
  void outputResult(ERModel& er_model);
  void outputGCellCSV(ERModel& er_model);
  void outputLayerSupplyCSV(ERModel& er_model);
  void outputLayerGuide(ERModel& er_model);
  void outputLayerNetCSV(ERModel& er_model);
  void outputLayerOverflowCSV(ERModel& er_model);
  void cleanTempResult(ERModel& er_model);

#if 1  // update env
  void updateDemandToGraph(ERModel& er_model, ChangeType change_type, MTree<PlanarCoord>& coord_tree);
  void updateDemandToGraph(ERModel& er_model, ChangeType change_type, MTree<LayerCoord>& coord_tree);
#endif

#if 1  // exhibit
  void updateSummary(ERModel& er_model);
  void printSummary(ERModel& er_model);
#endif

#if 1  // debug
  void debugPlotERModel(ERModel& er_model, std::string flag);
#endif
};

}  // namespace irt
