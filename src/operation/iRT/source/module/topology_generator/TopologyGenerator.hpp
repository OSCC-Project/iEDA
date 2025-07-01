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
#include "Monitor.hpp"
#include "TGModel.hpp"

namespace irt {

#define RTTG (irt::TopologyGenerator::getInst())

class TopologyGenerator
{
 public:
  static void initInst();
  static TopologyGenerator& getInst();
  static void destroyInst();
  // function
  void generate();

 private:
  // self
  static TopologyGenerator* _tg_instance;

  TopologyGenerator() = default;
  TopologyGenerator(const TopologyGenerator& other) = delete;
  TopologyGenerator(TopologyGenerator&& other) = delete;
  ~TopologyGenerator() = default;
  TopologyGenerator& operator=(const TopologyGenerator& other) = delete;
  TopologyGenerator& operator=(TopologyGenerator&& other) = delete;
  // function
  TGModel initTGModel();
  std::vector<TGNet> convertToTGNetList(std::vector<Net>& net_list);
  TGNet convertToTGNet(Net& net);
  void setTGComParam(TGModel& tg_model);
  void initTGTaskList(TGModel& tg_model);
  void buildTGNodeMap(TGModel& tg_model);
  void buildTGNodeNeighbor(TGModel& tg_model);
  void buildOrientSupply(TGModel& tg_model);
  void generateTGModel(TGModel& tg_model);
  void routeTGTask(TGModel& tg_model, TGNet* tg_net);
  void initSingleTask(TGModel& tg_model, TGNet* tg_net);
  std::vector<Segment<PlanarCoord>> getPlanarTopoList(TGModel& tg_model);
  std::vector<Segment<PlanarCoord>> getRoutingSegmentList(TGModel& tg_model, Segment<PlanarCoord>& planar_topo);
  std::vector<Segment<PlanarCoord>> getRoutingSegmentListByStraight(TGModel& tg_model, Segment<PlanarCoord>& planar_topo);
  std::vector<Segment<PlanarCoord>> getRoutingSegmentListByLPattern(TGModel& tg_model, Segment<PlanarCoord>& planar_topo);
  double getNodeCost(TGModel& tg_model, std::vector<Segment<PlanarCoord>>& routing_segment_list);
  MTree<PlanarCoord> getCoordTree(TGModel& tg_model, std::vector<Segment<PlanarCoord>>& routing_segment_list);
  void uploadNetResult(TGModel& tg_model, MTree<PlanarCoord>& coord_tree);
  void resetSingleTask(TGModel& tg_model);

#if 1  // update env
  void updateDemandToGraph(TGModel& tg_model, ChangeType change_type, MTree<PlanarCoord>& coord_tree);
#endif

#if 1  // exhibit
  void updateSummary(TGModel& tg_model);
  void printSummary(TGModel& tg_model);
  void outputGuide(TGModel& tg_model);
  void outputNetCSV(TGModel& tg_model);
  void outputOverflowCSV(TGModel& tg_model);
  void outputNetJson(TGModel& tg_model);
  void outputOverflowJson(TGModel& tg_model);
#endif

#if 1  // debug
  void debugCheckTGModel(TGModel& tg_model);
#endif
};

}  // namespace irt
