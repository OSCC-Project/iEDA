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
#include "PRModel.hpp"
#include "flute3/flute.h"

namespace irt {

#define RTPR (irt::PlanarRouter::getInst())

class PlanarRouter
{
 public:
  static void initInst();
  static PlanarRouter& getInst();
  static void destroyInst();
  // function
  void route();

 private:
  // self
  static PlanarRouter* _pr_instance;

  PlanarRouter() { Flute::readLUT(); }
  PlanarRouter(const PlanarRouter& other) = delete;
  PlanarRouter(PlanarRouter&& other) = delete;
  ~PlanarRouter() { Flute::deleteLUT(); }
  PlanarRouter& operator=(const PlanarRouter& other) = delete;
  PlanarRouter& operator=(PlanarRouter&& other) = delete;
  // function
  PRModel initPRModel();
  std::vector<PRNet> convertToPRNetList(std::vector<Net>& net_list);
  PRNet convertToPRNet(Net& net);
  void setPRParameter(PRModel& pr_model);
  void buildNodeMap(PRModel& pr_model);
  void buildPRNodeNeighbor(PRModel& pr_model);
  void buildOrientSupply(PRModel& pr_model);
  void sortPRModel(PRModel& pr_model);
  bool sortByMultiLevel(PRModel& pr_model, int32_t net_idx1, int32_t net_idx2);
  SortStatus sortByClockPriority(PRNet& net1, PRNet& net2);
  SortStatus sortByRoutingAreaASC(PRNet& net1, PRNet& net2);
  SortStatus sortByLengthWidthRatioDESC(PRNet& net1, PRNet& net2);
  SortStatus sortByPinNumDESC(PRNet& net1, PRNet& net2);
  void routePRModel(PRModel& pr_model);
  void routePRNet(PRModel& pr_model, PRNet& pr_net);
  std::vector<Segment<PlanarCoord>> getPlanarTopoListByFlute(PRNet& pr_net);
  std::vector<Segment<PlanarCoord>> getRoutingSegmentList(PRModel& pr_model, Segment<PlanarCoord>& planar_topo);
  std::vector<std::vector<PlanarCoord>> getInflectionListListByStraight(Segment<PlanarCoord>& planar_topo);
  std::vector<std::vector<PlanarCoord>> getInflectionListListByLPattern(Segment<PlanarCoord>& planar_topo);
  std::vector<std::vector<PlanarCoord>> getInflectionListListByZPattern(Segment<PlanarCoord>& planar_topo);
  std::vector<int32_t> getMidIndexList(int32_t start_idx, int32_t end_idx);
  std::vector<std::vector<PlanarCoord>> getInflectionListListByInner3Bends(Segment<PlanarCoord>& planar_topo);
  double getNodeCost(PRModel& pr_model, std::vector<Segment<PlanarCoord>>& routing_segment_list);
  MTree<PlanarCoord> getCoordTree(PRNet& pr_net, std::vector<Segment<PlanarCoord>>& routing_segment_list);
  void updateDemand(PRModel& pr_model, PRNet& pr_net, MTree<PlanarCoord>& coord_tree);
  void updatePRModel(PRModel& pr_model);
};

}  // namespace irt
