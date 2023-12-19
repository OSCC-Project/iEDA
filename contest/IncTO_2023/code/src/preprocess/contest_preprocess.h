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
/**
 * @File Name: contest_evaluation.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2023-09-15
 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <string>

#include "astar.hh"
#include "builder.h"
#include "contest_db.h"
#include "def_service.h"
#include "flute3/flute.h"
#include "gridmap.hh"
#include "lef_service.h"

namespace ieda_contest {
class ContestDataManager;

class ContestPreprocess
{
 public:
  ContestPreprocess(ContestDataManager* data_manager);
  ~ContestPreprocess() = default;

  void doPreprocess();

 private:
  ContestDataManager* _data_manager = nullptr;

  gridmap::Map<astar::Node<3>> _grid_map;
  astar::PathFinder<3> _pathfinder;

  void makeLayerInfo();
  void makeGCellInfo();

#if 1  // placer
  void place();
#endif

#if 1  // router
  void route();
  void makeNetList();
  ContestCoord getGCellCoord(const ContestCoord& coord);
  void routeGuide();
  void createMap();
  void routeNet(ContestNet& contest_net);
  std::vector<ContestCoord> makeKeyCoordList(ContestNet& contest_net);
  std::vector<ContestSegment> getTopoListByGreedy(std::vector<ContestCoord>& coord_list);
  std::vector<ContestSegment> getTopoListByFlute(std::vector<ContestCoord>& coord_list);
  std::vector<ContestSegment> getRoutingSegmentList(ContestSegment& topo, ContestNet& contest_net);
  bool connectivityCheckPassed();
  bool overflowCheckPassed();
  void updateGuideList();
  std::vector<ContestGuide> getGuide(ContestSegment& routing_segment);
  void outputGCellGrid();
#endif
};

}  // namespace ieda_contest