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
#ifndef IPL_DPOPERATOR_H
#define IPL_DPOPERATOR_H

#include <string>

#include "GridManager.hh"
#include "TopologyManager.hh"
#include "data/Rectangle.hh"
#include "database/DPDatabase.hh"

namespace ipl {

class DPOperator
{
 public:
  DPOperator();

  DPOperator(const DPOperator&) = delete;
  DPOperator(DPOperator&&) = delete;
  ~DPOperator();

  DPOperator& operator=(const DPOperator&) = delete;
  DPOperator& operator=(DPOperator&&) = delete;

  TopologyManager* get_topo_manager() const { return _topo_manager; }
  GridManager* get_grid_manager() const { return _grid_manager; }

  void initDPOperator(DPDatabase* database, DPConfig* config);
  void updateTopoManager();
  void updateGridManager();
  void initPlaceableArea();

  std::pair<int32_t, int32_t> obtainOptimalXCoordiLine(DPInstance* inst);
  std::pair<int32_t, int32_t> obtainOptimalYCoordiLine(DPInstance* inst);
  Rectangle<int32_t> obtainOptimalCoordiRegion(DPInstance* inst);

  int64_t calInstAffectiveHPWL(DPInstance* inst);
  int64_t calInstPairAffectiveHPWL(DPInstance* inst_1, DPInstance* inst_2);

  bool checkIfClustered();
  void updateInstClustering();
  void pickAndSortMovableInstList(std::vector<DPInstance*>& movable_inst_list);
  DPCluster* createClsuter(DPInstance* inst, DPInterval* interval);

  bool checkOverlap(int32_t boundary_min, int32_t boundary_max, int32_t query_min, int32_t query_max);
  bool checkInNest(Rectangle<int32_t>& inner_box, Rectangle<int32_t>& outer_box);
  Rectangle<int32_t> obtainOverlapRectangle(Rectangle<int32_t>& box_1, Rectangle<int32_t>& box_2);
  std::pair<int32_t, int32_t> obtainOverlapRange(int32_t boundary_min, int32_t boundary_max, int32_t query_min, int32_t query_max);
  bool checkInBox(int32_t boundary_min, int32_t boundary_max, int32_t query_min, int32_t query_max);

  int64_t calTotalHPWL();

 private:
  DPDatabase* _database;
  DPConfig* _config;
  TopologyManager* _topo_manager;
  GridManager* _grid_manager;

  void initTopoManager();
  void initGridManager();
  void initGridManagerFixedArea();
  bool isCoreOverlap(DPInstance* inst);
  void cutOutShape(Rectangle<int32_t>& shape);
};
}  // namespace ipl

#endif