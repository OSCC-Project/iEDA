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

#include <map>
#include <set>

#include "BoostType.h"
#include "DRCUtil.h"
#include "DrcConflictGraph.h"
#include "DrcDesign.h"
#include "RegionQuery.h"
#include "Tech.h"

namespace idrc {

class CornerFillSpacingCheck
{
 public:
  // static CornerFillSpacingCheck* getInstance(DrcConfig* config = nullptr, Tech* tech = nullptr)
  // {
  //   static CornerFillSpacingCheck instance(config, tech);
  //   return &instance;
  // }
  CornerFillSpacingCheck(DrcConfig* config, Tech* tech, RegionQuery* region_query) { init(config, tech, region_query); }
  // interact with other operations
  CornerFillSpacingCheck(Tech* tech, RegionQuery* region_query)
  {
    _tech = tech;
    _region_query = region_query;
    _interact_with_op = true;
  }
  CornerFillSpacingCheck(const CornerFillSpacingCheck& other) = delete;
  CornerFillSpacingCheck(CornerFillSpacingCheck&& other) = delete;
  ~CornerFillSpacingCheck() {}
  CornerFillSpacingCheck& operator=(const CornerFillSpacingCheck& other) = delete;
  CornerFillSpacingCheck& operator=(CornerFillSpacingCheck&& other) = delete;

  // function
  void checkCornerFillSpacing(DrcNet* target_net);
  void checkCornerFillSpacing(DrcPoly* target_poly);
  void checkCornerFillSpacing(DrcEdge* edge);

  // operation api
  bool check(DrcNet* target_net);
  bool check(DrcPoly* poly);
  void getScope(DrcPoly* target_poly, std::vector<DrcRect*>& max_scope_list);
  void addScope(DrcPoly* target_poly, RegionQuery* rq);

  void reset();

  // init conflict graph by polygon
  // void initConflictGraphByPolygon() { _conflict_graph->initGraph(_conflict_polygon_map); }

 private:
  bool _is_edge_length2;
  DrcRect _corner_fill_rect;
  DrcConfig* _config;
  Tech* _tech;
  RegionQuery* _region_query;
  std::set<DrcRect*> _checked_rect_list;
  std::shared_ptr<idb::routinglayer::Lef58CornerFillSpacing> _rule;
  // std::map<DrcPolygon*, std::set<DrcPolygon*>> _conflict_polygon_map;
  // std::map<int, bgi::rtree<RTreeBox, bgi::quadratic<16>>> _layer_to_violation_box_tree;
  // std::vector<std::pair<DrcRect*, DrcRect*>> _violation_rect_pair_list;

  //-----------------------------------------------
  //-----------------------------------------------
  //----------interact with other operations----------
  bool _interact_with_op = false;
  bool _check_result = true;

  // function
  bool isConCaveCornerTriggerMet(DrcEdge* edge);
  bool isLengthTriggerMet(DrcEdge* edge);
  bool isEOLTriggerMet(DrcEdge* edge);
  bool isEdgeEOL(DrcEdge* edge);
  void getCornerFillRect(DrcEdge* edge);
  void getQueryBox(DrcEdge* edge, RTreeBox& query_box);
  bool intersectionExceptJustEdgeTouch(RTreeBox* query_rect, DrcRect* result_rect);
  bool skipCheck(DrcRect* result_rect);
  bool isSameNetRectConnect(DrcRect* result_rect);
  void checkSpacingToRect(DrcRect* result_rect);
  bool checkShort(DrcRect* result_rect);
  void checkSpacing(DrcRect* result_rect);
  bool isParallelOverlap(DrcRect* result_rect);
  void checkCornerSpacing(DrcRect* result_rect);
  void checkXYSpacing(DrcRect* result_rect);

  // DrcAPI
  void getScopeOfEdge(DrcEdge* target_edge, std::vector<DrcRect*>& max_scope_list);
  void getScopeRect(DrcEdge* edge, DrcRect* scope_rect);

  /// init
  void init(DrcConfig* config, Tech* tech, RegionQuery* region_query)
  {
    _config = config;
    _tech = tech;
    _region_query = region_query;
  }

  /// storage spot
};
}  // namespace idrc