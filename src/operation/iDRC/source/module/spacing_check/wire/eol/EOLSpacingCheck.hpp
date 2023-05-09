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
typedef idb::routinglayer::Lef58SpacingEol DBEol;
class EOLSpacingCheck
{
 public:
  // static EOLSpacingCheck* getInstance(DrcConfig* config = nullptr, Tech* tech = nullptr)
  // {
  //   static EOLSpacingCheck instance(config, tech);
  //   return &instance;
  // }
  EOLSpacingCheck(DrcConfig* config, Tech* tech, RegionQuery* region_query) { init(config, tech, region_query); }
  // interact with other operations
  EOLSpacingCheck(Tech* tech, RegionQuery* region_query)
  {
    _tech = tech;
    _region_query = region_query;
    _interact_with_op = true;
  }
  EOLSpacingCheck(const EOLSpacingCheck& other) = delete;
  EOLSpacingCheck(EOLSpacingCheck&& other) = delete;
  ~EOLSpacingCheck() {}
  EOLSpacingCheck& operator=(const EOLSpacingCheck& other) = delete;
  EOLSpacingCheck& operator=(EOLSpacingCheck&& other) = delete;

  // setter
  void set_lef58_eol_spacing_rule_list(const std::vector<std::shared_ptr<idb::routinglayer::Lef58SpacingEol>>& in) { _rules = in; }
  // getter
  std::vector<std::shared_ptr<idb::routinglayer::Lef58SpacingEol>>& get_lef58_eol_spacing_rule_list() { return _rules; }
  std::map<int, std::vector<DrcSpot>>& get_routing_layer_to_eol_spacing_spots_list() { return _routing_layer_to_eol_spacing_spots_list; }
  std::map<int, std::vector<DrcSpot>>& get_routing_layer_to_e2e_spacing_spots_list() { return _routing_layer_to_e2e_spacing_spots_list; }

  // function

  void checkEOLSpacing(DrcNet* target_net);
  void checkEOLSpacing(DrcPoly* target_poly);
  void checkEOLSpacingEnd2End(DrcEdge* edge);
  void checkEOLSpacing(DrcEdge* edge);
  // operation api
  bool check(DrcNet* target_net);
  bool check(void* poly, DrcRect* rect);
  bool check(DrcPoly* poly);
  void getScope(DrcPoly* target_poly, std::vector<DrcRect*>& max_scope_list, bool is_max);
  void addScope(DrcPoly* target_poly, bool is_max, RegionQuery* rq);

  int get_eol_violation_num();
  int get_e2e_violation_num();

  void reset();

  // init conflict graph by polygon
  // void initConflictGraphByPolygon() { _conflict_graph->initGraph(_conflict_polygon_map); }

 private:
  int _rule_index;
  DrcConfig* _config;
  Tech* _tech;
  RegionQuery* _region_query;
  std::set<DrcRect*> _checked_rect_list;
  std::map<int, std::vector<DrcSpot>> _routing_layer_to_eol_spacing_spots_list;
  std::map<int, std::vector<DrcSpot>> _routing_layer_to_e2e_spacing_spots_list;
  // std::map<DrcPolygon*, std::set<DrcPolygon*>> _conflict_polygon_map;
  // std::map<int, bgi::rtree<RTreeBox, bgi::quadratic<16>>> _layer_to_violation_box_tree;
  // std::vector<std::pair<DrcRect*, DrcRect*>> _violation_rect_pair_list;
  std::vector<std::shared_ptr<idb::routinglayer::Lef58SpacingEol>> _rules;

  //-----------------------------------------------
  //-----------------------------------------------
  //----------interact with other operations----------
  bool _interact_with_op = false;
  bool _check_result = true;

  // function

  /// init
  void init(DrcConfig* config, Tech* tech, RegionQuery* region_query)
  {
    _config = config;
    _tech = tech;
    _region_query = region_query;
  }
  /// check routing spacing of target rect
  void checkRoutingSpacing(DrcRect* target_rect);
  /// get spacing querybox
  int getLayerMaxRequireSpacing(int routingLayerId);
  RTreeBox getSpacingQueryBox(DrcRect* target_rect, int spacing);
  /// get query result through query box
  std::vector<std::pair<RTreeBox, DrcRect*>> getQueryResult(int routingLayer, RTreeBox& query_box);
  /// check spacing from query result
  void checkSpacingFromQueryResult(int layerId, DrcRect* target_rect, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result);
  bool skipCheck(DrcRect* result_rect, DrcEdge* edge, DrcRect* query_box);
  bool checkShort(DrcRect* target_rect, DrcRect* result_rect);
  bool checkSpacingViolation(int routingLayerId, DrcRect* target_rect, DrcRect* result_rect,
                             std::vector<std::pair<RTreeBox, DrcRect*>>& query_result);
  bool intersectionExceptJustEdgeTouch(DrcRect* result_rect, DrcRect* query_box);

  bool isEdgeEOL(DrcEdge* edge, bool is_end2end);
  bool isAdjEdgeConsMet(DrcEdge* edge);
  bool isPRLConsMet(DrcEdge* edge);
  bool isCutConsMet(DrcEdge* edge);
  void getCutEncloseDistQueryBox(RTreeBox& query_box, DrcEdge* edge);
  void getCutToMetalSpaceQueryBox(RTreeBox& query_box, DrcEdge* edge, DrcRect* cut_rect);
  bool isCutConsMetOneDir(DrcEdge* edge, bool is_below);
  bool checkCutToMetalSpace(DrcEdge* edge, DrcRect* cut);
  bool checkExistPRLEdge(DrcEdge* edge, bool is_dir_left);
  void getExtPrlEdgeRect(DrcEdge* edge, BoostRect& rect);
  void getPRLQueryBox(DrcEdge* edge, bool is_dir_left, RTreeBox& query_box);
  void getCheckRuleQueryBox(DrcEdge* edge, RTreeBox& query_box);
  void checkEOLSpacingHelper(DrcEdge* edge);
  bool isEnd2EndTriggered(DrcEdge* result_edge, DrcEdge* edge);
  bool isTwoEOLHasPrlLength(DrcEdge* result_edge, DrcEdge* edge);
  void getEnd2EndQueryBox(RTreeBox& query_box, DrcEdge* edge);
  bool skipCheck();
  void checkEOLSpacingRule(int check_spacing, DrcEdge* edge);
  bool hasPrlOverlap(DrcEdge* result_edge, DrcEdge* edge);
  bool checkPointSpacing(DrcEdge* target_edge, DrcEdge* result_edge);
  bool checkPrlSpacing(DrcEdge* target_edge, DrcEdge* result_edge);
  bool isOnlyEdgeTouch(BoostRect result_rect, BoostRect target_rect);
  bool isTwoEdgeOppsite(DrcEdge* edge1, DrcEdge* edge2);
  bool isEdgeRectInterectTargetEdge(DrcEdge* edge, DrcEdge* target_edge);
  void storeEnd2EndViolationResult(DrcEdge* result_edge, DrcEdge* edge);
  void storeEOLViolationResult(DrcRect* rect, DrcEdge* edge);
  bool isSameMetalMet(RTreeBox result_box, DrcEdge* edge);
  bool checkExistPRLEdge_SubWidth(DrcEdge* edge);

  // DrcAPI
  void getEOLSpacingScopeRect(DrcRect* eol_spacing_max_scope, DrcEdge* edge, bool is_max);
  bool checkEOLSpacing_api(DrcEdge* edge, DrcRect* rect);
  bool checkEOLSpacingHelper_api(DrcEdge* edge, DrcRect* rect);
  void getPreciseScope(DrcEdge* edge, DrcRect* precise_scope);

  /// storage spot
  void addSpot(DrcEdge* target_rect, DrcRect* result_rect);
  void addE2ESpot(DrcEdge* edge, DrcEdge* result_edge);
};
}  // namespace idrc
