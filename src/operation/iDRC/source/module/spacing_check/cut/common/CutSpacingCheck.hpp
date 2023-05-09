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

#include <set>

#include "BoostType.h"
#include "DRCUtil.h"
#include "DrcDesign.h"
#include "RegionQuery.h"
#include "Tech.h"

namespace idrc {

class DrcConfig;
class Tech;
class RoutingSpacingCheck;
class CutSpacingCheck;
class RoutingWidthCheck;
class EnclosedAreaCheck;
class DrcSpot;
class DrcRect;
class DrcNet;
class RegionQuery;
enum class ViolationType;

class CutSpacingCheck
{
 public:
  // static CutSpacingCheck* getInstance(DrcConfig* config = nullptr, Tech* tech = nullptr)
  // {
  //   static CutSpacingCheck instance(config, tech);
  //   return &instance;
  // }
  CutSpacingCheck(Tech* tech, RegionQuery* region_query)
  {
    _tech = tech;
    _region_query = region_query;
    _interact_with_op = true;
  }
  CutSpacingCheck(DrcConfig* config, Tech* tech, RegionQuery* region_query) { init(config, tech, region_query); }
  CutSpacingCheck(const CutSpacingCheck& other) = delete;
  CutSpacingCheck(CutSpacingCheck&& other) = delete;
  ~CutSpacingCheck() {}
  CutSpacingCheck& operator=(const CutSpacingCheck& other) = delete;
  CutSpacingCheck& operator=(CutSpacingCheck&& other) = delete;

  // setter
  // getter
  std::map<int, std::vector<DrcSpot>>& get_cut_layer_to_spacing_spots_list() { return _cut_layer_to_spacing_spots_list; }
  // function
  void checkCutSpacing(DrcNet* target_net);
  // interact api
  bool check(DrcRect* target_rect);
  // Read in iRT data for interaction
  void checkRoutingSpacing(const LayerNameToRTreeMap& layer_to_rects_tree_map);

  // List of offending rectangle pairs returned in interaction mode with iRT
  std::vector<std::pair<DrcRect*, DrcRect*>>& get_violation_rect_pair_list() { return _violation_rect_pair_list; }
  // Switch to the mode of interacting with iRT
  void switchToiRTMode() { _interact_with_op = true; }

  void reset();

  int get_spacing_violation_num();

 private:
  int _rule_index = 0;
  bool _interact_with_op = false;
  bool _check_result = true;

  RegionQuery* _region_query;
  DrcConfig* _config;
  Tech* _tech;
  std::vector<std::shared_ptr<idb::cutlayer::Lef58SpacingTable>> _lef58_spacing_table_list;
  std::vector<std::shared_ptr<idb::cutlayer::Lef58Cutclass>> _lef58_cut_class_list;
  std::map<int, std::vector<DrcSpot>> _cut_layer_to_spacing_spots_list;
  std::map<int, bgi::rtree<RTreeBox, bgi::quadratic<16>>> _layer_to_violation_box_tree;
  std::vector<std::pair<DrcRect*, DrcRect*>> _violation_rect_pair_list;
  //-----------------------------------------------
  //-----------------------------------------------
  void getSpacing2QueryBoxList_PrlNeg(std::vector<RTreeBox>& query_box_list, DrcRect* target_rect);
  void getSpacing2QueryBox_PrlNeg(RTreeBox& query_box, QueryBoxDir dir, DrcRect* target_rect);
  void checkSpacing2_PrlNeg(DrcRect* target_rect);
  void checkSpacing1_PrlNeg(DrcRect* target_rect);
  void getSpacing1QueryBoxList_PrlNeg(std::vector<RTreeBox>& query_box_list, DrcRect* drc_rect);
  void getSpacing1QueryBox_PrlNeg(RTreeBox& query_box, QueryBoxDir dir, DrcRect* drc_rect);
  void checkSpacing1_TwoRect_PrlNeg(DrcRect* result_rect, DrcRect* target_rect);
  int getRequiredSpacing1(DrcRect* result_rect, DrcRect* target_rect);
  int getCutClassIndex(DrcRect* cut_rect);
  void checkCornerSpacing(DrcRect* result_rect, DrcRect* target_rect, int required_spacing);
  void checkSpacing_PrlPos(DrcRect* target_rect);
  int getQuerySpacing_PrlPos(int cut_class_index, DrcRect* target_rect);
  void getQueryBox(RTreeBox& query_box, DrcRect* target_rect, int cut_class_query_spacing);
  int getQueryLayerId_PrlPos();
  void checkQueryResult_PrlPos(std::vector<std::pair<RTreeBox, DrcRect*>>& query_result, DrcRect* target_rect, RTreeBox& query_box);
  bool skipCheck(DrcRect* target_rect, DrcRect* result_rect);
  void checkSpacing_TwoRect_PrlPos(DrcRect* target_rect, DrcRect* result_rect);
  int getRequiredSpacingOfTwoRect(DrcRect* target_rect, DrcRect* result_rect);
  int getPrlOfTwoRect(DrcRect* target_rect, DrcRect* result_rect);
  void checkMaxXYSpacing(DrcRect* target_rect, DrcRect* result_rect, int required_spacing);
  void checkCutSpacing_SingleValue(DrcRect* target_rect);
  void checkQueryResult_SingleValue(std::vector<std::pair<RTreeBox, DrcRect*>>& query_result, DrcRect* target_rect, RTreeBox query_box);
  void checkSpacing_TwoRect_SingleValue(DrcRect* target_rect, DrcRect* result_rect);

  /// init
  void init(DrcConfig* config, Tech* tech, RegionQuery* region_query);
  /// check routing spacing of target rect
  void checkCutSpacing(DrcRect* target_rect);
  /// get spacing querybox
  RTreeBox getSpacingQueryBox(DrcRect* target_rect, int spacing);
  /// get query result through query box
  std::vector<std::pair<RTreeBox, DrcRect*>> getQueryResult(int routingLayer, RTreeBox& query_box);
  /// check spacing from query result
  void checkSpacingFromQueryResult(int layerId, DrcRect* target_rect, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result);
  bool checkSpacingViolation(int routingLayerId, DrcRect* target_rect, DrcRect* result_rect,
                             std::vector<std::pair<RTreeBox, DrcRect*>>& query_result);
  /// checking spacing between two rect
  bool intersectionExceptJustEdgeTouch(DrcRect* target_rect, DrcRect* result_rect);
  bool isParallelOverlap(DrcRect* target_rect, DrcRect* result_rect);
  bool checkCornerSpacingViolation(DrcRect* target_rect, DrcRect* result_rect, int require_spacing);
  bool checkSpanBoxCornorIntersectedByExitedRect(DrcRect* target_rect, DrcRect* result_rect,
                                                 std::vector<std::pair<RTreeBox, DrcRect*>>& query_result);
  bool checkXYSpacingViolation(DrcRect* target_rect, DrcRect* result_rect, int require_spacing,
                               std::vector<std::pair<RTreeBox, DrcRect*>>& query_result);
  bool checkSpanBoxCoveredByExistedRect(const RTreeBox& span_box, bool isHorizontalParallelOverlap,
                                        std::vector<std::pair<RTreeBox, DrcRect*>>& query_result);
  /// storage spot
  void storeViolationResult(int routingLayerId, DrcRect* target_rect, DrcRect* result_rect, ViolationType type);

  /// query violation box
  void searchIntersectedViolationBox(int routingLayerId, const RTreeBox& query_box, std::vector<RTreeBox>& result);
  void addViolationBox(int layerId, DrcRect* target_rect, DrcRect* result_rect);
  void initSpacingSpotListFromRtree();

  ////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////interact with iRT
  void checkCutSpacing(int layerId, DrcRect* target_rect, const RectRTree& rtree);

  void addCutSpacingSpot(DrcRect* target_rect, DrcRect* result_rect);
  void addDiffLayerSpot(DrcRect* target_rect, DrcRect* result_rect);
};
}  // namespace idrc
