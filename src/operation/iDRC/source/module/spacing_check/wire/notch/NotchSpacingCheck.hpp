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

class NotchSpacingCheck
{
 public:
  // static NotchSpacingCheck* getInstance(DrcConfig* config = nullptr, Tech* tech = nullptr)
  // {
  //   static NotchSpacingCheck instance(config, tech);
  //   return &instance;
  // }

  NotchSpacingCheck(DrcConfig* config, Tech* tech, RegionQuery* region_query) { init(config, tech, region_query); }
  NotchSpacingCheck(Tech* tech, RegionQuery* region_query)
  {
    _tech = tech;
    _region_query = region_query;
    _interact_with_op = true;
  }
  NotchSpacingCheck(const NotchSpacingCheck& other) = delete;
  NotchSpacingCheck(NotchSpacingCheck&& other) = delete;
  ~NotchSpacingCheck() {}
  NotchSpacingCheck& operator=(const NotchSpacingCheck& other) = delete;
  NotchSpacingCheck& operator=(NotchSpacingCheck&& other) = delete;

  // setter
  void set_notch_spacing_rule(const idb::IdbLayerSpacingNotchLength& in)
  {
    _notch_spacing_rule.set_min_spacing(in.get_min_spacing());
    _notch_spacing_rule.set_notch_length(in.get_notch_length());
  }
  void set_lef58_notch_spacing_rule(const std::shared_ptr<idb::routinglayer::Lef58SpacingNotchlength> in)
  {
    _lef58_notch_spacing_rule = in;
  }

  // getter
  std::shared_ptr<idb::routinglayer::Lef58SpacingNotchlength> get_lef58_notch_spacing_rule() { return _lef58_notch_spacing_rule; }
  std::map<int, std::vector<DrcSpot>>& get_routing_layer_to_notch_spacing_spots_list()
  {
    return _routing_layer_to_notch_spacing_spots_list;
  }

  // function

  void checkNotchSpacing(DrcNet* target_net);
  void checkNotchSpacing(DrcPoly* target_poly);
  void checkNotchSpacing(DrcEdge* edge);

  int get_notch_violation_num();

  void reset();

  // operation check api
  bool check(DrcNet* target_net);
  bool check(DrcPoly* target_poly);

  // init conflict graph by polygon
  // void initConflictGraphByPolygon() { _conflict_graph->initGraph(_conflict_polygon_map); }

 private:
  DrcConfig* _config;
  Tech* _tech;
  RegionQuery* _region_query;
  std::set<DrcRect*> _checked_rect_list;
  std::map<int, std::vector<DrcSpot>> _routing_layer_to_notch_spacing_spots_list;
  // std::map<DrcPolygon*, std::set<DrcPolygon*>> _conflict_polygon_map;
  // std::map<int, bgi::rtree<RTreeBox, bgi::quadratic<16>>> _layer_to_violation_box_tree;
  // std::vector<std::pair<DrcRect*, DrcRect*>> _violation_rect_pair_list;
  idb::IdbLayerSpacingNotchLength _notch_spacing_rule;
  std::shared_ptr<idb::routinglayer::Lef58SpacingNotchlength> _lef58_notch_spacing_rule;

  //-----------------------------------------------
  //-----------------------------------------------
  //----------interact with other operations----------
  bool _interact_with_op = false;
  bool _check_result = true;

  // function
  bool isNotchRuleLef58() { return _lef58_notch_spacing_rule != nullptr; }
  bool isEdgeNotchBottom(DrcEdge* edge);
  bool checkNotchShape(DrcEdge* edge);
  bool checkLef58NotchShape(DrcEdge* edge);
  bool checkLef58NotchSidesWidth(DrcEdge* pre_edge, DrcEdge* next_edge);
  int getRectOfEdgeMaxWidth(DrcEdge* edge);

  void getEdgeExtQueryBox(DrcEdge* edge, RTreeBox& rect);
  int getRectWidth(DrcRect* rect, DrcEdge* edge);
  void checkNotchSpacingRule(DrcEdge* edge);

  /// init
  void init(DrcConfig* config, Tech* tech, RegionQuery* region_query)
  {
    _config = config;
    _tech = tech;
    _region_query = region_query;
  }
  /// check routing spacing of target rect

  void storeEnd2EndViolationResult(DrcEdge* result_edge, DrcEdge* edge);
  void storeNOTCHViolationResult(DrcRect* rect, DrcEdge* edge);

  /// storage spot
  void addSpot(DrcEdge* edge);
};
}  // namespace idrc
