#pragma once

#include <map>
#include <set>
#include <vector>

#include "BoostType.h"
#include "DRCUtil.h"
#include "DrcConfig.h"
#include "DrcDesign.h"
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
class EnclosureRule;
class RegionQuery;

class EnclosureCheck
{
 public:
  // static EnclosureCheck* getInstance(DrcConfig* config = nullptr, Tech* tech = nullptr)
  // {
  //   static EnclosureCheck instance(config, tech);
  //   return &instance;
  // }
  //-----------------------------------------------
  EnclosureCheck(Tech* tech, RegionQuery* region_query)
  {
    _tech = tech;
    _region_query = region_query;
  }
  EnclosureCheck(DrcConfig* config, Tech* tech, RegionQuery* region_query) { init(config, tech, region_query); }
  EnclosureCheck(const EnclosureCheck& other) = delete;
  EnclosureCheck(EnclosureCheck&& other) = delete;
  ~EnclosureCheck() {}
  EnclosureCheck& operator=(const EnclosureCheck& other) = delete;
  EnclosureCheck& operator=(EnclosureCheck&& other) = delete;

  // setter
  // getter
  std::map<int, std::vector<DrcSpot>>& get_cut_layer_to_enclosure_spots_list() { return _cut_layer_to_enclosure_spots_list; }

  bool check(DrcRect* target_rect);
  // function
  void checkEnclosure(DrcNet* target_net);
  // Read in iRT data for interaction
  void checkRoutingSpacing(const LayerNameToRTreeMap& layer_to_rects_tree_map);

  // List of violation rectangles returned in interaction mode with iRT
  std::vector<DrcRect*>& get_violation_rect_list() { return _violation_rect_list; }
  // Switch to the mode of interacting with iRT
  void switchToiRTMode() { _interact_with_irt = true; }

  void reset();

  int get_enclosure_violation_num();

 private:
  bool _interact_with_irt = false;
  bool _check_result = true;
  int _edge_rule_index = 0;
  int _left_overhang_above = -1;
  int _right_overhang_above = -1;
  int _top_overhang_above = -1;
  int _bottom_overhang_above = -1;
  DrcRect* _left_overhang_rect_above = nullptr;
  DrcRect* _right_overhang_rect_above = nullptr;
  DrcRect* _top_overhang_rect_above = nullptr;
  DrcRect* _bottom_overhang_rect_above = nullptr;

  int _left_overhang_below = -1;
  int _right_overhang_below = -1;
  int _top_overhang_below = -1;
  int _bottom_overhang_below = -1;
  DrcRect* _left_overhang_rect_below = nullptr;
  DrcRect* _right_overhang_rect_below = nullptr;
  DrcRect* _top_overhang_rect_below = nullptr;
  DrcRect* _bottom_overhang_rect_below = nullptr;

  DrcConfig* _config;
  Tech* _tech;
  RegionQuery* _region_query;
  DrcPoly* _cut_above_poly = nullptr;
  DrcPoly* _cut_below_poly = nullptr;
  std::vector<std::shared_ptr<idb::cutlayer::Lef58Enclosure>> _lef58_enclosure_list;
  std::vector<std::shared_ptr<idb::cutlayer::Lef58EnclosureEdge>> _lef58_enclosure_edge_list;
  std::vector<std::shared_ptr<idb::cutlayer::Lef58Cutclass>> _lef58_cut_class_list;

  std::map<int, std::vector<DrcSpot>> _cut_layer_to_enclosure_spots_list;

  std::map<int, bgi::rtree<RTreeBox, bgi::quadratic<16>>> _layer_to_violation_box_tree;
  std::vector<DrcRect*> _violation_rect_list;

  //-----------------------------------------------

  // function

  // refresh
  void reFresh()
  {
    _left_overhang_above = -1;
    _right_overhang_above = -1;
    _top_overhang_above = -1;
    _bottom_overhang_above = -1;
    _left_overhang_rect_above = nullptr;
    _right_overhang_rect_above = nullptr;
    _top_overhang_rect_above = nullptr;
    _bottom_overhang_rect_above = nullptr;

    _left_overhang_below = -1;
    _right_overhang_below = -1;
    _top_overhang_below = -1;
    _bottom_overhang_below = -1;
    _left_overhang_rect_below = nullptr;
    _right_overhang_rect_below = nullptr;
    _top_overhang_rect_below = nullptr;
    _bottom_overhang_rect_below = nullptr;
    _cut_above_poly = nullptr;
    _cut_below_poly = nullptr;
  }

  /// init
  void init(DrcConfig* config, Tech* tech, RegionQuery* region_query);
  /// check routing spacing of target rect
  void checkEnclosure(DrcRect* target_rect);
  void getEnclosureQueryBox(DrcRect* target_rect, int overhang1, int overhang2, std::vector<RTreeBox>& query_box_list);
  /// get query result through query box
  std::vector<std::pair<RTreeBox, DrcRect*>> getQueryResult(int CutLayerId, RTreeBox& query_box, EnclosureRule* enclosure_rule,
                                                            bool get_above_result, bool get_below_result);
  /// check spacing from query result
  bool checkEnclosureFromQueryResult(int layer_id, DrcRect* target_rect, EnclosureRule* enclosure_rule,
                                     std::vector<std::pair<RTreeBox, DrcRect*>> query_result);

  void checkAboveEnclosureRule(int cutLayerId, DrcRect* target_rect, std::vector<EnclosureRule*> above_enclosure_rule_list);
  void checkBelowEnclosureRule(int cutLayerId, DrcRect* target_rect, std::vector<EnclosureRule*> below_enclosure_rule_list);

  /// storage spot
  void storeViolationResult(int routingLayerId, DrcRect* target_rect);
  /// build conflict graph

  /// query violation box
  void searchIntersectedViolationBox(int routingLayerId, const RTreeBox& query_box, std::vector<RTreeBox>& result);
  void addViolationBox(int layerId, DrcRect* target_rect, DrcRect* result_rect);
  void initEnclosureSpotListFromRtree();

  bool checkOverhang_above(std::vector<std::pair<RTreeBox, DrcRect*>>& above_metal_rect_list, DrcRect* target_cut_rect);
  bool checkOverhang_below(std::vector<std::pair<RTreeBox, DrcRect*>>& above_metal_rect_list, DrcRect* target_cut_rect);
  std::string getCutClassName(DrcRect* cut_rect);

  void getBelowMetalRectList(DrcRect* target_cut_rect, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result);
  void getAboveMetalRectList(DrcRect* target_cut_rect, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result);

  // edge enclosure
  bool checkEdgeEnclosure(DrcRect* target_cut_rect);
  bool checkConvexCons(DrcEdge* drc_edge, DrcRect* target_cut_rect);
  bool checkParWithin(DrcRect* target_cut_rect, DrcEdge* drc_edge);
  void getTriggerEdgeQueryBox(RTreeBox& query_box, DrcRect* target_cut_rect, int required_overhang);
  void getWithinQueryBox(RTreeBox& within_query_box, DrcRect* target_cut_rect, DrcEdge* drc_edge);
  bool checkParallelCons(RTreeBox& query_box, DrcRect* target_enclosure_rect);
  void getPrlQueryBox_above(RTreeBox& query_box, EdgeDirection edge_dir);
  void getPrlQueryBox_below(RTreeBox& query_box, EdgeDirection edge_dir);

  // spot
  void addSpot(DrcRect* target_cut_rect);
  void addEdgeEnclosureSpot(DrcRect* target_cut_rect);
};
}  // namespace idrc