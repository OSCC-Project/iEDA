#pragma once

#include <set>

#include "BoostType.h"
#include "DrcConflictGraph.h"
#include "DrcDesign.h"
#include "RegionQuery.h"
#include "Tech.h"

namespace idrc {
class RoutingSpacingCheck
{
 public:
  // static RoutingSpacingCheck* getInstance(DrcConfig* config = nullptr, Tech* tech = nullptr, DrcConflictGraph* graph = nullptr)
  // {
  //   static RoutingSpacingCheck instance(config, tech, graph);
  //   return &instance;
  // }
  RoutingSpacingCheck(DrcConfig* config, Tech* tech, RegionQuery* region_query) { init(config, tech, region_query); }
  RoutingSpacingCheck(Tech* tech, RegionQuery* region_query)
  {
    _tech = tech;
    _region_query = region_query;
  }

  RoutingSpacingCheck(Tech* tech) : _tech(tech), _interact_with_op(false) {}
  RoutingSpacingCheck(const RoutingSpacingCheck& other) = delete;
  RoutingSpacingCheck(RoutingSpacingCheck&& other) = delete;
  ~RoutingSpacingCheck() {}
  RoutingSpacingCheck& operator=(const RoutingSpacingCheck& other) = delete;
  RoutingSpacingCheck& operator=(RoutingSpacingCheck&& other) = delete;

  // setter
  // getter
  std::map<int, std::vector<DrcSpot>>& get_routing_layer_to_short_spots_list() { return _routing_layer_to_short_spots_list; }
  std::map<int, std::vector<DrcSpot>>& get_routing_layer_to_spacing_spots_list() { return _routing_layer_to_spacing_spots_list; }
  // function
  void checkRoutingSpacing(DrcNet* target_net);
  // //读入iRT数据进行交互
  // void checkRoutingSpacing(const LayerNameToRTreeMap& layer_to_rects_tree_map);

  // //与iRT交互模式下返回的违规矩形对列表
  // std::vector<std::pair<DrcRect*, DrcRect*>>& get_violation_rect_pair_list() { return _violation_rect_pair_list; }
  //切换到与其他工具交互的模式
  void switchToiRTMode() { _interact_with_op = true; }

  ////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////interact with operation
  bool check(DrcRect* target_rect);
  bool check(void* target, DrcRect* check_rect);

  void reset();

  int get_short_violation_num();
  int get_spacing_violation_num();
  // init conflict graph by polygon
  void initConflictGraphByPolygon() { _conflict_graph->initGraph(_conflict_polygon_map); }

 private:
  DrcConfig* _config;
  Tech* _tech;
  RegionQuery* _region_query;
  DrcConflictGraph* _conflict_graph;
  std::set<DrcRect*> _checked_rect_list;
  std::map<int, std::vector<DrcSpot>> _routing_layer_to_short_spots_list;
  std::map<int, std::vector<DrcSpot>> _routing_layer_to_spacing_spots_list;
  std::map<DrcPolygon*, std::set<DrcPolygon*>> _conflict_polygon_map;

  std::map<int, bgi::rtree<RTreeBox, bgi::quadratic<16>>> _layer_to_violation_box_tree;
  std::vector<std::pair<DrcRect*, DrcRect*>> _violation_rect_pair_list;

  //----------interact with other operations----------
  bool _interact_with_op = false;
  bool _check_result = true;

  //-----------------------------------------------
  //-----------------------------------------------

  // function
  int getPRLRunLength(DrcRect* target_rect, DrcRect* result_rect);

  /// init
  void init(DrcConfig* config, Tech* tech, RegionQuery* region_query);
  /// check routing spacing of target rect
  void checkRoutingSpacing(DrcRect* target_rect);
  /// get spacing querybox
  int getLayerMaxRequireSpacing(int routingLayerId);
  RTreeBox getSpacingQueryBox(DrcRect* target_rect, int spacing);
  /// get query result through query box
  std::vector<std::pair<RTreeBox, DrcRect*>> getQueryResult(int routingLayer, RTreeBox& query_box);
  /// check spacing from query result
  void checkSpacingFromQueryResult(int layerId, DrcRect* target_rect, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result);
  bool skipCheck(DrcRect* target_rect, DrcRect* result_rect);
  bool checkShort(DrcRect* target_rect, DrcRect* result_rect);
  bool checkSpacingViolation(int routingLayerId, DrcRect* target_rect, DrcRect* result_rect,
                             std::vector<std::pair<RTreeBox, DrcRect*>>& query_result);
  /// Conditions for skipping the check
  bool isSameNetRectConnect(DrcRect* target_rect, DrcRect* result_rect);
  bool isChecked(DrcRect* result_rect);
  /// checking spacing between two rect
  int getRequireSpacing(int routingLayerId, DrcRect* target_rect, DrcRect* result_rect);
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
  void add_spot(int routingLayerId, DrcRect* target_rect, DrcRect* result_rect, ViolationType type);
  void addShortSpot(DrcRect* target_rect, DrcRect* result_rect);
  void addSpacingSpot(DrcRect* target_rect, DrcRect* result_rect);
  /// build conflict graph

  /// query violation box
  void searchIntersectedViolationBox(int routingLayerId, const RTreeBox& query_box, std::vector<RTreeBox>& result);
  void addViolationBox(int layerId, DrcRect* target_rect, DrcRect* result_rect);
  void initSpacingSpotListFromRtree();

  // ////////////////////////////////////////////////////////////////////////////////////
  // ////////////////////////////////////////////////////////////////////////////////////
  // //////////////////////////////////////////////////////////interact with iRT
  // void checkRoutingSpacing(int layerId, DrcRect* target_rect, const RectRTree& rtree);
};
}  // namespace idrc
