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
#ifndef IDRC_SRC_MODULE_REGION_QUERY_H_
#define IDRC_SRC_MODULE_REGION_QUERY_H_

#include <algorithm>
#include <map>
#include <memory>
#include <vector>

#include "BoostType.h"
#include "DRCUtil.h"
#include "DrcDesign.h"
#include "Tech.h"

namespace idrc {

class DrcConfig;

class RegionQuery
{
 public:
  // static RegionQuery* getInstance(DrcConfig* config = nullptr, DrcDesign* design = nullptr)
  // {
  //   static RegionQuery instance(config, design);
  //   return &instance;
  // }
  //-------------------
  RegionQuery() = default;
  RegionQuery(Tech* in) { _tech = in; }
  RegionQuery(DrcConfig* config, DrcDesign* design) { init(config, design); }
  // RegionQuery(const RegionQuery& other) = delete;
  // RegionQuery(RegionQuery&& other) = delete;
  ~RegionQuery() {}
  // RegionQuery& operator=(const RegionQuery& other) = default;
  // RegionQuery& operator=(RegionQuery&& other) = delete;

  // function
  void init(DrcConfig* config, DrcDesign* design);
  int get_short_vio_nums();
  int get_prl_spacing_vio_nums();
  int get_metal_eol_vio_nums();

  void initPrlVioSpot();
  void initShortVioSpot();
  void initMetalEOLVioSpot();

  void getRegionReport(std::map<std::string, int>& viotype_to_nums_map);
  void getRegionDetailReport(std::map<std::string, std::vector<DrcViolationSpot*>>& vio_map);

  void getIntersectPoly(std::set<DrcPoly*>& intersect_poly_set, std::vector<DrcRect*> drc_rect_list);
  void deleteIntersectPoly(std::set<DrcPoly*>& intersect_poly_set);
  DrcPoly* rebuildPoly_add(std::set<DrcPoly*>& intersect_poly_set, std::vector<DrcRect*> drc_rect_list);
  std::vector<DrcPoly*> rebuildPoly_del(std::set<DrcPoly*>& intersect_poly_set, std::vector<DrcRect*> drc_rect_list);
  void addPoly(DrcPoly* new_polygon);
  void addPolyEdge_NotAddToRegion(DrcPoly* new_poly);

  void addPolyList(std::vector<DrcPoly*>& new_poly_list);
  void addScopeToMaxScopeRTree(DrcRect* scope);
  void addScopeToMinScopeRTree(DrcRect* scope);

  void addViolation(ViolationType vio_type);
  bool addPRLRunLengthSpacingViolation(DrcRect* target_rect, DrcRect* result_rect);
  void addPRLRunLengthSpacingViolation(int layer_id, RTreeBox span_box);
  bool addShortViolation(DrcRect* target_rect, DrcRect* result_rect);
  void addShortViolation(int layer_id, RTreeBox span_box);
  void addMetalEOLSpacingViolation(int layer_id, RTreeBox span_box);

  bool addCutSpacingViolation(DrcRect* target_rect, DrcRect* result_rect);
  bool addCutDiffLayerSpacingViolation(DrcRect* target_rect, DrcRect* result_rect);
  bool addCutEOLSpacingViolation(DrcRect* target_rect, DrcRect* result_rect);

  // setter
  // getter
  std::set<DrcRect*>& getCutRectSet() { return _cut_rect_set; }
  std::set<DrcRect*>& getRoutingRectSet() { return _routing_rect_set; }
  std::map<int, std::map<int, std::set<DrcPoly*>>>& getRegionPolysMap() { return _region_polys_map; }
  std::map<int, bgi::rtree<std::pair<RTreeBox, DrcRect*>, bgi::quadratic<16>>>& get_layer_to_routing_rects_tree_map()
  {
    return _layer_to_routing_rects_tree_map;
  }
  std::map<int, bgi::rtree<std::pair<RTreeBox, DrcRect*>, bgi::quadratic<16>>>& get_layer_to_fixed_rects_tree_map()
  {
    return _layer_to_fixed_rects_tree_map;
  }
  std::map<int, DrcNet>& get_nets_map() { return _nets_map; }
  // function
  /**********目前用到的接口**************/

  void clear()
  {
    _layer_to_routing_rects_tree_map.clear();
    _layer_to_fixed_rects_tree_map.clear();
    _layer_to_cut_rects_tree_map.clear();
    _layer_to_block_edges.clear();
    _layer_to_routing_edges.clear();
  }
  // 获得搜索区域内的矩形
  void queryInRoutingLayer(int routingLayerId, RTreeBox query_box, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result);
  // find cut rect enclosure
  void queryEnclosureInRoutingLayer(int LayerId, RTreeBox query_box, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result);
  // 将线段或通孔矩形添加到对应金属层的R树
  void add_routing_rect_to_rtree(int routingLayerId, DrcRect* drcRect);

  // 将Pin或Blockage矩形添加到对应金属层的R树
  void add_fixed_rect_to_rtree(int routingLayerId, DrcRect* drcRect);

  /**********目前用到的接口**************/
  void add_routing_rect_to_api_rtree(int routingLayerId, DrcRect* rect);
  void add_spacing_min_region(DrcRect* common_spacing_min_region);
  void add_spacing_max_region(DrcRect* common_spacing_max_region);
  void addDrcRect(DrcRect* drc_rect, Tech* tech);
  void removeDrcRect(DrcRect* drc_rect);
  void addRectScope(DrcRect* drc_rect, Tech* tech);
  void removeRectScope(DrcRect* drc_rect);

  void queryContainsInRoutingLayer(int routingLayerId, RTreeBox query_box, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result);
  void queryIntersectsInRoutingLayer(int routingLayerId, RTreeBox query_box, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result);
  void queryInCutLayer(int cutLayerId, RTreeBox query_box, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result);
  void queryEdgeInRoutingLayer(int routingLayerId, RTreeBox query_box, std::vector<std::pair<RTreeSegment, DrcEdge*>>& result);

  void clear_layer_to_routing_rects_tree_map() { _layer_to_routing_rects_tree_map.clear(); }
  void initOnlyRoutingRectsFromDesign();
  // cut层目前没用
  void add_cut_rect_to_rtree(int cutLayerId, DrcRect* drcRect);
  // add edge 目前没用
  void add_routing_edge_to_rtree(int routingLayerId, DrcEdge* drcEdge);
  void add_block_edge_to_rtree(int routingLayerId, DrcEdge* drcEdge);
  // query edge

  // check
  // bool isExistingRectangleInRoutingRTree(int layerId, const DrcRectangle<int>& rectangle);
  // debug
  void printRoutingRectsRTree();
  void printFixedRectsRTree();

  std::set<DrcPoly*> getPolys(int net_id, int layer_id) { return _region_polys_map[net_id][layer_id]; }

  void queryInMaxScope(int layer_id, RTreeBox check_rect, std::map<void*, std::map<ScopeType, std::vector<DrcRect*>>>& query_result);
  void queryInMinScope(int layer_id, RTreeBox check_rect, std::map<void*, std::map<ScopeType, std::vector<DrcRect*>>>& query_result);
  std::vector<DrcViolationSpot*> _short_vio_spot_list;
  std::vector<DrcViolationSpot*> _cut_spacing_spot_list;
  std::vector<DrcViolationSpot*> _cut_eol_spacing_spot_list;
  std::vector<DrcViolationSpot*> _cut_diff_layer_spacing_spot_list;
  std::vector<DrcViolationSpot*> _cut_enclosure_spot_list;
  std::vector<DrcViolationSpot*> _cut_enclosure_edge_spot_list;
  std::vector<DrcViolationSpot*> _prl_run_length_spacing_spot_list;

  std::map<int, bgi::rtree<RTreeBox, bgi::quadratic<16>>> _layer_to_prl_vio_box_tree;
  std::map<int, bgi::rtree<RTreeBox, bgi::quadratic<16>>> _layer_to_short_vio_box_tree;
  std::map<int, bgi::rtree<RTreeBox, bgi::quadratic<16>>> _layer_to_metal_EOL_vio_box_tree;

  std::vector<DrcViolationSpot*> _metal_corner_fill_spacing_spot_list;
  std::vector<DrcViolationSpot*> _metal_jog_spacing_spot_list;
  std::vector<DrcViolationSpot*> _metal_eol_spacing_spot_list;
  std::vector<DrcViolationSpot*> _metal_notch_spacing_spot_list;
  std::vector<DrcViolationSpot*> _min_area_spot_list;
  std::vector<DrcViolationSpot*> _min_step_spot_list;
  std::vector<DrcViolationSpot*> _min_hole_spot_list;

 private:
  int _cut_diff_layer_spacing_count = 0;
  int _common_spacing_count = 0;
  int _eol_spacing_count = 0;
  int _short_count = 0;
  int _corner_fill_spacing_count = 0;
  int _notch_spacing_count = 0;
  int _jog_spacing_count = 0;
  int _cut_common_spacing_count = 0;
  int _cut_eol_spacing_count = 0;
  int _area_count = 0;
  int _common_enclosure_count = 0;
  int _egde_enclosure_count = 0;
  int _width_count = 0;
  int _minstep_count = 0;
  int _min_hole_count = 0;

  std::set<std::pair<DrcRect*, DrcRect*>> _prl_spacing_vio_set;
  std::set<std::pair<DrcRect*, DrcRect*>> _short_vio_set;
  std::set<std::pair<DrcRect*, DrcRect*>> _cut_spacing_vio_set;

  std::set<std::pair<DrcRect*, DrcRect*>> _cut_diff_layer_spacing_vio_set;

  std::set<std::pair<DrcRect*, DrcRect*>> _cut_eol_spacing_vio_set;

  DrcConfig* _config;
  DrcDesign* _drc_design;
  Tech* _tech;
  std::set<DrcRect*> _cut_rect_set;
  std::set<DrcRect*> _routing_rect_set;
  std::map<int, DrcNet> _nets_map;
  std::map<int, std::map<int, std::set<DrcPoly*>>> _region_polys_map;  // Use net_id and layer_id to store poly in order
  // routing layer
  std::map<int, bgi::rtree<std::pair<RTreeBox, DrcRect*>, bgi::quadratic<16>>> _layer_to_routing_rects_tree_map;  // via and segment
  std::map<int, bgi::rtree<std::pair<RTreeBox, DrcRect*>, bgi::quadratic<16>>> _layer_to_fixed_rects_tree_map;    // pin and block

  std::map<int, bgi::rtree<std::pair<RTreeBox, DrcRect*>, bgi::quadratic<16>>> _layer_to_routing_min_region_tree_map;
  std::map<int, bgi::rtree<std::pair<RTreeBox, DrcRect*>, bgi::quadratic<16>>> _layer_to_routing_max_region_tree_map;

  ////////////////////////////////
  ///////下面的没用到
  // cut layer
  std::map<int, bgi::rtree<std::pair<RTreeBox, DrcRect*>, bgi::quadratic<16>>> _layer_to_cut_rects_tree_map;
  // drc edge
  std::map<int, bgi::rtree<std::pair<RTreeSegment, DrcEdge*>, bgi::quadratic<16>>> _layer_to_block_edges;  // block edges
  std::map<int, bgi::rtree<std::pair<RTreeSegment, DrcEdge*>, bgi::quadratic<16>>>
      _layer_to_routing_edges;  // pin and segment via merge edges

  // add rect to rtree
  // void add_routing_rect_to_rtree(int routingLayerId, DrcRect* drcRect);
  // void add_fixed_rect_to_rtree(int routingLayerId, DrcRect* drcRect);
  // void add_cut_rect_to_rtree(int cutLayerId, DrcRect* drcRect);

  // query
  void searchRoutingRect(int routingLayerId, RTreeBox query_box, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result);
  void searchFixedRect(int routingLayerId, RTreeBox query_box, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result);
  void searchCutRect(int cutLayerId, RTreeBox query_box, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result);

  void searchRoutingEdge(int routingLayerId, RTreeBox query_box, std::vector<std::pair<RTreeSegment, DrcEdge*>>& result);
  void searchBlockEdge(int routingLayerId, RTreeBox query_box, std::vector<std::pair<RTreeSegment, DrcEdge*>>& result);

  // std::vector<std::pair<RTreeSegment, DrcEdge*>> searchRoutingRectEdge(int routingLayerId, RTreeBox query_box);

  // DrcAPI
  void deletePolyInNet(DrcPoly* poly);
  void deletePolyInEdgeRTree(DrcPoly* poly);
  void deletePolyInScopeRTree(DrcPoly* poly);
  void removeFromMaxScopeRTree(DrcRect* scope_rect);
  void removeFromMinScopeRTree(DrcRect* scope_rect);
  void addPolyEdge(DrcPoly* new_poly);
  void addPolyToRegionQuery(DrcPoly* new_poly);
  void addPolyScopes(DrcPoly* new_poly);
  void getCommonSpacingMinRegion(DrcRect* common_spacing_min_region, DrcRect* drc_rect, Tech* tech);
  void getCommonSpacingMaxRegion(DrcRect* common_spacing_min_region, DrcRect* drc_rect, Tech* tech);
  void initPolyOuterEdges(DrcNet* net, DrcPoly* poly, DrcPolygon* polygon, int layer_id);
  void initPolyInnerEdges(DrcNet* net, DrcPoly* poly, const BoostPolygon& hole_poly, int layer_id);
  void initPolyOuterEdges_NotAddToRegion(DrcNet* net, DrcPoly* poly, DrcPolygon* polygon, int layer_id);
  void initPolyInnerEdges_NotAddToRegion(DrcNet* net, DrcPoly* poly, const BoostPolygon& hole_poly, int layer_id);

  void addPolyMinScopes(DrcPoly* new_poly);
  void addPolyMaxScopes(DrcPoly* new_poly);
  void getEOLSpacingScope(DrcPoly* new_poly, bool is_max);
  void getCornerFillSpacingScope(DrcPoly* new_poly);

  RTreeSegment getRTreeSegment(DrcEdge* drcEdge);
  RTreeBox getRTreeBox(DrcRect* drcRect);
};
}  // namespace idrc

#endif