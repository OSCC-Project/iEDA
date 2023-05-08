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
class CutEolSpacingCheck;
class RoutingWidthCheck;
class EnclosedAreaCheck;
class DrcSpot;
class DrcRect;
class DrcNet;
class RegionQuery;
enum class ViolationType;

class CutEolSpacingCheck
{
 public:
  // static CutEolSpacingCheck* getInstance(DrcConfig* config = nullptr, Tech* tech = nullptr)
  // {
  //   static CutEolSpacingCheck instance(config, tech);
  //   return &instance;
  // }
  CutEolSpacingCheck(Tech* tech, RegionQuery* region_query)
  {
    _tech = tech;
    _region_query = region_query;
    _interact_with_op = true;
  }
  CutEolSpacingCheck(DrcConfig* config, Tech* tech, RegionQuery* region_query) { init(config, tech, region_query); }
  CutEolSpacingCheck(const CutEolSpacingCheck& other) = delete;
  CutEolSpacingCheck(CutEolSpacingCheck&& other) = delete;
  ~CutEolSpacingCheck() {}
  CutEolSpacingCheck& operator=(const CutEolSpacingCheck& other) = delete;
  CutEolSpacingCheck& operator=(CutEolSpacingCheck&& other) = delete;

  // setter
  // getter
  std::map<int, std::vector<DrcSpot>>& get_cut_layer_to_spacing_spots_list() { return _cut_layer_to_spacing_spots_list; }
  // function
  void checkCutEolSpacing(DrcNet* target_net);
  void checkCutEolSpacing(DrcRect* target_rect);

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
  bool _is_left_trigger_edge = false;
  bool _is_right_trigger_edge = false;
  bool _is_top_trigger_edge = false;
  bool _is_bottom_trigger_edge = false;
  int _rule_index = 0;
  bool _interact_with_op = false;
  bool _check_result = true;
  DrcPoly* _cut_above_poly;
  std::set<EdgeDirection> _cut_edges_need_to_check;
  std::vector<std::pair<DrcRect*, EdgeDirection>> _above_metal_rect_list;
  std::vector<std::tuple<RTreeBox, CornerDirEnum, EdgeDirection>> _ext_query_box_list;

  RegionQuery* _region_query;
  DrcConfig* _config;
  Tech* _tech;
  std::shared_ptr<idb::cutlayer::Lef58EolSpacing> _lef58_cut_eol_spacing;
  std::vector<std::shared_ptr<idb::cutlayer::Lef58Cutclass>> _lef58_cut_class_list;
  std::map<int, std::vector<DrcSpot>> _cut_layer_to_spacing_spots_list;
  std::map<int, bgi::rtree<RTreeBox, bgi::quadratic<16>>> _layer_to_violation_box_tree;
  std::vector<std::pair<DrcRect*, DrcRect*>> _violation_rect_pair_list;
  //-----------------------------------------------
  //-----------------------------------------------
  // check enclosure
  void refresh();
  bool isOverhangMet(DrcRect* target_rect);
  void getAboveMetalRectList(DrcRect* target_rect, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result);
  bool checkEnclosure(std::vector<std::pair<RTreeBox, DrcRect*>>& above_metal_rect_list, DrcRect* target_cut_rect);

  // trigger
  bool isSpanLengthMet(DrcRect* target_rect);
  bool isEolWidthMet(DrcRect* target_rect);
  void checkNeighborWireMet();

  void getExtBoxes();

  void queryInExtBoxes();

  void checkSpacing1_PrlNeg(DrcRect* target_rect, EdgeDirection edge_dir);
  void checkSpacing2_PrlNeg(DrcRect* target_rect, EdgeDirection edge_dir);
  void getSpacing2QueryBox_PrlNeg(RTreeBox& query_box, DrcRect* drc_rect, EdgeDirection edge_dir);
  void getSpacing1QueryBox_PrlNeg(RTreeBox& query_box, DrcRect* drc_rect, EdgeDirection edge_dir);

  void checkSpacing1_TwoRect_PrlNeg(DrcRect* target_rect, DrcRect* query_cut_rect, EdgeDirection edge);
  void checkCornerSpacing(DrcRect* result_rect, DrcRect* target_rect, EdgeDirection edge_dir, int required_spacing);
  int getRequiredSpacing1(DrcRect* result_rect, DrcRect* target_rect);
  std::string getCutClassName(DrcRect* cut_rect);
  bool isPrlOverlap(DrcRect* target_rect, DrcRect* result_rect, EdgeDirection edge_dir);
  void checkEdgeSpacing(DrcRect* target_rect, DrcRect* result_rect, EdgeDirection edge_dir, int required_spacing);

  /// init
  void init(DrcConfig* config, Tech* tech, RegionQuery* region_query);

  void addSpot(DrcRect* target_rect, DrcRect* result_rect);
};
}  // namespace idrc