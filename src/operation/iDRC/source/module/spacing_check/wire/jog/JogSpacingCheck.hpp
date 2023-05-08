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
class JogSpacingCheck
{
 public:
  // static JogSpacingCheck* getInstance(DrcConfig* config = nullptr, Tech* tech = nullptr)
  // {
  //   static JogSpacingCheck instance(config, tech);
  //   return &instance;
  // }
  JogSpacingCheck(DrcConfig* config, Tech* tech, RegionQuery* region_query) { init(config, tech, region_query); }
  // interact with other operations
  JogSpacingCheck(Tech* tech, RegionQuery* region_query)
  {
    _tech = tech;
    _region_query = region_query;
    _interact_with_op = true;
  }
  JogSpacingCheck(const JogSpacingCheck& other) = delete;
  JogSpacingCheck(JogSpacingCheck&& other) = delete;
  ~JogSpacingCheck() {}
  JogSpacingCheck& operator=(const JogSpacingCheck& other) = delete;
  JogSpacingCheck& operator=(JogSpacingCheck&& other) = delete;

  // function

  void checkJogSpacing(DrcNet* target_net);
  void checkJogSpacing(DrcRect* target_rect);

  // operation api
  bool check(DrcNet* target_net);

  void reset();

  // init conflict graph by polygon
  // void initConflictGraphByPolygon() { _conflict_graph->initGraph(_conflict_polygon_map); }

 private:
  bool _wid_rect_dir_is_horizontal;
  int _width_list_index;
  bool _find_violation = false;
  DrcConfig* _config;
  Tech* _tech;
  RegionQuery* _region_query;
  std::set<DrcRect*> _checked_rect_list;
  std::shared_ptr<idb::routinglayer::Lef58SpacingTableJogToJog> _rule;
  // std::map<DrcPolygon*, std::set<DrcPolygon*>> _conflict_polygon_map;
  // std::map<int, bgi::rtree<RTreeBox, bgi::quadratic<16>>> _layer_to_violation_box_tree;
  // std::vector<std::pair<DrcRect*, DrcRect*>> _violation_rect_pair_list;

  //-----------------------------------------------
  //-----------------------------------------------
  //----------interact with other operations----------
  bool _interact_with_op = false;
  bool _check_result = true;

  // function
  void checkJog2JogSpacing(std::vector<DrcRect>& jogs_need_to_check_jog2jog_spacing);
  void checkJog2JogSpacing(DrcRect& rect1, DrcRect& rect2);

  bool skipCheck(DrcRect* wid_rect, DrcRect* trigger_rect, DrcRect* result_rect);

  void interceptResultRect_Vertical(DrcRect* wid_rect, DrcRect* trigger_rect, DrcRect& intercept_result_rect);
  void interceptResultRect_Horizontal(DrcRect* wid_rect, DrcRect* trigger_rect, DrcRect& intercept_result_rect);

  bool isJog(DrcRect* result_rect, DrcRect* wid_rect, DrcRect* trigger_rect);
  bool isTriggerRect(DrcRect* wid_rect, DrcRect* result_rect);
  bool isJogOfWidRect(DrcRect* result_rect, DrcRect* wid_rect);

  void checkJogSpacing_up(DrcRect* wid_rect);
  void checkJogSpacing_down(DrcRect* wid_rect);
  void checkJogSpacing_left(DrcRect* wid_rect);
  void checkJogSpacing_right(DrcRect* wid_rect);
  void checkSpacingToTriggerRect(DrcRect* result_rect, DrcRect* trigger_rect);

  void checkTriggerRect_Vertical(DrcRect* wid_rect, DrcRect* trigger_rect);
  void checkTriggerRect_Horizontal(DrcRect* wid_rect, DrcRect* trigger_rect);

  int getPRLRunLength(DrcRect* target_rect, DrcRect* result_rect);
  void getJogQueryBox_up(RTreeBox& query_box, DrcRect* wid_rect, DrcRect* trigger_rect);
  void getParallelQueryBox_up(RTreeBox& query_box, DrcRect* target_rect);
  void getParallelQueryBox_down(RTreeBox& query_box, DrcRect* target_rect);
  void getParallelQueryBox_left(RTreeBox& query_box, DrcRect* wid_rect);
  void getParallelQueryBox_right(RTreeBox& query_box, DrcRect* wid_rect);

  void checkSpacing_Vertical(DrcRect* intercept_result_rect, DrcRect* check_rect, DrcRect* rect,
                             std::vector<DrcRect>& jogs_need_to_check_jog2jog_spacing);
  void checkSpacing_Horizontal(DrcRect* intercept_result_rect, DrcRect* check_rect, DrcRect* rect,
                               std::vector<DrcRect>& jogs_need_to_check_jog2jog_spacing);

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