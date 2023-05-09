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
#ifndef IDRC_SRC_DB_ROUTING_WIDTH_CHECK_H_
#define IDRC_SRC_DB_ROUTING_WIDTH_CHECK_H_

#include <map>
#include <set>

#include "DRCUtil.h"
#include "DrcConfig.h"
#include "DrcDesign.h"
#include "RegionQuery.h"
#include "Tech.h"
namespace idrc {
class RoutingWidthCheck
{
 public:
  // static RoutingWidthCheck* getInstance(DrcConfig* config = nullptr, Tech* tech = nullptr)
  // {
  //   static RoutingWidthCheck instance(config, tech);
  //   return &instance;
  // }
  //----------------------------
  RoutingWidthCheck(DrcConfig* config, Tech* tech, RegionQuery* region_query) { init(config, tech, region_query); }
  RoutingWidthCheck(Tech* tech, RegionQuery* region_query)
  {
    _tech = tech;
    _region_query = region_query;
    _interact_with_op = true;
  }

  RoutingWidthCheck(const RoutingWidthCheck& other) = delete;
  RoutingWidthCheck(RoutingWidthCheck&& other) = delete;
  ~RoutingWidthCheck() {}
  RoutingWidthCheck& operator=(const RoutingWidthCheck& other) = delete;
  RoutingWidthCheck& operator=(RoutingWidthCheck&& other) = delete;

  // getter
  std::map<int, std::vector<DrcSpot>>& get_routing_layer_to_spots_map() { return _routing_layer_to_spots_map; }

  // function
  void checkRoutingWidth(DrcNet* target_net);
  void checkRoutingWidth(const LayerNameToRTreeMap& layer_to_rects_tree_map);

  void reset();
  int get_width_violation_num();

  bool check(DrcNet* target_net);
  bool check(DrcRect* target_rect);

 private:
  DrcConfig* _config;
  Tech* _tech;
  RegionQuery* _region_query;
  std::set<DrcRect*> _checked_rect_list;

  std::map<int, std::vector<DrcSpot>> _routing_layer_to_spots_map;

  //----------interact with other operations----------
  bool _interact_with_op = false;
  bool _check_result = true;

  // function
  void init(DrcConfig* config, Tech* tech, RegionQuery* region_query);

  void checkRoutingWidth(DrcRect* target_rect);
  void checkTargetRectWidth(int layerId, DrcRect* target_rect, int require_width);
  void checkDiagonalLengthOfOverlapRect(int layerId, DrcRect* target_rect, int require_width,
                                        std::vector<std::pair<RTreeBox, DrcRect*>>& query_result);
  RTreeBox getQueryBox(DrcRect* target_rect);
  bool skipCheck(DrcRect* target_rect, DrcRect* result_rect);
  bool isOverlapBoxCoveredByExistedRect(DrcRect* target_rect, DrcRect* result_rect, const DrcRectangle<int>& overlap_rect,
                                        std::vector<std::pair<RTreeBox, DrcRect*>>& query_result);
  void add_spot(int layerId, DrcRect* rect, ViolationType type);
  void add_spot(int layerId, DrcRect* target_rect, DrcRect* result_rect);

  void checkDiagonalLengthOfOverlapBox();
  ////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////interact with iRT
  void checkRoutingWidth(int layerId, DrcRect* target_rect, const RectRTree& rtree);
};
}  // namespace idrc

#endif