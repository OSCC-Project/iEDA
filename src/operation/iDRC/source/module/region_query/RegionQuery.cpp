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
#include "RegionQuery.h"

#include "CornerFillSpacingCheck.hpp"
#include "DrcConfig.h"
#include "EOLSpacingCheck.hpp"

namespace idrc {
void RegionQuery::init(DrcConfig* config, DrcDesign* design)
{
  _config = config;
  _drc_design = design;
  // 现在在IDBWrapper的过程中就初始化RegionQuery中的R树数据了
}

void RegionQuery::queryIntersectsInRoutingLayer(int routingLayerId, RTreeBox query_box,
                                                std::vector<std::pair<RTreeBox, DrcRect*>>& query_result)
{
  // _layer_to_routing_rects_tree_map[routingLayerId].query(bgi::intersects(query_box), std::back_inserter(query_result));
  _layer_to_routing_rects_tree_map[routingLayerId].query(bgi::contains(query_box), std::back_inserter(query_result));
  _layer_to_routing_rects_tree_map[routingLayerId].query(bgi::overlaps(query_box), std::back_inserter(query_result));
  _layer_to_routing_rects_tree_map[routingLayerId].query(bgi::covers(query_box), std::back_inserter(query_result));
  _layer_to_routing_rects_tree_map[routingLayerId].query(bgi::covered_by(query_box), std::back_inserter(query_result));
  // _layer_to_routing_rects_tree_map[routingLayerId].query(bgi::intersects(query_box), std::back_inserter(query_result));
  // _layer_to_routing_rects_tree_map[routingLayerId].query(bgi::disjoint(query_box), std::back_inserter(query_result));
  // _layer_to_routing_rects_tree_map[routingLayerId].query(bgi::within(query_box), std::back_inserter(query_result));
  // _layer_to_fixed_rects_tree_map[routingLayerId].query(bgi::intersects(query_box), std::back_inserter(query_result));
  _layer_to_fixed_rects_tree_map[routingLayerId].query(bgi::contains(query_box), std::back_inserter(query_result));
  _layer_to_fixed_rects_tree_map[routingLayerId].query(bgi::overlaps(query_box), std::back_inserter(query_result));
  _layer_to_fixed_rects_tree_map[routingLayerId].query(bgi::covers(query_box), std::back_inserter(query_result));
  _layer_to_fixed_rects_tree_map[routingLayerId].query(bgi::covered_by(query_box), std::back_inserter(query_result));
  // _layer_to_fixed_rects_tree_map[routingLayerId].query(bgi::intersects(query_box), std::back_inserter(query_result));
  // _layer_to_fixed_rects_tree_map[routingLayerId].query(bgi::disjoint(query_box), std::back_inserter(query_result));
  // _layer_to_fixed_rects_tree_map[routingLayerId].query(bgi::within(query_box), std::back_inserter(query_result));
}

/**
 * @brief query and store results
 *        搜索绕线层上与目标区域相交的所有矩形，并把搜索结果存放于搜索结果列表中
 * @param routingLayerId 绕线层Id
 * @param query_box 搜索区域
 * @param query_result 搜索结果列表
 */
void RegionQuery::queryInRoutingLayer(int routingLayerId, RTreeBox query_box, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result)
{
  searchRoutingRect(routingLayerId, query_box, query_result);
  searchFixedRect(routingLayerId, query_box, query_result);
}

/**
 * @brief query to get enclosures of the cut on a layer
 *
 * @param LayerId
 * @param query_box
 * @param query_result
 */
void RegionQuery::queryEnclosureInRoutingLayer(int LayerId, RTreeBox query_box, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result)
{
  _layer_to_routing_rects_tree_map[LayerId].query(bgi::covers(query_box), std::back_inserter(query_result));
  _layer_to_fixed_rects_tree_map[LayerId].query(bgi::intersects(query_box), std::back_inserter(query_result));
}

/**
 * @brief 搜索绕线层上与目标区域相交的所有通孔与线段矩形，并把搜索结果存放于搜索结果列表中
 *
 * @param routingLayerId 绕线层Id
 * @param query_box 搜索区域
 * @param query_result 搜索结果列表
 */
void RegionQuery::searchRoutingRect(int routingLayerId, RTreeBox query_box, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result)
{
  _layer_to_routing_rects_tree_map[routingLayerId].query(bgi::overlaps(query_box), std::back_inserter(query_result));
}

/**
 * @brief 搜索绕线层上与目标区域相交的所有Pin与Blockage矩形，并把搜索结果存放于搜索结果列表中
 *
 * @param routingLayerId 绕线层Id
 * @param query_box 搜索区域
 * @param query_result 搜索结果列表
 */
void RegionQuery::searchFixedRect(int routingLayerId, RTreeBox query_box, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result)
{
  _layer_to_fixed_rects_tree_map[routingLayerId].query(bgi::overlaps(query_box), std::back_inserter(query_result));
}

// 下面的目前没用到
void RegionQuery::queryInCutLayer(int cutLayerId, RTreeBox query_box, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result)
{
  searchCutRect(cutLayerId, query_box, query_result);
}

void RegionQuery::searchCutRect(int cutLayerId, RTreeBox query_box, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result)
{
  _layer_to_cut_rects_tree_map[cutLayerId].query(bgi::overlaps(query_box), std::back_inserter(query_result));
}

void RegionQuery::queryInMaxScope(int layer_id, RTreeBox check_rect,
                                  std::map<void*, std::map<ScopeType, std::vector<DrcRect*>>>& query_result)
{
  std::vector<std::pair<RTreeBox, DrcRect*>> origin_result;
  _layer_to_routing_max_region_tree_map[layer_id].query(bgi::overlaps(check_rect), std::back_inserter(origin_result));
  for (auto [rtree_box, drc_rect] : origin_result) {
    query_result[drc_rect->get_scope_owner()][drc_rect->getScopeType()].push_back(drc_rect);
  }
}

void RegionQuery::queryInMinScope(int layer_id, RTreeBox check_rect,
                                  std::map<void*, std::map<ScopeType, std::vector<DrcRect*>>>& query_result)
{
  std::vector<std::pair<RTreeBox, DrcRect*>> origin_result;
  _layer_to_routing_min_region_tree_map[layer_id].query(bgi::overlaps(check_rect), std::back_inserter(origin_result));
  for (auto [rtree_box, drc_rect] : origin_result) {
    query_result[drc_rect->get_scope_owner()][drc_rect->getScopeType()].push_back(drc_rect);
  }
}

/**
 * @brief transform drc rect to RTreeBox
 *
 * @param rect
 * @return RTreeBox
 */
RTreeBox RegionQuery::getRTreeBox(DrcRect* rect)
{
  RTreePoint leftBottom(rect->get_left(), rect->get_bottom());
  RTreePoint rightTop(rect->get_right(), rect->get_top());
  return RTreeBox(leftBottom, rightTop);
}

/**
 * @brief 将对应金属层的线段或通孔矩形加入R树
 *
 * @param routingLayerId 金属层Id
 * @param rect 线段或通孔矩形
 */
void RegionQuery::add_routing_rect_to_rtree(int routingLayerId, DrcRect* rect)
{
  RTreeBox rTreeBox = getRTreeBox(rect);
  _layer_to_routing_rects_tree_map[routingLayerId].insert(std::make_pair(rTreeBox, rect));
}

// void RegionQuery::add_routing_rect_to_api_rtree(int routingLayerId, DrcRect* rect)
// {
//   RTreeBox rTreeBox = getRTreeBox(rect);
//   _layer_to_routing_rects_tree_map[routingLayerId].insert(std::make_pair(rTreeBox, rect));
//   add_routing_rect_to_min_rtree();
//   add_routing_rect_to_max_rtree();
// }
/**
 * @brief 将对应金属层的Pin或Blockage矩形加入R树
 *
 * @param routingLayerId 金属层Id
 * @param rect 线段或通孔矩形
 */
void RegionQuery::add_fixed_rect_to_rtree(int routingLayerId, DrcRect* rect)
{
  RTreeBox rTreeBox = getRTreeBox(rect);
  _layer_to_fixed_rects_tree_map[routingLayerId].insert(std::make_pair(rTreeBox, rect));
}

void RegionQuery::add_cut_rect_to_rtree(int cutLayerId, DrcRect* rect)
{
  RTreeBox rTreeBox = getRTreeBox(rect);
  _layer_to_cut_rects_tree_map[cutLayerId].insert(std::make_pair(rTreeBox, rect));
}

// // check
// bool RegionQuery::isExistingRectangleInRoutingRTree(int layerId, const DrcRectangle<int>& rectangle)
// {
//   RTreeBox search_box = DRCUtil::getRTreeBox(rectangle);
//   std::vector<std::pair<RTreeBox, DrcRect*>> query_result_list;
//   searchRoutingRect(layerId, search_box, query_result_list);
//   if (query_result_list.size() == 0) {
//     return false;
//   }
//   for (auto result_pair : query_result_list) {
//     DrcRect* result = result_pair.second;
//     if (DRCUtil::areTwoEqualRectangles(rectangle, result)) {
//       return true;
//     }
//   }
//   return false;
// }

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
// DEBUG
void RegionQuery::printRoutingRectsRTree()
{
  std::cout << "[PRINT routing rtree rect ]:::::::::::::::::::::::::" << std::endl;
  for (auto& [layerId, rtree] : _layer_to_routing_rects_tree_map) {
    std::cout << "[routing rtree rect on layer] : " << layerId << std::endl;
    for (auto it = rtree.begin(); it != rtree.end(); ++it) {
      DrcRect* drc_rect = it->second;
      std::cout << "LeftBottom :: "
                << "(" << drc_rect->get_left() << "," << drc_rect->get_bottom() << ")"
                << " RightTop :: "
                << "(" << drc_rect->get_right() << "," << drc_rect->get_top() << ")" << std::endl;
    }
  }
  std::cout << "-----------------------------------" << std::endl;
}
void RegionQuery::printFixedRectsRTree()
{
  std::cout << "[PRINT fixed rtree rect ]:::::::::::::::::::::::::" << std::endl;
  for (auto& [layerId, rtree] : _layer_to_fixed_rects_tree_map) {
    std::cout << "[fixed rtree rect on layer] : " << layerId << std::endl;
    for (auto it = rtree.begin(); it != rtree.end(); ++it) {
      DrcRect* drc_rect = it->second;
      std::cout << "LeftBottom :: "
                << "(" << drc_rect->get_left() << "," << drc_rect->get_bottom() << ")"
                << " RightTop :: "
                << "(" << drc_rect->get_right() << "," << drc_rect->get_top() << ")" << std::endl;
    }
  }
  std::cout << "-----------------------------------" << std::endl;
}

bool RegionQuery::addCutSpacingViolation(DrcRect* target_rect, DrcRect* result_rect)
{
  if (target_rect > result_rect) {
    return _cut_spacing_vio_set.insert(std::make_pair(target_rect, result_rect)).second;
  } else {
    return _cut_spacing_vio_set.insert(std::make_pair(result_rect, target_rect)).second;
  }
  return false;
}

bool RegionQuery::addCutDiffLayerSpacingViolation(DrcRect* target_rect, DrcRect* result_rect)
{
  if (target_rect > result_rect) {
    return _cut_diff_layer_spacing_vio_set.insert(std::make_pair(target_rect, result_rect)).second;
  } else {
    return _cut_diff_layer_spacing_vio_set.insert(std::make_pair(result_rect, target_rect)).second;
  }
  return false;
}

void RegionQuery::initPrlVioSpot()
{
  for (auto& [layer_id, _rtree] : _layer_to_prl_vio_box_tree) {
    for (auto it = _rtree.begin(); it != _rtree.end(); ++it) {
      RTreeBox box = *it;
      DrcViolationSpot* spot = new DrcViolationSpot();
      spot->set_layer_id(layer_id);
      spot->set_layer_name(_tech->getRoutingLayerNameById(layer_id));
      // spot->set_net_id(target_rect->get_net_id());
      spot->set_vio_type(ViolationType::kRoutingSpacing);
      spot->setCoordinate(box.min_corner().x(), box.min_corner().y(), box.max_corner().x(), box.max_corner().y());
      _prl_run_length_spacing_spot_list.emplace_back(spot);
    }
  }
  std::cout << "prl_run_length::" << _prl_run_length_spacing_spot_list.size() << std::endl;
}

void RegionQuery::initMetalEOLVioSpot()
{
  for (auto& [layer_id, _rtree] : _layer_to_metal_EOL_vio_box_tree) {
    for (auto it = _rtree.begin(); it != _rtree.end(); ++it) {
      RTreeBox box = *it;
      DrcViolationSpot* spot = new DrcViolationSpot();
      spot->set_layer_id(layer_id);
      spot->set_layer_name(_tech->getRoutingLayerNameById(layer_id));
      // spot->set_net_id(target_rect->get_net_id());
      spot->set_vio_type(ViolationType::kShort);
      spot->setCoordinate(box.min_corner().x(), box.min_corner().y(), box.max_corner().x(), box.max_corner().y());
      _metal_eol_spacing_spot_list.emplace_back(spot);
    }
  }
}

void RegionQuery::initShortVioSpot()
{
  for (auto& [layer_id, _rtree] : _layer_to_short_vio_box_tree) {
    for (auto it = _rtree.begin(); it != _rtree.end(); ++it) {
      RTreeBox box = *it;
      DrcViolationSpot* spot = new DrcViolationSpot();
      spot->set_layer_id(layer_id);
      spot->set_layer_name(_tech->getRoutingLayerNameById(layer_id));
      // spot->set_net_id(target_rect->get_net_id());
      spot->set_vio_type(ViolationType::kShort);
      spot->setCoordinate(box.min_corner().x(), box.min_corner().y(), box.max_corner().x(), box.max_corner().y());
      _short_vio_spot_list.emplace_back(spot);
    }
  }
}

bool RegionQuery::addCutEOLSpacingViolation(DrcRect* target_rect, DrcRect* result_rect)
{
  if (target_rect > result_rect) {
    return _cut_eol_spacing_vio_set.insert(std::make_pair(target_rect, result_rect)).second;
  } else {
    return _cut_eol_spacing_vio_set.insert(std::make_pair(result_rect, target_rect)).second;
  }
  return false;
}

void RegionQuery::addPRLRunLengthSpacingViolation(int layer_id, RTreeBox span_box)
{
  std::vector<RTreeBox> query_result;
  _layer_to_prl_vio_box_tree[layer_id].query(bgi::intersects(span_box), std::back_inserter(query_result));
  for (auto& box : query_result) {
    // if (DRCUtil::isPenetratedIntersected(box, span_box)) {
    int lb_x = std::min(box.min_corner().get<0>(), span_box.min_corner().get<0>());
    int lb_y = std::min(box.min_corner().get<1>(), span_box.min_corner().get<1>());
    int rt_x = std::max(box.max_corner().get<0>(), span_box.max_corner().get<0>());
    int rt_y = std::max(box.max_corner().get<1>(), span_box.max_corner().get<1>());
    span_box.min_corner().set<0>(lb_x);
    span_box.min_corner().set<1>(lb_y);
    span_box.max_corner().set<0>(rt_x);
    span_box.max_corner().set<1>(rt_y);
    _layer_to_prl_vio_box_tree[layer_id].remove(box);
    // }
  }
  _layer_to_prl_vio_box_tree[layer_id].insert(span_box);
}

bool RegionQuery::addPRLRunLengthSpacingViolation(DrcRect* target_rect, DrcRect* result_rect)
{
  if (target_rect > result_rect) {
    return _prl_spacing_vio_set.insert(std::make_pair(target_rect, result_rect)).second;
  } else {
    return _prl_spacing_vio_set.insert(std::make_pair(result_rect, target_rect)).second;
  }
  return false;
}

void RegionQuery::addMetalEOLSpacingViolation(int layer_id, RTreeBox span_box)
{
  std::vector<RTreeBox> query_result;
  _layer_to_metal_EOL_vio_box_tree[layer_id].query(bgi::intersects(span_box), std::back_inserter(query_result));
  for (auto& box : query_result) {
    // if (DRCUtil::isPenetratedIntersected(box, span_box)) {
    int lb_x = std::min(box.min_corner().get<0>(), span_box.min_corner().get<0>());
    int lb_y = std::min(box.min_corner().get<1>(), span_box.min_corner().get<1>());
    int rt_x = std::max(box.max_corner().get<0>(), span_box.max_corner().get<0>());
    int rt_y = std::max(box.max_corner().get<1>(), span_box.max_corner().get<1>());
    span_box.min_corner().set<0>(lb_x);
    span_box.min_corner().set<1>(lb_y);
    span_box.max_corner().set<0>(rt_x);
    span_box.max_corner().set<1>(rt_y);
    _layer_to_metal_EOL_vio_box_tree[layer_id].remove(box);
    // }
  }
  _layer_to_metal_EOL_vio_box_tree[layer_id].insert(span_box);
}

void RegionQuery::addShortViolation(int layer_id, RTreeBox span_box)
{
  std::vector<RTreeBox> query_result;
  _layer_to_short_vio_box_tree[layer_id].query(bgi::intersects(span_box), std::back_inserter(query_result));
  for (auto& box : query_result) {
    // if (DRCUtil::isPenetratedIntersected(box, span_box)) {
    int lb_x = std::min(box.min_corner().get<0>(), span_box.min_corner().get<0>());
    int lb_y = std::min(box.min_corner().get<1>(), span_box.min_corner().get<1>());
    int rt_x = std::max(box.max_corner().get<0>(), span_box.max_corner().get<0>());
    int rt_y = std::max(box.max_corner().get<1>(), span_box.max_corner().get<1>());
    span_box.min_corner().set<0>(lb_x);
    span_box.min_corner().set<1>(lb_y);
    span_box.max_corner().set<0>(rt_x);
    span_box.max_corner().set<1>(rt_y);
    _layer_to_short_vio_box_tree[layer_id].remove(box);
    // }
  }
  _layer_to_short_vio_box_tree[layer_id].insert(span_box);
}

bool RegionQuery::addShortViolation(DrcRect* target_rect, DrcRect* result_rect)
{
  if (target_rect > result_rect) {
    return _short_vio_set.insert(std::make_pair(target_rect, result_rect)).second;
  } else {
    return _short_vio_set.insert(std::make_pair(result_rect, target_rect)).second;
  }
  return false;
}

void RegionQuery::addViolation(ViolationType vio_type)
{
  switch (vio_type) {
    case ViolationType::kCutShort:
      break;
    case ViolationType::kEnd2EndEOLSpacing:
      break;
    case ViolationType::kNone:
      break;

    case ViolationType::kDensity:
      break;
    case ViolationType::kShort:
      ++_short_count;
      break;
    case ViolationType::kRoutingSpacing:
      ++_common_spacing_count;
      break;
    case ViolationType::kNotchSpacing:
      ++_notch_spacing_count;
      break;
    case ViolationType::kJogSpacing:
      ++_jog_spacing_count;
      break;
    case ViolationType::kCornerFillingSpacing:
      ++_corner_fill_spacing_count;
      break;
    case ViolationType::kEOLSpacing:
      ++_eol_spacing_count;
      break;
    case ViolationType::kMinStep:
      ++_minstep_count;
      break;
    case ViolationType::kEnclosedArea:
      ++_min_hole_count;
      break;
    case ViolationType::kRoutingArea:
      ++_area_count;
      break;
    case ViolationType::kRoutingWidth:
      ++_width_count;
      break;
    case ViolationType::kEnclosure:
      ++_common_enclosure_count;
      break;
    case ViolationType::kEnclosureEdge:
      ++_egde_enclosure_count;
      break;
    case ViolationType::kCutDiffLayerSpacing:
      ++_cut_diff_layer_spacing_count;
      break;
    case ViolationType::kCutEOLSpacing:
      ++_cut_eol_spacing_count;
      break;
    case ViolationType::kCutSpacing:
      ++_cut_common_spacing_count;
      break;
  }
}

void RegionQuery::getRegionDetailReport(std::map<std::string, std::vector<DrcViolationSpot*>>& vio_map)
{
  vio_map.insert(std::make_pair("Cut EOL Spacing", _cut_eol_spacing_spot_list));
  vio_map.insert(std::make_pair("Cut Spacing", _cut_spacing_spot_list));
  vio_map.insert(std::make_pair("Cut Enclosure", _cut_enclosure_spot_list));
  initMetalEOLVioSpot();
  vio_map.insert(std::make_pair("Metal EOL Spacing", _metal_eol_spacing_spot_list));
  initShortVioSpot();
  vio_map.insert(std::make_pair("Metal Short", _short_vio_spot_list));
  initPrlVioSpot();
  vio_map.insert(std::make_pair("Metal Parallel Run Length Spacing", _prl_run_length_spacing_spot_list));
  vio_map.insert(std::make_pair("Metal Notch Spacing", _metal_notch_spacing_spot_list));
  vio_map.insert(std::make_pair("MinStep", _min_step_spot_list));
  vio_map.insert(std::make_pair("Minimal Area", _min_area_spot_list));
}

int RegionQuery::get_prl_spacing_vio_nums()
{
  int res = 0;

  for (auto& _rtree : _layer_to_prl_vio_box_tree) {
    // std::cout << "layer" << _rtree.first << ":" << _rtree.second.size() << std::endl;
    res += _rtree.second.size();
  }
  return res;
}

int RegionQuery::get_short_vio_nums()
{
  int res = 0;

  for (auto& _rtree : _layer_to_short_vio_box_tree) {
    // std::cout << "layer" << _rtree.first << ":" << _rtree.second.size() << std::endl;
    res += _rtree.second.size();
  }
  return res;
}

int RegionQuery::get_metal_eol_vio_nums()
{
  int res = 0;

  for (auto& _rtree : _layer_to_metal_EOL_vio_box_tree) {
    // std::cout << "layer" << _rtree.first << ":" << _rtree.second.size() << std::endl;
    res += _rtree.second.size();
  }
  return res;
}

void RegionQuery::getRegionReport(std::map<std::string, int>& viotype_to_nums_map)
{
  viotype_to_nums_map.insert(std::make_pair("Cut EOL Spacing", _cut_eol_spacing_vio_set.size()));
  viotype_to_nums_map.insert(std::make_pair("Cut Different Layer Spacing", _cut_diff_layer_spacing_vio_set.size()));
  viotype_to_nums_map.insert(std::make_pair("Cut Spacing", _cut_spacing_vio_set.size()));
  viotype_to_nums_map.insert(std::make_pair("Cut Enclosure", _common_enclosure_count));
  viotype_to_nums_map.insert(std::make_pair("Cut EnclosureEdge", _egde_enclosure_count));
  // viotype_to_nums_map.insert(std::make_pair("Cut EOL Enclosure", ));

  viotype_to_nums_map.insert(std::make_pair("Metal Corner Filling Spacing", _corner_fill_spacing_count));
  viotype_to_nums_map.insert(std::make_pair("Metal EOL Spacing", get_metal_eol_vio_nums()));
  viotype_to_nums_map.insert(std::make_pair("Metal JogToJog Spacing", _jog_spacing_count));
  viotype_to_nums_map.insert(std::make_pair("Metal Short", get_short_vio_nums()));
  viotype_to_nums_map.insert(std::make_pair("Metal Parallel Run Length Spacing", get_prl_spacing_vio_nums()));
  // viotype_to_nums_map.insert(std::make_pair("Metal Short", _short_count));
  // viotype_to_nums_map.insert(std::make_pair("Metal Parallel Run Length Spacing", _common_spacing_count));
  viotype_to_nums_map.insert(std::make_pair("Metal Notch Spacing", _notch_spacing_count));
  viotype_to_nums_map.insert(std::make_pair("MinStep", _minstep_count));
  viotype_to_nums_map.insert(std::make_pair("MinHole", _min_hole_count));
  viotype_to_nums_map.insert(std::make_pair("Minimal Area", _area_count));
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
// drc edge Not currently used

void RegionQuery::addPolyEdge_NotAddToRegion(DrcPoly* new_poly)
{
  initPolyOuterEdges_NotAddToRegion(new_poly->getNet(), new_poly, new_poly->getPolygon(), new_poly->get_layer_id());
  // pending

  auto polygon = new_poly->getPolygon();

  for (auto holeIt = polygon->begin_holes(); holeIt != polygon->end_holes(); holeIt++) {
    auto& hole_poly = *holeIt;

    initPolyInnerEdges_NotAddToRegion(new_poly->getNet(), new_poly, hole_poly, new_poly->get_layer_id());
  }
}

void RegionQuery::initPolyOuterEdges_NotAddToRegion(DrcNet* net, DrcPoly* poly, DrcPolygon* polygon, int layer_id)
{
  DrcCoordinate bp(-1, -1), ep(-1, -1), firstPt(-1, -1);
  BoostPoint bp1, ep1, firstPt1;
  std::vector<std::unique_ptr<DrcEdge>> tmpEdges;
  // skip the first pt

  auto outerIt = polygon->begin();

  bp.set((*outerIt).x(), (*outerIt).y());

  bp1 = *outerIt;
  firstPt.set((*outerIt).x(), (*outerIt).y());
  firstPt1 = *outerIt;
  outerIt++;
  // loop from second to last pt (n-1) edges

  for (; outerIt != polygon->end(); outerIt++) {
    ep.set((*outerIt).x(), (*outerIt).y());
    ep1 = *outerIt;
    // auto edge = make_unique<DrcEdge>();
    std::unique_ptr<DrcEdge> edge(new DrcEdge);
    edge->set_layer_id(layer_id);
    edge->addToPoly(poly);
    edge->addToNet(net);
    // edge->setPoints(bp, ep);
    edge->setSegment(bp1, ep1);
    edge->setDir();
    edge->set_is_fixed(false);

    if (!tmpEdges.empty()) {
      edge->setPrevEdge(tmpEdges.back().get());
      tmpEdges.back()->setNextEdge(edge.get());
    }
    tmpEdges.push_back(std::move(edge));
    bp.set(ep);
    bp1 = ep1;
  }

  // last edge
  auto edge = std::make_unique<DrcEdge>();
  edge->set_layer_id(layer_id);
  edge->addToPoly(poly);
  edge->addToNet(net);
  // edge->setPoints(bp, firstPt);
  edge->setSegment(bp1, firstPt1);
  edge->setDir();
  edge->set_is_fixed(false);
  edge->setPrevEdge(tmpEdges.back().get());
  tmpEdges.back()->setNextEdge(edge.get());
  // set first edge
  tmpEdges.front()->setPrevEdge(edge.get());
  edge->setNextEdge(tmpEdges.front().get());

  tmpEdges.push_back(std::move(edge));
  // add to polygon edges
  poly->addEdges(tmpEdges);
}

void RegionQuery::initPolyInnerEdges_NotAddToRegion(DrcNet* net, DrcPoly* poly, const BoostPolygon& hole_poly, int layer_id)
{
  DrcCoordinate bp(-1, -1), ep(-1, -1), firstPt(-1, -1);
  BoostPoint bp1, ep1, firstPt1;
  std::vector<std::unique_ptr<DrcEdge>> tmpEdges;
  // skip the first pt
  auto innerIt = hole_poly.begin();
  bp.set((*innerIt).x(), (*innerIt).y());
  bp1 = *innerIt;
  firstPt.set((*innerIt).x(), (*innerIt).y());
  firstPt1 = *innerIt;
  innerIt++;
  // loop from second to last pt (n-1) edges
  for (; innerIt != hole_poly.end(); innerIt++) {
    ep.set((*innerIt).x(), (*innerIt).y());
    ep1 = *innerIt;
    auto edge = std::make_unique<DrcEdge>();
    edge->set_layer_id(layer_id);
    edge->addToPoly(poly);
    edge->addToNet(net);
    edge->setDir();
    edge->setSegment(bp1, ep1);

    edge->set_is_fixed(false);
    if (!tmpEdges.empty()) {
      edge->setPrevEdge(tmpEdges.back().get());
      tmpEdges.back()->setNextEdge(edge.get());
    }
    tmpEdges.push_back(std::move(edge));
    bp.set(ep);
    bp1 = ep1;
  }
  auto edge = std::make_unique<DrcEdge>();
  edge->set_layer_id(layer_id);
  edge->addToPoly(poly);
  edge->addToNet(net);
  edge->setSegment(bp1, firstPt1);
  edge->setDir();
  edge->set_is_fixed(false);
  edge->setPrevEdge(tmpEdges.back().get());
  tmpEdges.back()->setNextEdge(edge.get());
  // set first edge
  tmpEdges.front()->setPrevEdge(edge.get());
  edge->setNextEdge(tmpEdges.front().get());

  tmpEdges.push_back(std::move(edge));
  // add to polygon edges
  poly->addEdges(tmpEdges);
}

RTreeSegment RegionQuery::getRTreeSegment(DrcEdge* drcEdge)
{
  RTreePoint point1(drcEdge->get_min_x(), drcEdge->get_min_y());
  RTreePoint point2(drcEdge->get_max_x(), drcEdge->get_max_y());
  return RTreeSegment(point1, point2);
}

void RegionQuery::add_routing_edge_to_rtree(int routingLayerId, DrcEdge* edge)
{
  RTreeSegment rTreeSegment = getRTreeSegment(edge);
  _layer_to_routing_edges[routingLayerId].insert(std::make_pair(rTreeSegment, edge));
}

void RegionQuery::add_block_edge_to_rtree(int routingLayerId, DrcEdge* edge)
{
  RTreeSegment rTreeSegment = getRTreeSegment(edge);
  _layer_to_block_edges[routingLayerId].insert(std::make_pair(rTreeSegment, edge));
}

void RegionQuery::queryEdgeInRoutingLayer(int routingLayerId, RTreeBox query_box, std::vector<std::pair<RTreeSegment, DrcEdge*>>& result)
{
  searchRoutingEdge(routingLayerId, query_box, result);
  searchBlockEdge(routingLayerId, query_box, result);
}

void RegionQuery::searchRoutingEdge(int routingLayerId, RTreeBox query_box, std::vector<std::pair<RTreeSegment, DrcEdge*>>& result)
{
  _layer_to_routing_edges[routingLayerId].query(bgi::intersects(query_box), std::back_inserter(result));
}

void RegionQuery::searchBlockEdge(int routingLayerId, RTreeBox query_box, std::vector<std::pair<RTreeSegment, DrcEdge*>>& result)
{
  _layer_to_block_edges[routingLayerId].query(bgi::intersects(query_box), std::back_inserter(result));
}

void RegionQuery::addDrcRect(DrcRect* drc_rect, Tech* tech)
{
  RTreeBox rTreeBox = getRTreeBox(drc_rect);
  // 只支持routing
  int layer_id = drc_rect->get_layer_id();
  if (drc_rect->get_owner_type() == RectOwnerType::kRoutingMetal) {
    _layer_to_routing_rects_tree_map[layer_id].insert(std::make_pair(rTreeBox, drc_rect));
    _routing_rect_set.insert(drc_rect);
  }
  if (drc_rect->get_owner_type() == RectOwnerType::kViaCut) {
    _layer_to_cut_rects_tree_map[layer_id].insert(std::make_pair(rTreeBox, drc_rect));
    _cut_rect_set.insert(drc_rect);
  }
  // 添加与矩形相关的影响范围
  addRectScope(drc_rect, tech);
}

void RegionQuery::addScopeToMaxScopeRTree(DrcRect* scope)
{
  int layer_id = scope->get_layer_id();
  RTreeBox rTreeBox = getRTreeBox(scope);
  _layer_to_routing_max_region_tree_map[layer_id].insert(std::make_pair(rTreeBox, scope));
}

void RegionQuery::addScopeToMinScopeRTree(DrcRect* scope)
{
  int layer_id = scope->get_layer_id();
  RTreeBox rTreeBox = getRTreeBox(scope);
  _layer_to_routing_min_region_tree_map[layer_id].insert(std::make_pair(rTreeBox, scope));
}

void RegionQuery::addRectScope(DrcRect* drc_rect, Tech* tech)
{
  DrcRect* common_spacing_min_region = new DrcRect();
  getCommonSpacingMinRegion(common_spacing_min_region, drc_rect, tech);
  addScopeToMinScopeRTree(common_spacing_min_region);
  DrcRect* common_spacing_max_region = new DrcRect();
  getCommonSpacingMaxRegion(common_spacing_max_region, drc_rect, tech);
  addScopeToMaxScopeRTree(common_spacing_max_region);
}

void RegionQuery::getCommonSpacingMinRegion(DrcRect* common_spacing_min_region, DrcRect* drc_rect, Tech* tech)
{
  int layer_id = drc_rect->get_layer_id();
  int net_id = drc_rect->get_net_id();
  int spacing = tech->get_drc_routing_layer_list()[layer_id]->get_spacing_table()->get_parallel()->get_spacing(0, 0);
  int lb_x = drc_rect->get_left() - spacing;
  int lb_y = drc_rect->get_bottom() - spacing;
  int rt_x = drc_rect->get_right() + spacing;
  int rt_y = drc_rect->get_top() + spacing;
  common_spacing_min_region->set_layer_id(layer_id);
  common_spacing_min_region->set_coordinate(lb_x, lb_y, rt_x, rt_y);
  common_spacing_min_region->setScopeType(ScopeType::Common);
  // common_spacing_min_region->set_owner_type(RectOwnerType::kCommonRegion);
  common_spacing_min_region->set_net_id(net_id);
  common_spacing_min_region->set_scope_owner(drc_rect);
  drc_rect->set_min_scope(common_spacing_min_region);
}

void RegionQuery::getCommonSpacingMaxRegion(DrcRect* common_spacing_max_region, DrcRect* drc_rect, Tech* tech)
{
  int layer_id = drc_rect->get_layer_id();
  int net_id = drc_rect->get_net_id();
  int width = drc_rect->getWidth();
  int length = drc_rect->getLength();
  int spacing = tech->get_drc_routing_layer_list()[layer_id]->get_spacing_table()->get_parallel()->get_spacing(width, length);
  int lb_x = drc_rect->get_left() - spacing;
  int lb_y = drc_rect->get_bottom() - spacing;
  int rt_x = drc_rect->get_right() + spacing;
  int rt_y = drc_rect->get_top() + spacing;
  common_spacing_max_region->set_layer_id(layer_id);
  common_spacing_max_region->set_coordinate(lb_x, lb_y, rt_x, rt_y);
  common_spacing_max_region->set_owner_type(RectOwnerType::kCommonRegion);
  common_spacing_max_region->set_net_id(net_id);
  common_spacing_max_region->set_scope_owner(drc_rect);
  drc_rect->set_max_scope(common_spacing_max_region);
}

void RegionQuery::removeDrcRect(DrcRect* drc_rect)
{
  RTreeBox rTreeBox = getRTreeBox(drc_rect);
  // 只支持routing
  int layer_id = drc_rect->get_layer_id();
  if (drc_rect->get_owner_type() == RectOwnerType::kRoutingMetal) {
    if (_layer_to_routing_rects_tree_map[layer_id].remove(std::make_pair(rTreeBox, drc_rect)) == 0) {
      std::cout << "[DrcAPI Warning]:rect is not exist,delete failed" << std::endl;
      return;
    }
    _routing_rect_set.erase(drc_rect);
  }
  if (drc_rect->get_owner_type() == RectOwnerType::kViaCut) {
    if (_layer_to_cut_rects_tree_map[layer_id].remove(std::make_pair(rTreeBox, drc_rect)) == 0) {
      std::cout << "[DrcAPI Warning]:rect is not exist,delete failed" << std::endl;
      return;
    }
    _cut_rect_set.erase(drc_rect);
  }
  // 删除与矩形相关的影响范围
  removeRectScope(drc_rect);
}

void RegionQuery::removeRectScope(DrcRect* drc_rect)
{
  auto drc_rect_min_scope = drc_rect->get_min_scope();
  auto drc_rect_max_scope = drc_rect->get_max_scope();
  removeFromMaxScopeRTree(drc_rect_max_scope);
  removeFromMinScopeRTree(drc_rect_min_scope);
}

void RegionQuery::getIntersectPoly(std::set<DrcPoly*>& intersect_poly_set, std::vector<DrcRect*> drc_rect_list)
{
  std::vector<std::pair<RTreeSegment, DrcEdge*>> query_result;
  for (auto& drc_rect : drc_rect_list) {
    if (drc_rect->get_owner_type() == RectOwnerType::kRoutingMetal) {
      int layer_id = drc_rect->get_layer_id();
      RTreeBox boost_box = DRCUtil::getRTreeBox(drc_rect);
      queryEdgeInRoutingLayer(layer_id, boost_box, query_result);
    }
  }
  for (auto& [rtree_segment, drc_edge] : query_result) {
    // 是否需要判断短路？
    intersect_poly_set.insert(drc_edge->get_owner_polygon());
  }
}

void RegionQuery::deletePolyInEdgeRTree(DrcPoly* poly)
{
  int layer_id = poly->get_layer_id();
  for (auto& edges : poly->getEdges()) {
    for (auto& edge : edges) {
      int begin_x = edge->get_begin_x();
      int begin_y = edge->get_begin_y();
      int end_x = edge->get_end_x();
      int end_y = edge->get_end_y();
      RTreePoint begin_point(begin_x, begin_y);
      RTreePoint end_point(end_x, end_y);
      RTreeSegment rtree_seg(begin_point, end_point);
      DrcEdge* edge1 = edge.get();
      _layer_to_routing_edges[layer_id].remove(std::make_pair(rtree_seg, edge1));
      // _layer_to_routing_edges[layer_id].remove(rtree_seg);
      // delete edge;
    }
  }
}
void RegionQuery::deletePolyInNet(DrcPoly* poly)
{
  int net_id = poly->getNetId();
  int layer_id = poly->get_layer_id();
  _region_polys_map[net_id][layer_id].erase(poly);
}

void RegionQuery::removeFromMaxScopeRTree(DrcRect* scope_rect)
{
  int layer_id = scope_rect->get_layer_id();
  RTreeBox scope_rtree_box = DRCUtil::getRTreeBox(scope_rect);
  if (_layer_to_routing_max_region_tree_map[layer_id].remove(std::make_pair(scope_rtree_box, scope_rect)) == 0) {
    std::cout << "[DrcAPI Warning]:max scope rect is not exist,delete failed" << std::endl;
  }
  // delete scope_rect;
}

void RegionQuery::removeFromMinScopeRTree(DrcRect* scope_rect)
{
  int layer_id = scope_rect->get_layer_id();
  RTreeBox scope_rtree_box = DRCUtil::getRTreeBox(scope_rect);

  if (_layer_to_routing_min_region_tree_map[layer_id].remove(std::make_pair(scope_rtree_box, scope_rect)) == 0) {
    std::cout << "[DrcAPI Warning]:min scope rect is not exist,delete failed" << std::endl;
  }

  // delete scope_rect;
}

void RegionQuery::deletePolyInScopeRTree(DrcPoly* poly)
{
  // int layer_id = poly->get_layer_id();
  auto& scope_set = poly->getScopes();
  for (auto scope_rect : scope_set) {
    // 如果是最大影响范围
    if (scope_rect->is_scope_max()) {
      removeFromMaxScopeRTree(scope_rect);
    } else {
      removeFromMinScopeRTree(scope_rect);
    }
  }
}

void RegionQuery::deleteIntersectPoly(std::set<DrcPoly*>& intersect_poly_set)
{
  for (auto& poly : intersect_poly_set) {
    // 删掉Net中的信息
    deletePolyInNet(poly);
    // 删掉RTree中的相关edge
    deletePolyInEdgeRTree(poly);
    // 删掉与之相关的Scope
    deletePolyInScopeRTree(poly);
    // 删掉自身
    //  delete poly;
  }
}

DrcPoly* RegionQuery::rebuildPoly_add(std::set<DrcPoly*>& intersect_poly_set, std::vector<DrcRect*> drc_rect_list)
{
  std::vector<PolygonWithHoles> new_polygon_list;
  for (auto poly : intersect_poly_set) {
    std::vector<bp::rectangle_data<int>> rects;
    auto boost_polygon = poly->getPolygon()->get_polygon();
    new_polygon_list += boost_polygon;
  }
  for (auto rect : drc_rect_list) {
    if (rect->get_owner_type() == RectOwnerType::kRoutingMetal) {
      new_polygon_list += DRCUtil::getBoostRect(rect);
    }
  }
  if (new_polygon_list.size() != 1) {
    std::cout << "[iDRC API] Warning : Rect list merge failed!" << std::endl;
    return nullptr;
  }
  DrcPoly* poly = new DrcPoly();
  int layer_id = drc_rect_list[0]->get_layer_id();
  int net_id = drc_rect_list[0]->get_net_id();
  poly->setNetId(net_id);
  poly->set_layer_id(layer_id);
  DrcPolygon* polygon = new DrcPolygon(new_polygon_list[0], layer_id, poly, net_id);
  poly->setPolygon(polygon);
  return poly;
}

std::vector<DrcPoly*> RegionQuery::rebuildPoly_del(std::set<DrcPoly*>& intersect_poly_set, std::vector<DrcRect*> drc_rect_list)
{
  std::vector<DrcPoly*> rebuild_poly;
  std::vector<PolygonWithHoles> new_polygon_list;
  for (auto poly : intersect_poly_set) {
    std::vector<bp::rectangle_data<int>> rects;
    auto boost_polygon = poly->getPolygon()->get_polygon();
    new_polygon_list += boost_polygon;
  }
  for (auto rect : drc_rect_list) {
    if (rect->get_owner_type() == RectOwnerType::kRoutingMetal) {
      new_polygon_list -= DRCUtil::getBoostRect(rect);
    }
  }
  // 有可能会分裂吗？
  // if (new_polygon_list.size() > 1) {
  //   std::cout << "[iDRC API] Warning : Rect list merge failed!" << std::endl;
  // }
  for (auto new_polygon : new_polygon_list) {
    // DrcPoly* poly = new DrcPoly(new_polygon, drc_rect_list[0]->get_layer_id(), drc_rect_list[0]->get_net_id());
    DrcPoly* poly = new DrcPoly();
    int layer_id = drc_rect_list[0]->get_layer_id();
    int net_id = drc_rect_list[0]->get_net_id();
    poly->setNetId(net_id);
    poly->set_layer_id(layer_id);
    DrcPolygon* polygon = new DrcPolygon(new_polygon, layer_id, poly, net_id);
    poly->setPolygon(polygon);
    rebuild_poly.push_back(poly);
  }
  return rebuild_poly;
}

void RegionQuery::addPolyEdge(DrcPoly* new_poly)
{
  initPolyOuterEdges(new_poly->getNet(), new_poly, new_poly->getPolygon(), new_poly->get_layer_id());
  // pending

  auto polygon = new_poly->getPolygon();

  for (auto holeIt = polygon->begin_holes(); holeIt != polygon->end_holes(); holeIt++) {
    auto& hole_poly = *holeIt;

    initPolyInnerEdges(new_poly->getNet(), new_poly, hole_poly, new_poly->get_layer_id());
  }
}

void RegionQuery::initPolyOuterEdges(DrcNet* net, DrcPoly* poly, DrcPolygon* polygon, int layer_id)
{
  DrcCoordinate bp(-1, -1), ep(-1, -1), firstPt(-1, -1);
  BoostPoint bp1, ep1, firstPt1;
  std::vector<std::unique_ptr<DrcEdge>> tmpEdges;
  // skip the first pt

  auto outerIt = polygon->begin();

  bp.set((*outerIt).x(), (*outerIt).y());

  bp1 = *outerIt;
  firstPt.set((*outerIt).x(), (*outerIt).y());
  firstPt1 = *outerIt;
  outerIt++;
  // loop from second to last pt (n-1) edges

  for (; outerIt != polygon->end(); outerIt++) {
    ep.set((*outerIt).x(), (*outerIt).y());
    ep1 = *outerIt;
    // auto edge = make_unique<DrcEdge>();
    std::unique_ptr<DrcEdge> edge(new DrcEdge);
    edge->set_layer_id(layer_id);
    edge->addToPoly(poly);
    edge->addToNet(net);
    // edge->setPoints(bp, ep);
    edge->setSegment(bp1, ep1);
    edge->setDir();
    edge->set_is_fixed(false);
    add_routing_edge_to_rtree(layer_id, edge.get());

    if (!tmpEdges.empty()) {
      edge->setPrevEdge(tmpEdges.back().get());
      tmpEdges.back()->setNextEdge(edge.get());
    }
    tmpEdges.push_back(std::move(edge));
    bp.set(ep);
    bp1 = ep1;
  }

  // last edge
  auto edge = std::make_unique<DrcEdge>();
  edge->set_layer_id(layer_id);
  edge->addToPoly(poly);
  edge->addToNet(net);
  // edge->setPoints(bp, firstPt);
  edge->setSegment(bp1, firstPt1);
  edge->setDir();
  edge->set_is_fixed(false);
  edge->setPrevEdge(tmpEdges.back().get());
  tmpEdges.back()->setNextEdge(edge.get());
  // set first edge
  tmpEdges.front()->setPrevEdge(edge.get());
  edge->setNextEdge(tmpEdges.front().get());
  add_routing_edge_to_rtree(layer_id, edge.get());

  tmpEdges.push_back(std::move(edge));
  // add to polygon edges
  poly->addEdges(tmpEdges);
}

void RegionQuery::initPolyInnerEdges(DrcNet* net, DrcPoly* poly, const BoostPolygon& hole_poly, int layer_id)
{
  DrcCoordinate bp(-1, -1), ep(-1, -1), firstPt(-1, -1);
  BoostPoint bp1, ep1, firstPt1;
  std::vector<std::unique_ptr<DrcEdge>> tmpEdges;
  // skip the first pt
  auto innerIt = hole_poly.begin();
  bp.set((*innerIt).x(), (*innerIt).y());
  bp1 = *innerIt;
  firstPt.set((*innerIt).x(), (*innerIt).y());
  firstPt1 = *innerIt;
  innerIt++;
  // loop from second to last pt (n-1) edges
  for (; innerIt != hole_poly.end(); innerIt++) {
    ep.set((*innerIt).x(), (*innerIt).y());
    ep1 = *innerIt;
    auto edge = std::make_unique<DrcEdge>();
    edge->set_layer_id(layer_id);
    edge->addToPoly(poly);
    edge->addToNet(net);
    edge->setDir();
    edge->setSegment(bp1, ep1);
    add_routing_edge_to_rtree(layer_id, edge.get());

    edge->set_is_fixed(false);
    if (!tmpEdges.empty()) {
      edge->setPrevEdge(tmpEdges.back().get());
      tmpEdges.back()->setNextEdge(edge.get());
    }
    tmpEdges.push_back(std::move(edge));
    bp.set(ep);
    bp1 = ep1;
  }
  auto edge = std::make_unique<DrcEdge>();
  edge->set_layer_id(layer_id);
  edge->addToPoly(poly);
  edge->addToNet(net);
  edge->setSegment(bp1, firstPt1);
  edge->setDir();
  edge->set_is_fixed(false);
  add_routing_edge_to_rtree(layer_id, edge.get());
  edge->setPrevEdge(tmpEdges.back().get());
  tmpEdges.back()->setNextEdge(edge.get());
  // set first edge
  tmpEdges.front()->setPrevEdge(edge.get());
  edge->setNextEdge(tmpEdges.front().get());

  tmpEdges.push_back(std::move(edge));
  // add to polygon edges
  poly->addEdges(tmpEdges);
}

void RegionQuery::addPolyToRegionQuery(DrcPoly* new_poly)
{
  // 不需要从net找到poly，只需用net_id,layer_id索引来有序存放poly
  int net_id = new_poly->getNetId();
  int layer_id = new_poly->get_layer_id();
  _region_polys_map[net_id][layer_id].insert(new_poly);

  // if (_region_polys_map.count(net_id) == 0) {
  //   std::map<int, std::vector<DrcPoly*>> layer_to_polys_map;
  //   _region_polys_map.insert(std::make_pair(net_id, layer_to_polys_map));
  //   _region_polys_map[net_id][layer_id].push_back(new_poly);
  // } else {
  //   if (_region_polys_map[net_id].count(layer_id) == 0) {
  //     std::vector<DrcPoly*> single_layer_polys;
  //     _region_polys_map[net_id].insert(std::make_pair(layer_id, single_layer_polys));
  //     _region_polys_map[net_id][layer_id].push_back(new_poly);
  //   } else {
  //     _region_polys_map[net_id][layer_id].push_back(new_poly);
  //   }
  // }
}

void RegionQuery::getEOLSpacingScope(DrcPoly* new_poly, bool is_max)
{
  EOLSpacingCheck* eol_spacing_check = new EOLSpacingCheck(_tech, nullptr);
  // 构造出scope存放在poly之下，并插入到对应的RTree
  eol_spacing_check->addScope(new_poly, is_max, this);
  delete eol_spacing_check;
}

void RegionQuery::getCornerFillSpacingScope(DrcPoly* new_poly)
{
  CornerFillSpacingCheck* cornerfill_check = new CornerFillSpacingCheck(_tech, nullptr);
  cornerfill_check->addScope(new_poly, this);
  //
  delete cornerfill_check;
}

void RegionQuery::addPolyMinScopes(DrcPoly* new_poly)
{
  getEOLSpacingScope(new_poly, true);
  getCornerFillSpacingScope(new_poly);
}

void RegionQuery::addPolyMaxScopes(DrcPoly* new_poly)
{
  getEOLSpacingScope(new_poly, false);
  getCornerFillSpacingScope(new_poly);
}

void RegionQuery::addPolyScopes(DrcPoly* new_poly)
{
  addPolyMinScopes(new_poly);
  addPolyMaxScopes(new_poly);
}

void RegionQuery::addPolyList(std::vector<DrcPoly*>& new_poly_list)
{
  for (auto& new_poly : new_poly_list) {
    // 初始化多边形边
    addPolyEdge(new_poly);
    // 在Net中添加相应的poly
    addPolyToRegionQuery(new_poly);
    // 给多边形构造出scope
    addPolyScopes(new_poly);
  }
}

void RegionQuery::addPoly(DrcPoly* new_poly)
{
  // 初始化多边形边
  addPolyEdge(new_poly);
  // 在RegionQuery中添加相应的poly
  addPolyToRegionQuery(new_poly);
  // 给多边形构造出scope
  addPolyScopes(new_poly);
}
}  // namespace idrc
