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
#include "RoutingWidthCheck.h"

namespace idrc {

bool RoutingWidthCheck::check(DrcNet* target_net)
{
  checkRoutingWidth(target_net);
  return _check_result;
}

bool RoutingWidthCheck::check(DrcRect* target_rect)
{
  checkRoutingWidth(target_rect);
  return _check_result;
}

/**
 * @brief 对目标线网进行最小宽度违规检查
 *
 * @param target_net
 */
void RoutingWidthCheck::checkRoutingWidth(DrcNet* target_net)
{
  //清除上条net检查过程中的相关记录
  _checked_rect_list.clear();
  //对线网中的通孔和线段矩形进行检查
  for (auto& [layerId, routing_rect_list] : target_net->get_layer_to_routing_rects_map()) {
    for (auto target_rect : routing_rect_list) {
      //检查矩形是否存在最小宽度违规
      checkRoutingWidth(target_rect);
      //对已经检查过的矩形进行记录，避免区域搜索的过程中重复检查，产生重复报告
      _checked_rect_list.insert(target_rect);
    }
  }
}

/**
 * @brief  将目标检查矩形作为搜索区域矩形，目的是获得与目标检查矩形相交的其它矩形
 * 因为检查最小宽度包括检查两个矩形块相交部分矩形内对角线的长度值是否满足最小宽度值
 *
 * @param target_rect 目标检查矩形
 * @return RTreeBox 目标检查矩形对应的搜索区域
 */
RTreeBox RoutingWidthCheck::getQueryBox(DrcRect* target_rect)
{
  int lb_x = target_rect->get_left();
  int lb_y = target_rect->get_bottom();
  int rt_x = target_rect->get_right();
  int rt_y = target_rect->get_top();
  return RTreeBox(RTreePoint(lb_x, lb_y), RTreePoint(rt_x, rt_y));
}

/**
 * @brief 一些跳过检查的情况，这些情况下不检查最小宽度值
 *
 * @param target_rect 目标矩形
 * @param result_rect 搜索结果矩形
 * @return true 跳过检查
 * @return false 需要检查
 */
bool RoutingWidthCheck::skipCheck(DrcRect* target_rect, DrcRect* result_rect)
{
  //搜索结果矩形就是目标检查矩形本身
  if (target_rect == result_rect) {
    return true;
  }
  //与目标检查矩形相交的搜素结果矩形它们分属不同的net，属于短路的情况，间距检查模块已经检查了短路
  if (target_rect->get_net_id() != result_rect->get_net_id()) {
    // it is short violation check in spacing check
    return true;
  }
  //搜索结果矩形已经检查过，不再检查，避免出现重复的违规报告
  if (_checked_rect_list.find(result_rect) != _checked_rect_list.end()) {
    return true;
  }
  return false;
}

/**
 * @brief 检查目标矩形本身的宽度是否满足最小宽度规则
 *
 * @param layerId 金属层Id
 * @param target_rect 目标检查矩形
 * @param require_width 要求宽度
 */
void RoutingWidthCheck::checkTargetRectWidth(int layerId, DrcRect* target_rect, int require_width)
{
  if (target_rect->getWidth() < require_width) {
    _region_query->addViolation(ViolationType::kRoutingWidth);
    add_spot(layerId, target_rect, ViolationType::kRoutingWidth);
  }
}

/**
 * @brief
 * 检查两个矩形的相交矩形部分的对角线边长时，如果对角线长度不满足要求值，还要看相交的矩形部分是否被两个矩形之外的第三个矩形所覆盖，如果是则不算违规
 *
 * @param target_rect 目标检查矩形
 * @param result_rect 搜索结果矩形
 * @param overlap_rect target_rect与result_rect的相交部分矩形
 * @param query_result 以目标检查矩形为搜索区域的搜索结果
 * @return true 相交的矩形部分被两个矩形之外的第三个矩形所覆盖
 * @return false 相交的矩形部分不被两个矩形之外的第三个矩形所覆盖
 */
bool RoutingWidthCheck::isOverlapBoxCoveredByExistedRect(DrcRect* target_rect, DrcRect* result_rect, const DrcRectangle<int>& overlap_rect,
                                                         std::vector<std::pair<RTreeBox, DrcRect*>>& query_result)
{
  for (auto& rect_pair : query_result) {
    DrcRect* query_rect = rect_pair.second;
    if (query_rect == target_rect || query_rect == result_rect) {
      continue;
    }
    if (_checked_rect_list.find(query_rect) != _checked_rect_list.end()) {
      continue;
    }
    if (DRCUtil::isContainedBy(overlap_rect, query_rect)) {
      return true;
    }
  }
  return false;
}

/**
 * @brief 遍历所有与目标矩形相交的矩形，查看它们与目标矩形构成的相交矩形部分是否存在最小宽度违规
 *
 * @param layerId 金属层Id
 * @param target_rect 目标检查矩形
 * @param require_width 最小宽度要求
 * @param query_result 与目标矩形作为搜索区域获得所有与目标矩形相交的矩形
 */
void RoutingWidthCheck::checkDiagonalLengthOfOverlapRect(int layerId, DrcRect* target_rect, int require_width,
                                                         std::vector<std::pair<RTreeBox, DrcRect*>>& query_result)
{
  for (auto& rect_pair : query_result) {
    DrcRect* result_rect = rect_pair.second;
    if (skipCheck(target_rect, result_rect)) {
      continue;
    }
    //获得目标检查矩形与周边矩形相交的部分
    DrcRectangle<int> overlap_rect = DRCUtil::getSpanRectBetweenTwoRects(target_rect, result_rect);
    //获得相交矩形部分的对角线长度
    double diag_length = DRCUtil::getRectDiagonalLength(overlap_rect);
    double requireWidth = static_cast<double>(require_width);
    if (diag_length < requireWidth) {
      //检查两个矩形的相交矩形部分的对角线边长时，如果对角线长度不满足要求值，还要看相交的矩形部分是否被两个矩形之外的第三个矩形所覆盖，如果是则不算违规
      if (!isOverlapBoxCoveredByExistedRect(target_rect, result_rect, overlap_rect, query_result)) {
        if (_interact_with_op) {
          _check_result = false;
        } else {
          _region_query->addViolation(ViolationType::kRoutingWidth);
          add_spot(layerId, target_rect, result_rect);
        }
      }
    }
  }
}
/**
 * @brief 如果矩形本身的宽度值不满足最小宽度要求的情况，则将矩形本身作为违规矩形记录
 *
 * @param layerId 金属层Id
 * @param rect 违规矩形
 * @param type 违规类型
 */
void RoutingWidthCheck::add_spot(int layerId, DrcRect* rect, ViolationType type)
{
  DrcSpot spot;
  spot.set_violation_type(ViolationType::kRoutingWidth);
  spot.add_spot_rect(rect);
  _routing_layer_to_spots_map[layerId].emplace_back(spot);
}

/**
 * @brief 如果两个矩形的相交矩形部分不满足最小宽度，则将两矩形的相交部分作为违规矩形进行存储
 *
 * @param layerId 金属层Id
 * @param target_rect 目标检查矩形
 * @param result_rect 搜索结果矩形
 */
void RoutingWidthCheck::add_spot(int layerId, DrcRect* target_rect, DrcRect* result_rect)
{
  DrcSpot spot;
  spot.set_violation_type(ViolationType::kRoutingWidth);

  //获得两矩形的相交矩形部分
  DrcRectangle<int> violation_box = DRCUtil::getSpanRectBetweenTwoRects(target_rect, result_rect);
  DrcRect* drc_rect = new DrcRect();
  drc_rect->set_owner_type(RectOwnerType::kSpotMark);
  drc_rect->set_layer_id(layerId);
  drc_rect->set_rectangle(violation_box);
  spot.add_spot_rect(drc_rect);

  // spot.add_spot_rect(target_rect);
  // spot.add_spot_rect(result_rect);
  _routing_layer_to_spots_map[layerId].emplace_back(spot);
}

/**
 * @brief 检查目标矩形是否存在最小宽度违规，主要分为以下两种情况
 * 情况1：矩形本身存在最小宽度违规
 * 情况2：目标矩形与其它矩形相交部分矩形的对角线长度值不符合最小宽度要求
 *
 * @param target_rect
 */
void RoutingWidthCheck::checkRoutingWidth(DrcRect* target_rect)
{
  int routingLayerId = target_rect->get_layer_id();
  /////////////////////////////////////////////////////////////////
  /**检查矩形本身是否满足最小宽度要求**/
  int require_width = _tech->getRoutingMinWidth(routingLayerId);
  checkTargetRectWidth(routingLayerId, target_rect, require_width);
  ////////////////////////////////////////////////////////////////
  /**检查目标矩形与其它矩形相交部分矩形的对角线长度值不符合最小宽度要求**/
  //为获得与目标矩形相交的其它矩形，将目标矩形作为搜索区域进行区域搜索
  RTreeBox query_box = getQueryBox(target_rect);
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  _region_query->queryInRoutingLayer(routingLayerId, query_box, query_result);
  //遍历所有与目标矩形相交的矩形，查看它们与目标矩形构成的相交矩形部分是否存在最小宽度违规
  checkDiagonalLengthOfOverlapRect(routingLayerId, target_rect, require_width, query_result);
}

/**
 * @brief 初始化最小宽度检查模块
 *
 * @param config
 * @param tech
 */
void RoutingWidthCheck::init(DrcConfig* config, Tech* tech, RegionQuery* region_query)
{
  _config = config;
  _tech = tech;
  _region_query = region_query;
}

void RoutingWidthCheck::reset()
{
  _checked_rect_list.clear();

  for (auto& [LayerId, spot_list] : _routing_layer_to_spots_map) {
    for (auto& spot : spot_list) {
      spot.clearSpotRects();
    }
  }
  _routing_layer_to_spots_map.clear();
}

int RoutingWidthCheck::get_width_violation_num()
{
  int count = 0;
  for (auto& [layerId, spot_list] : _routing_layer_to_spots_map) {
    count += spot_list.size();
  }
  return count;
}
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////interact with iRT
// iRT目前关注间距违规和短路违规，宽度违规并没有用
void RoutingWidthCheck::checkRoutingWidth(const LayerNameToRTreeMap& layer_to_rects_tree_map)
{
  for (auto& [layerName, rtree] : layer_to_rects_tree_map) {
    int layerId = _tech->getLayerIdByLayerName(layerName);
    for (auto it = rtree.begin(); it != rtree.end(); ++it) {
      DrcRect* target_rect = it->second;
      checkRoutingWidth(layerId, target_rect, rtree);
      _checked_rect_list.insert(target_rect);
    }
  }
}

void RoutingWidthCheck::checkRoutingWidth(int layerId, DrcRect* target_rect, const RectRTree& rtree)
{
  /////////////////////////////////////////////////////////////////
  int require_width = _tech->getRoutingMinWidth(layerId);
  checkTargetRectWidth(layerId, target_rect, require_width);
  ////////////////////////////////////////////////////////////////
  RTreeBox query_box = getQueryBox(target_rect);
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  rtree.query(bgi::intersects(query_box), std::back_inserter(query_result));
  checkDiagonalLengthOfOverlapRect(layerId, target_rect, require_width, query_result);
}

}  // namespace idrc