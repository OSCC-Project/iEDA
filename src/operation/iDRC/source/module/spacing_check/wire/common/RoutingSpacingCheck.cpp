#include "RoutingSpacingCheck.h"

#include <fstream>

#include "DRCUtil.h"
// #define TEST 0

namespace idrc {
/**
 * @brief 获取短路违规的数目
 *
 * @return 返回短路违规的数目
 */
int RoutingSpacingCheck::get_short_violation_num()
{
  int short_violation_num = 0;
  for (auto& [layerId, short_spot_list] : _routing_layer_to_short_spots_list) {
    short_violation_num += static_cast<int>(short_spot_list.size());
  }
  return short_violation_num;
}

/**
 * @brief 获取间距违规的数目
 *
 * @return 返回间距违规的数目
 */
int RoutingSpacingCheck::get_spacing_violation_num()
{
  int count = 0;
  for (auto& [layerId, spacing_spot_list] : _routing_layer_to_spacing_spots_list) {
    count += spacing_spot_list.size();
  }
  return count;
}

/**
 * @brief 获得对应金属层routingLayerId下，两个矩形target_rect与result_rect间要求的合法最小间距.
 * 由于每一层的间距要求随着矩形宽度增大而增大，所以两个矩形间的间距要求要根据较大宽度矩形的宽度值来确定。
 *
 * @param routingLayerId 金属层Id
 * @param target_rect 目标检测矩形
 * @param result_rect 区域搜索的一个结果矩形
 * @return 两个矩形要求的最小间距
 */
int RoutingSpacingCheck::getRequireSpacing(int routingLayerId, DrcRect* target_rect, DrcRect* result_rect)
{
  // #if TEST
  //   return 250;
  // #endif
  //分别获取两个需要检查间距的矩形的宽度
  int target_rect_width = target_rect->getWidth();
  int result_rect_width = result_rect->getWidth();
  //获取两个矩形宽度值中较大的那个
  int width = std::max(target_rect_width, result_rect_width);
  //   The spacingtable rule is mutually exclusive with the spacing rule.
  // Check whether the layer has spacingtable rule.
  // If true, use spacingtable rule
  // If false, use the spacing rule
  auto spacing_table = _tech->get_drc_routing_layer_list()[routingLayerId]->get_spacing_table();
  if (spacing_table->is_parallel()) {
    int prl_run_length = getPRLRunLength(target_rect, result_rect);
    return spacing_table->get_parallel_spacing(width, prl_run_length);
  } else {
    //根据较大的宽度值从Tech中获取要求的间距值
    return _tech->getRoutingSpacing(routingLayerId, width);
  }
}

int RoutingSpacingCheck::getPRLRunLength(DrcRect* target_rect, DrcRect* result_rect)
{
  if (!isParallelOverlap(target_rect, result_rect) || DRCUtil::intersection(target_rect, result_rect)) {
    return 0;
  } else {
    if (((target_rect->get_bottom() > result_rect->get_top())) || ((target_rect->get_top() < result_rect->get_bottom()))) {
      return std::min(target_rect->get_right(), result_rect->get_right()) - std::max(target_rect->get_left(), result_rect->get_left());
    }
    if ((!(target_rect->get_left() > result_rect->get_right())) || (!(target_rect->get_right() < result_rect->get_left()))) {
      return std::min(target_rect->get_top(), result_rect->get_top()) - std::max(target_rect->get_bottom(), result_rect->get_bottom());
    }
  }
  return 0;
}

/**
 * @brief 对检查的目标矩形进行扩张，获得可能存在潜在违规矩形的搜索区域
 *
 * @param target_rect 目标检查矩形
 * @param spacing 间距要求也即目标检查矩形的扩张尺寸
 * @return RTreeBox 返回一个目标检查矩形扩张后得到的搜索区域
 */
RTreeBox RoutingSpacingCheck::getSpacingQueryBox(DrcRect* target_rect, int spacing)
{
  int lb_x = target_rect->get_left() - spacing;
  int lb_y = target_rect->get_bottom() - spacing;
  int rt_x = target_rect->get_right() + spacing;
  int rt_y = target_rect->get_top() + spacing;
  return RTreeBox(RTreePoint(lb_x, lb_y), RTreePoint(rt_x, rt_y));
}

/**
 * @brief 获得某一金属层搜索区域内所有的矩形
 *
 * @param routingLayerId 金属层Id
 * @param query_box 搜索区域
 * @return std::vector<std::pair<RTreeBox, DrcRect*>> 区域搜索的所有结果矩形，因为R树的存储方式所以返回的是一个矩形pair的Vector。
 * vector中的矩形pair是由表示同一矩形的两个不同数据结构组成的。
 */
std::vector<std::pair<RTreeBox, DrcRect*>> RoutingSpacingCheck::getQueryResult(int routingLayerId, RTreeBox& query_box)
{
  //调用R树的接口进行区域索引
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  _region_query->queryInRoutingLayer(routingLayerId, query_box, query_result);
  return query_result;
}

/**
 * @brief 判断是否需要跳过一些目标检测矩形和搜索结果矩形的DRC检测
 *
 * @param target_rect 目标检测矩形
 * @param result_rect 搜索结果矩形
 * @return true 跳过DRC检查
 * @return false 需要进行DRC检查
 */
bool RoutingSpacingCheck::skipCheck(DrcRect* target_rect, DrcRect* result_rect)
{
  // #if TEST
  //   return (target_rect == result_rect) || (target_rect->get_layer_id() != 1) || isChecked(result_rect) ||
  //   isSameNetRectConnect(target_rect, result_rect);
  // #endif
  //需要跳过检查的情况包括：(1)区域搜索的结果矩形就是目标检查矩形本身，(2)被检查的两个矩形都是fix的，(3)搜索结果矩形被检测过，(4)同一net的矩形两个相交矩形也跳过
  return (target_rect == result_rect) || (target_rect->is_fixed() && result_rect->is_fixed()) || isChecked(result_rect)
         || isSameNetRectConnect(target_rect, result_rect);
}

/**
 * @brief 判断target_rect与result_rect是否是两个属于同一net的矩形相交的情况
 *
 * @param target_rect 目标检查矩形
 * @param result_rect 搜索结果矩形
 * @return true 两个矩形属于同一net且相交的情况
 * @return false 两个矩形不属于同一net的且相交的情况
 */
bool RoutingSpacingCheck::isSameNetRectConnect(DrcRect* target_rect, DrcRect* result_rect)
{
  return DRCUtil::intersection(target_rect, result_rect, true) && (target_rect->get_net_id() == result_rect->get_net_id());
}

/**
 * @brief 判断区域搜索获得的搜索结果矩形result_rect是否已经被检查过
 *
 * @param result_rect 区域搜索的结果矩形
 * @return true result_rect已经被检查过
 * @return false result_rect还未被检查过
 */
bool RoutingSpacingCheck::isChecked(DrcRect* result_rect)
{
  return (_checked_rect_list.find(result_rect) != _checked_rect_list.end());
}

/**
 * @brief 判断搜索结果矩形与扩张得到的搜索区域是否仅仅是边沿相交
 * 如果是与通过扩张要求间距得到的搜索区域矩形边沿相交，那么搜索结果矩形是满足间距要求的，可跳过检查
 * _______
 *|       |________
 *| query | result |
 *| rect  | rect   |
 *|_______|________|
 * @param query_rect 搜索区域矩形
 * @param result_rect 区域搜索结果矩形
 * @return true 搜索结果矩形与搜索区域仅仅是边沿相交
 * @return false 并不是边沿相交
 */
bool RoutingSpacingCheck::intersectionExceptJustEdgeTouch(DrcRect* query_rect, DrcRect* result_rect)
{
  return DRCUtil::intersection(query_rect, result_rect, false);
}

/**
 * @brief ；两矩形是否平行交叠
 * 水平方向平行交叠         竖直方向平行交叠      不存在水平或竖直的平行交叠
 *  __________            ___                   ___
 * |__________|          |   |                 |   |
 *                       |   |  ___            |___|
 *        __________     |___| |   |                   ___
 *       |__________|          |___|                  |___|
 * @param target_rect 目标检测矩形
 * @param result_rect 搜索结果矩形
 * @return true 两矩形平行交叠
 * @return false 两矩形不存在平行交叠
 */
bool RoutingSpacingCheck::isParallelOverlap(DrcRect* target_rect, DrcRect* result_rect)
{
  return DRCUtil::isParallelOverlap(target_rect, result_rect);
}

/**
 * @brief check whether target rectangle has spacing violation with query result rectangle in corner direction
 * 检查不存在平行交叠的目标矩形与搜索结果矩形间是否存在“角间距”违规
 *
 * @param target_rect 目标矩形
 * @param result_rect 搜索结果矩形
 * @param require_spacing 最小间距要求
 * @return true there is a spacing violation between target rectangle and query result rectangle in corner direction
 * true：存在“角间距”违规
 * @return false there is no spacing violation between target rectangle and query result rectangle in corner direction
 * flase：不存在“角间距”违规
 */
bool RoutingSpacingCheck::checkCornerSpacingViolation(DrcRect* target_rect, DrcRect* result_rect, int require_spacing)
{
  int distanceX = std::min(std::abs(target_rect->get_left() - result_rect->get_right()),
                           std::abs(target_rect->get_right() - result_rect->get_left()));
  int distanceY = std::min(std::abs(target_rect->get_bottom() - result_rect->get_top()),
                           std::abs(target_rect->get_top() - result_rect->get_bottom()));
  return require_spacing * require_spacing > distanceX * distanceX + distanceY * distanceY;
}

/**
 * @brief 两个矩形间的跨度矩形是否被其它第三个矩形索贯穿
 * 竖直平行交叠下span_box   水平平行交叠下的span_box
 *             ___         ____________
 *  __________|   |       |__1_________|
 * |   |span  |   |           |span box|
 * | 1 |box   | 2 |           |________|____
 * |   |______|___|           |__2__________|
 * |___|
 * @param span_box 跨度矩形(矩形1和2的跨度矩形如上图所示)
 * @param isHorizontalParallelOverlap 是否是水平平行交叠下跨度矩形
 * @param query_result 搜索结果矩形
 * @return true 两个矩形间的
 * @return false
 */
bool RoutingSpacingCheck::checkSpanBoxCoveredByExistedRect(const RTreeBox& span_box, bool isHorizontalParallelOverlap,
                                                           std::vector<std::pair<RTreeBox, DrcRect*>>& query_result)
{
  for (auto& rect_pair : query_result) {
    // TEST
    // RTreeBox result_box = rect_pair.first;
    DrcRect* result_rect = rect_pair.second;

    // if (bg::covered_by(span_box, result_box)) {
    //   return true;
    // }

    if (DRCUtil::isBoxPenetratedByRect(span_box, result_rect, isHorizontalParallelOverlap)) {
      return true;
    }
  }
  return false;
}
//没用
bool RoutingSpacingCheck::checkSpanBoxCornorIntersectedByExitedRect(DrcRect* target_rect, DrcRect* result_rect,
                                                                    std::vector<std::pair<RTreeBox, DrcRect*>>& query_result)
{
  // std::pair<DrcCoordinate<int>, DrcCoordinate<int>> cornor_pair = DRCUtil::getSpanCornerPairBetweenTwoRects(target_rect, result_rect);
  DrcRectangle<int> spanRect = DRCUtil::getSpanRectBetweenTwoRects(target_rect, result_rect);
  for (auto& rect_pair : query_result) {
    DrcRect* query_rect = rect_pair.second;
    if (result_rect == query_rect || target_rect == query_rect) {
      continue;
    }

    if (DRCUtil::intersection(spanRect, query_rect, false)) {
      return true;
    }
    // if (DRCUtil::isRectContainCoordinate(query_rect, cornor_pair.first) || DRCUtil::isRectContainCoordinate(query_rect,
    // cornor_pair.second)) {
    //   return true;
    // }
  }
  return false;
}
/**
 * @brief check whether target rectangle has spacing violation with query result rectangle in xy direction
 * 检查两个存在平行交叠的矩形在xy（水平或竖直）方向上的距离是否满足要求
 *
 * @param target_rect 目标检查矩形
 * @param result_rect 搜索结果矩形
 * @param require_spacing 要求间距
 * @param query_result 区域搜索得到的所有矩形结果
 * @return true there is a spacing violation between target rectangle and query result rectangle in xy direction
 * true：目标矩形target_rect与搜索结果矩形result_rect存在水平或竖直方向间距违规
 * @return false there is no spacing violation between target rectangle and query result rectangle in xy direction
 * false：不存在间距违规
 */
bool RoutingSpacingCheck::checkXYSpacingViolation(DrcRect* target_rect, DrcRect* result_rect, int require_spacing,
                                                  std::vector<std::pair<RTreeBox, DrcRect*>>& query_result)
{
  RTreeBox span_box = DRCUtil::getSpanBoxBetweenTwoRects(target_rect, result_rect);
  bool isHorizontalParallelOverlap = false;
  int spacing = -1;
  int lb_x = span_box.min_corner().get<0>();
  int lb_y = span_box.min_corner().get<1>();
  int rt_x = span_box.max_corner().get<0>();
  int rt_y = span_box.max_corner().get<1>();
  if (DRCUtil::isHorizontalParallelOverlap(target_rect, result_rect)) {
    isHorizontalParallelOverlap = true;
    spacing = std::abs(rt_y - lb_y);
  } else {
    spacing = std::abs(rt_x - lb_x);
  }

  if (spacing < require_spacing) {
    //如果两矩形间的间距不满足间距要求且还要判断两矩形的跨度矩形是否被第三个个矩形所贯穿，如果不是则存在违规，
    return !checkSpanBoxCoveredByExistedRect(span_box, isHorizontalParallelOverlap, query_result);
  }
  return false;
}

/**
 * @brief check whether target rectangle has spacing violation with query result rectangles
 * 检查目标矩形与搜索结果矩形之间是否存在间距违规
 *
 * @param target_rect 目标矩形
 * @param result_rect 搜索结果矩形
 * @param query_result 区域搜索的所有矩形结果
 * @return true there is a spacing violation between target rectangle and query result rectangles
 * true：存在违规
 * @return false there is no short violation between target rectangle and query result rectangles
 * false：不存在违规
 */
bool RoutingSpacingCheck::checkSpacingViolation(int routingLayerId, DrcRect* target_rect, DrcRect* result_rect,
                                                std::vector<std::pair<RTreeBox, DrcRect*>>& query_result)
{
  // int max_require_spacing = _tech->getRoutingMaxRequireSpacing(routingLayerId);
  // DrcRect query_rect = DRCUtil::enlargeRect(target_rect, max_require_spacing);
  // The rectangle that just intersects the edge of the search box is not violation
  //如果搜索结果矩形与膨胀后获得的搜索区域矩形仅仅是边沿相交则不存在违规，排除这些搜索结果矩形的检查
  // if (intersectionExceptJustEdgeTouch(&query_rect, result_rect)) {
  int require_spacing = getRequireSpacing(routingLayerId, target_rect, result_rect);
  // RTreeBox span_box = DRCUtil::getSpanBoxBetweenTwoRects(target_rect, result_rect);
  // std::vector<std::pair<RTreeBox, DrcRect*>> span_box_query_result;
  // _region_query->queryInRoutingLayer(routingLayerId, span_box, span_box_query_result);
  // if (!span_box_query_result.empty()) {
  //   return false;
  // }
  if (!isParallelOverlap(target_rect, result_rect)) {
    // case no Parallel Overlap between two rect ,need check corner spacing
    // if corner spacing is not meet require_spacing,it is a violation
    //如果两个矩形不存在平行交叠则检查角间距
    return checkCornerSpacingViolation(target_rect, result_rect, require_spacing);
  } else {
    // There is  Parallel Overlap between two rect
    // need check span box is covered by exited rect
    //存在平行交叠检查X或Y方向上的间距
    return checkXYSpacingViolation(target_rect, result_rect, require_spacing, query_result);
  }
  // }
  return false;
}

/**
 * @brief check whether target rectangle has short violation with query result rectangle
 * 检查两个矩形target_rect与result_rect是否存在短路违规
 * @param target_rect 目标矩形
 * @param result_rect 区域搜索结果矩形
 * @return true there is a short violation between target rectangle and query result rectangle
 * true：存在短路违规
 * @return false there is no short violation between target rectangle and query result rectangle
 * false：不存在短路违规
 */
bool RoutingSpacingCheck::checkShort(DrcRect* target_rect, DrcRect* result_rect)
{
  if ((DRCUtil::intersection(target_rect, result_rect, true)) && (target_rect->get_net_id() != result_rect->get_net_id())) {
    // static int count = 0;
    // count++;
    // if (count > 100000) {
    //   exit(0);
    // }
    // std::ofstream record_file_stream = std::ofstream("/home/zhangmz/Download/i-eda-4/irefactor/build/drc.log", std::ios_base::app);

    // record_file_stream << "count:: " << count << std::endl;
    // record_file_stream << "target::  "
    //                    << "(" << target_rect->get_left() << "," << target_rect->get_bottom() << ")"
    //                    << "(" << target_rect->get_right() << "," << target_rect->get_top() << std::endl;
    // record_file_stream << "layer_id:  " << target_rect->get_layer_id() << std::endl;
    // record_file_stream << "net_name:  " << target_rect->net_name << std::endl;
    // record_file_stream << "net_id:  " << target_rect->get_net_id() << std::endl;
    // record_file_stream << "result::  "
    //                    << "(" << result_rect->get_left() << "," << result_rect->get_bottom() << ")"
    //                    << "(" << result_rect->get_right() << "," << result_rect->get_top() << std::endl;
    // record_file_stream << "layer_id:  " << result_rect->get_layer_id() << std::endl;
    // record_file_stream << "net_name:  " << result_rect->net_name << std::endl;
    // record_file_stream << "net_id:  " << result_rect->get_net_id() << std::endl;
    return true;
  }
  return false;
}
/**
 * @brief check whether target rectangle has spacing violation with query result rectangle
 * 检查区域搜索获得的所有搜索结果矩形中是否存在矩形与目标矩形间存在间距违规，并存储违规结果
 *
 * @param target_rect 目标检查矩形
 * @param query_result 区域搜索结果
 */
void RoutingSpacingCheck::checkSpacingFromQueryResult(int routingLayerId, DrcRect* target_rect,
                                                      std::vector<std::pair<RTreeBox, DrcRect*>>& query_result)
{
  for (auto rect_pair : query_result) {
    DrcRect* result_rect = rect_pair.second;
    //跳过一些不需要检查的情况

    if (skipCheck(target_rect, result_rect)) {
      continue;
    }
    //检查是否短路
    RTreeBox span_box = DRCUtil::getSpanBoxBetweenTwoRects(target_rect, result_rect);

    if (checkShort(target_rect, result_rect)) {
      if (_region_query->addShortViolation(target_rect, result_rect)) {
        // addShortSpot(target_rect, result_rect);
        _region_query->addShortViolation(routingLayerId, span_box);
      }
      // _region_query->addViolation(ViolationType::kShort);
      // storeViolationResult(routingLayerId, target_rect, result_rect, ViolationType::kShort);
      continue;
    }
    std::vector<std::pair<RTreeBox, DrcRect*>> span_box_query_result;
    // std::vector<std::pair<RTreeSegment, DrcEdge*>> span_box_query_result;

    _region_query->queryIntersectsInRoutingLayer(routingLayerId, span_box, span_box_query_result);
    // _region_query->queryEdgeInRoutingLayer(routingLayerId, span_box, span_box_query_result);
    if (!span_box_query_result.empty()) {
      continue;
    }
    //检查是否存在间距违规
    if (checkSpacingViolation(routingLayerId, target_rect, result_rect, query_result)) {
      // _region_query->addViolation(ViolationType::kRoutingSpacing);
      _region_query->addPRLRunLengthSpacingViolation(routingLayerId, span_box);
      // if (_region_query->addPRLRunLengthSpacingViolation(routingLayerId, span_box)) {
      //   // addSpacingSpot(target_rect, result_rect);
      // }
      // storeViolationResult(routingLayerId, target_rect, result_rect, ViolationType::kRoutingSpacing);
      /// build conflict graph
      // by rect
      // _conflict_graph->addEdge(target_rect, result_rect);
      // by polygon
      // DrcPolygon* polygon1 = target_rect->get_owner_polygon();
      // DrcPolygon* polygon2 = result_rect->get_owner_polygon();
      // if (polygon1 != polygon2) {
      //   if (polygon1 != nullptr && polygon2 != nullptr) {
      //     _conflict_polygon_map[polygon1].insert(polygon2);
      //     _conflict_polygon_map[polygon2].insert(polygon1);
      //   } else {
      //     std::cout << "rect has nullptr polygon !!!" << std::endl;
      //   }
      // }
    }
  }
  // _checked_rect_list.insert(target_rect);
}

/**
 * @brief store violation results
 *
 * @param target_rect：目标检测矩形
 * @param result_rect：与目标检测矩形产生违规的矩形
 * @param type：违规类型
 */
void RoutingSpacingCheck::add_spot(int routingLayerId, DrcRect* target_rect, DrcRect* result_rect, ViolationType type)
{
  DrcSpot spot;
  spot.set_violation_type(type);
  spot.add_spot_rect(target_rect);
  spot.add_spot_rect(result_rect);

  if (type == ViolationType::kRoutingSpacing) {
    _routing_layer_to_spacing_spots_list[routingLayerId].emplace_back(spot);
  } else if (type == ViolationType::kShort) {
    // DrcRect* drc_rect = new DrcRect();
    // drc_rect->set_layer_id(routingLayerId);
    // drc_rect->set_owner_type(RectOwnerType::kSpotMark);
    // DrcRectangle<int> violation_box = DRCUtil::getSpanRectBetweenTwoRects(target_rect, result_rect);
    // drc_rect->set_rectangle(violation_box);
    // spot.add_spot_rect(drc_rect);
    _routing_layer_to_short_spots_list[routingLayerId].emplace_back(spot);
  }
}

void RoutingSpacingCheck::storeViolationResult(int routingLayerId, DrcRect* target_rect, DrcRect* result_rect, ViolationType type)
{
  if (type == ViolationType::kShort) {
    if (_interact_with_op) {
      // _violation_rect_pair_list.push_back(std::make_pair(target_rect, result_rect));
      _check_result = false;
    } else {
      add_spot(routingLayerId, target_rect, result_rect, ViolationType::kShort);
    }
  } else if (type == ViolationType::kRoutingSpacing) {
    if (_interact_with_op) {
      // _violation_rect_pair_list.push_back(std::make_pair(target_rect, result_rect));
      _check_result = false;
    } else {
      addViolationBox(routingLayerId, target_rect, result_rect);
    }
  }
}

/**
 * @brief check whether the target rectangle has spacing violation with other rectangle
 * 检查目标矩形与周边其它矩形是否存在间距违规或短路违规
 *
 * @param target_rect 目标矩形
 */
void RoutingSpacingCheck::checkRoutingSpacing(DrcRect* target_rect)
{
  int routingLayerId = target_rect->get_layer_id();
  //获得当前金属层的最大金属宽度对应的要求间距
  int layer_max_require_spacing = _tech->getRoutingMaxRequireSpacing(routingLayerId, target_rect);
  //通过当前层的最大金属宽度所对应间距要求膨胀矩形获得搜索区域
  RTreeBox query_box = getSpacingQueryBox(target_rect, layer_max_require_spacing);
  //通过搜索区域获得对应金属层在区域内的所有矩形
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result = getQueryResult(routingLayerId, query_box);
  //检查当前金属层的检查目标矩形与搜索区域中的矩形是否存在间距违规
  checkSpacingFromQueryResult(routingLayerId, target_rect, query_result);
}

/**
 * @brief check spacing drc between rectangles in target net
 * 检查目标net中的矩形是否存在间距违规
 *
 * @param target_net
 */
void RoutingSpacingCheck::checkRoutingSpacing(DrcNet* target_net)
{
  for (auto& [layerId, routing_rect_list] : target_net->get_layer_to_routing_rects_map()) {
    for (auto target_rect : routing_rect_list) {
      checkRoutingSpacing(target_rect);
    }
  }

  // #if TEST
  // for (auto& [layerId, pin_rect_list] : target_net->get_layer_to_pin_rects_map()) {
  //   for (auto target_rect : pin_rect_list) {
  //     checkRoutingSpacing(target_rect);
  //   }
  // }
  // #endif
  //将检查net过程中以R树形式存储的违规标记矩形转存到map<layer,vector<spot>>中
  // initSpacingSpotListFromRtree();
  // //每条net清除一下，不然下条net会重复记录
  // _layer_to_violation_box_tree.clear();
}

/**
 * @brief 初始化RoutingSpacingCheck模块
 *
 * @param config
 * @param tech
 * @param graph
 */
void RoutingSpacingCheck::init(DrcConfig* config, Tech* tech, RegionQuery* region_query)
{
  _config = config;
  _tech = tech;
  _region_query = region_query;
}

/**
 * @brief 重置RoutingSpacingCheck模块的过程数据或存储结果
 *
 */
void RoutingSpacingCheck::reset()
{
  _checked_rect_list.clear();

  for (auto& [LayerId, short_spot_list] : _routing_layer_to_short_spots_list) {
    for (auto& short_spot : short_spot_list) {
      short_spot.clearSpotRects();
    }
  }
  _routing_layer_to_short_spots_list.clear();

  for (auto& [LayerId, spacing_spot_list] : _routing_layer_to_spacing_spots_list) {
    for (auto& spacing_spot : spacing_spot_list) {
      spacing_spot.clearSpotRects();
    }
  }
  _routing_layer_to_spacing_spots_list.clear();
  _layer_to_violation_box_tree.clear();
  _conflict_graph->clearAllNode();
  _violation_rect_pair_list.clear();
}

/**
 * @brief 获取对应金属层下搜索区域内的所有违规标记矩形
 *
 * @param routingLayerId 金属层Id
 * @param query_box 搜索区域矩形
 * @param result 区域搜索的所有违规标记矩形结果
 */
void RoutingSpacingCheck::searchIntersectedViolationBox(int routingLayerId, const RTreeBox& query_box, std::vector<RTreeBox>& result)
{
  _layer_to_violation_box_tree[routingLayerId].query(bgi::intersects(query_box), std::back_inserter(result));
}

/**
 * @brief 将两个存在的间距违规矩形的跨度矩形span_box作为违规标记矩形violation_box存入R树中
 * 在存入违规标记矩形的过程中，通过R树获取当前违规标记矩形是否与已经存入R树的的违规标记矩形可以合并成一个矩形，如果可以，则将它们合并成一个矩形再存入，并删除已经存入R树且参与合并的矩形
 *
 * @param layerId 金属层Id
 * @param target_rect 目标检测矩形
 * @param result_rect 搜索结果矩形
 */
void RoutingSpacingCheck::addViolationBox(int layerId, DrcRect* target_rect, DrcRect* result_rect)
{
  // int layerId = target_rect->get_layer_id();
  std::vector<RTreeBox> query_result;
  RTreeBox query_box = DRCUtil::getSpanBoxBetweenTwoRects(target_rect, result_rect);
  searchIntersectedViolationBox(layerId, query_box, query_result);

  for (auto& box : query_result) {
    if (DRCUtil::isPenetratedIntersected(box, query_box)) {
      int lb_x = std::min(box.min_corner().get<0>(), query_box.min_corner().get<0>());
      int lb_y = std::min(box.min_corner().get<1>(), query_box.min_corner().get<1>());
      int rt_x = std::max(box.max_corner().get<0>(), query_box.max_corner().get<0>());
      int rt_y = std::max(box.max_corner().get<1>(), query_box.max_corner().get<1>());
      query_box.min_corner().set<0>(lb_x);
      query_box.min_corner().set<1>(lb_y);
      query_box.max_corner().set<0>(rt_x);
      query_box.max_corner().set<1>(rt_y);
      _layer_to_violation_box_tree[layerId].remove(box);
    }
  }
  _layer_to_violation_box_tree[layerId].insert(query_box);
}

/**
 * @brief 在检查完一条net后，将R树中存储的违规标记矩形更新到map<layer,vector<spot>>中
 *
 */
void RoutingSpacingCheck::initSpacingSpotListFromRtree()
{
  for (auto& [layerId, rtree] : _layer_to_violation_box_tree) {
    for (auto it = rtree.begin(); it != rtree.end(); ++it) {
      RTreeBox box = *it;
      DrcRectangle<int> rect = DRCUtil::getRectangleFromRTreeBox(box);

      DrcSpot spot;
      spot.set_violation_type(ViolationType::kRoutingSpacing);

      DrcRect* drc_rect = new DrcRect();
      drc_rect->set_owner_type(RectOwnerType::kSpotMark);
      drc_rect->set_layer_id(layerId);
      drc_rect->set_rectangle(rect);
      spot.add_spot_rect(drc_rect);

      _routing_layer_to_spacing_spots_list[layerId].emplace_back(spot);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////interact with iRT
//在于iRT的交互过程中，区域搜索的结果是由iRT传入的R树提供的
// void RoutingSpacingCheck::checkRoutingSpacing(const LayerNameToRTreeMap& layer_to_rects_tree_map)
// {
//   for (auto& [layerName, rtree] : layer_to_rects_tree_map) {
//     int layerId = _tech->getLayerIdByLayerName(layerName);
//     for (auto it = rtree.begin(); it != rtree.end(); ++it) {
//       DrcRect* target_rect = it->second;
//       checkRoutingSpacing(layerId, target_rect, rtree);
//       _checked_rect_list.insert(target_rect);
//     }
//   }
// }

bool RoutingSpacingCheck::check(DrcRect* target_rect)
{
  int layer_id = target_rect->get_layer_id();
  // if(_layer_rect_rtree_map->find(layer_id)==_layer_rect_rtree_map->end()){
  //   std::cout<<"[DRC Common Spacing Check Warning]: the layer task belong to of env has nothing!"<<std::endl;
  //   return true;
  // }
  // _rtree = _layer_rect_rtree_map->find(layer_id)->second;
  int layer_max_require_spacing = _tech->getRoutingMaxRequireSpacing(layer_id, target_rect);
  RTreeBox query_box = getSpacingQueryBox(target_rect, layer_max_require_spacing);
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  _region_query->queryInRoutingLayer(layer_id, query_box, query_result);
  checkSpacingFromQueryResult(layer_id, target_rect, query_result);
  return _check_result;
}

bool RoutingSpacingCheck::check(void* target, DrcRect* check_rect)
{
  DrcRect* target_rect = static_cast<DrcRect*>(target);
  int layer_id = target_rect->get_layer_id();
  int require_spacing = getRequireSpacing(layer_id, target_rect, check_rect);
  RTreeBox query_box = getSpacingQueryBox(target_rect, require_spacing);
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  _region_query->queryInRoutingLayer(layer_id, query_box, query_result);
  if (checkSpacingViolation(layer_id, target_rect, check_rect, query_result)) {
    return false;
  }
  return true;
}

void RoutingSpacingCheck::addShortSpot(DrcRect* target_rect, DrcRect* result_rect)
{
  auto box = DRCUtil::getSpanBoxBetweenTwoRects(target_rect, result_rect);
  DrcViolationSpot* spot = new DrcViolationSpot();
  int layer_id = target_rect->get_layer_id();
  spot->set_layer_id(layer_id);
  spot->set_layer_name(_tech->getRoutingLayerNameById(layer_id));
  spot->set_net_id(target_rect->get_net_id());
  spot->set_vio_type(ViolationType::kShort);
  spot->setCoordinate(box.min_corner().x(), box.min_corner().y(), box.max_corner().x(), box.max_corner().y());
  _region_query->_short_vio_spot_list.emplace_back(spot);
}

void RoutingSpacingCheck::addSpacingSpot(DrcRect* target_rect, DrcRect* result_rect)
{
  auto box = DRCUtil::getSpanBoxBetweenTwoRects(target_rect, result_rect);
  DrcViolationSpot* spot = new DrcViolationSpot();
  int layer_id = target_rect->get_layer_id();
  spot->set_layer_id(layer_id);
  spot->set_layer_name(_tech->getRoutingLayerNameById(layer_id));
  spot->set_net_id(target_rect->get_net_id());
  spot->set_vio_type(ViolationType::kRoutingSpacing);
  spot->setCoordinate(box.min_corner().x(), box.min_corner().y(), box.max_corner().x(), box.max_corner().y());
  _region_query->_prl_run_length_spacing_spot_list.emplace_back(spot);
}

}  // namespace idrc