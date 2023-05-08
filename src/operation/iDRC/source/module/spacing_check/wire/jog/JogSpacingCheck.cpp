#include "JogSpacingCheck.hpp"

#include "DRCUtil.h"
namespace idrc {

void JogSpacingCheck::checkJogSpacing(DrcNet* target_net)
{
  for (auto& [layerId, routing_rect_list] : target_net->get_layer_to_routing_rects_map()) {
    for (auto target_rect : routing_rect_list) {
      int layer_id = target_rect->get_layer_id();
      _rule = _tech->get_drc_routing_layer_list()[layer_id]->get_lef58_jog_spacing_rule();
      if (_rule == nullptr) {
        continue;
      }

      auto width_list = _rule->get_width_list();
      _find_violation = false;
      int size = width_list.size();
      for (_width_list_index = 0; _width_list_index < size; _width_list_index++) {
        if (target_rect->getWidth() >= width_list[_width_list_index].get_width()) {
          if (target_rect->isHorizontal()) {
            _wid_rect_dir_is_horizontal = true;
          } else {
            _wid_rect_dir_is_horizontal = false;
          }
          checkJogSpacing(target_rect);
          if (_find_violation == true) {
            break;
          }
        }
      }
    }
  }
}

/**
 * @brief Search for trigger rect above the wide rect
 *
 * @param query_box
 * @param wid_rect
 */
void JogSpacingCheck::getParallelQueryBox_up(RTreeBox& query_box, DrcRect* wid_rect)
{
  int query_box_width = _rule->get_width_list()[_width_list_index].get_par_within();

  bg::set<bg::min_corner, 0>(query_box, wid_rect->get_left());
  bg::set<bg::min_corner, 1>(query_box, wid_rect->get_top());
  bg::set<bg::max_corner, 0>(query_box, wid_rect->get_right());
  bg::set<bg::max_corner, 1>(query_box, wid_rect->get_top() + query_box_width);

  if (query_box.min_corner().x() < 0) {
    bg::set<bg::min_corner, 0>(query_box, 0);
  }
  if (query_box.min_corner().y() < 0) {
    bg::set<bg::min_corner, 1>(query_box, 0);
  }
  if (query_box.max_corner().x() < 0) {
    bg::set<bg::max_corner, 0>(query_box, 0);
  }
  if (query_box.max_corner().y() < 0) {
    bg::set<bg::max_corner, 1>(query_box, 0);
  }
}

/**
 * @brief Search for trigger rect down the wide rect
 *
 * @param query_box
 * @param wid_rect
 */
void JogSpacingCheck::getParallelQueryBox_down(RTreeBox& query_box, DrcRect* wid_rect)
{
  int query_box_width = _rule->get_width_list()[_width_list_index].get_par_within();

  bg::set<bg::min_corner, 0>(query_box, wid_rect->get_left());
  bg::set<bg::min_corner, 1>(query_box, wid_rect->get_bottom() - query_box_width);
  bg::set<bg::max_corner, 0>(query_box, wid_rect->get_right());
  bg::set<bg::max_corner, 1>(query_box, wid_rect->get_bottom());

  if (query_box.min_corner().x() < 0) {
    bg::set<bg::min_corner, 0>(query_box, 0);
  }
  if (query_box.min_corner().y() < 0) {
    bg::set<bg::min_corner, 1>(query_box, 0);
  }
  if (query_box.max_corner().x() < 0) {
    bg::set<bg::max_corner, 0>(query_box, 0);
  }
  if (query_box.max_corner().y() < 0) {
    bg::set<bg::max_corner, 1>(query_box, 0);
  }
}

/**
 * @brief Search for trigger on the left side of the wide rect
 *
 * @param query_box
 * @param wid_rect
 */
void JogSpacingCheck::getParallelQueryBox_left(RTreeBox& query_box, DrcRect* wid_rect)
{
  int query_box_width = _rule->get_width_list()[_width_list_index].get_par_within();

  bg::set<bg::min_corner, 0>(query_box, wid_rect->get_left() - query_box_width);
  bg::set<bg::min_corner, 1>(query_box, wid_rect->get_bottom());
  bg::set<bg::max_corner, 0>(query_box, wid_rect->get_left());
  bg::set<bg::max_corner, 1>(query_box, wid_rect->get_top());

  if (query_box.min_corner().x() < 0) {
    bg::set<bg::min_corner, 0>(query_box, 0);
  }
  if (query_box.min_corner().y() < 0) {
    bg::set<bg::min_corner, 1>(query_box, 0);
  }
  if (query_box.max_corner().x() < 0) {
    bg::set<bg::max_corner, 0>(query_box, 0);
  }
  if (query_box.max_corner().y() < 0) {
    bg::set<bg::max_corner, 1>(query_box, 0);
  }
}

/**
 * @brief Search for trigger on the right side of the wide line
 *
 * @param query_box
 * @param wid_rect
 */
void JogSpacingCheck::getParallelQueryBox_right(RTreeBox& query_box, DrcRect* wid_rect)
{
  int query_box_width = _rule->get_width_list()[_width_list_index].get_par_within();

  bg::set<bg::min_corner, 0>(query_box, wid_rect->get_right());
  bg::set<bg::min_corner, 1>(query_box, wid_rect->get_bottom());
  bg::set<bg::max_corner, 0>(query_box, wid_rect->get_right() + query_box_width);
  bg::set<bg::max_corner, 1>(query_box, wid_rect->get_top());

  if (query_box.min_corner().x() < 0) {
    bg::set<bg::min_corner, 0>(query_box, 0);
  }
  if (query_box.min_corner().y() < 0) {
    bg::set<bg::min_corner, 1>(query_box, 0);
  }
  if (query_box.max_corner().x() < 0) {
    bg::set<bg::max_corner, 0>(query_box, 0);
  }
  if (query_box.max_corner().y() < 0) {
    bg::set<bg::max_corner, 1>(query_box, 0);
  }
}

/**
 * @brief get the lengths of the parallel run length of the two rects
 *
 * @param target_rect
 * @param result_rect
 * @return int parallel run length
 */
int JogSpacingCheck::getPRLRunLength(DrcRect* target_rect, DrcRect* result_rect)
{
  if (!DRCUtil::isParallelOverlap(target_rect, result_rect)) {
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
 * @brief Determines whether a rectangle triggers a JogSpacing check
 *        Triggered when the length of the parallel edge is greater than the specified value
 * @param wid_rect
 * @param result_rect
 * @return true triggered
 * @return false not triggered
 */
bool JogSpacingCheck::isTriggerRect(DrcRect* wid_rect, DrcRect* result_rect)
{
  int parallel_run_length = _rule->get_width_list()[_width_list_index].get_par_length();
  return getPRLRunLength(wid_rect, result_rect) > parallel_run_length;
}

/**
 * @brief Get the query_box for search Jog or neighbor rect
 *
 * @param query_box
 * @param wid_rect
 * @param trigger_rect
 */
void JogSpacingCheck::getJogQueryBox_up(RTreeBox& query_box, DrcRect* wid_rect, DrcRect* trigger_rect)
{
  bg::set<bg::min_corner, 0>(query_box, trigger_rect->get_left());
  bg::set<bg::min_corner, 1>(query_box, wid_rect->get_top());
  bg::set<bg::max_corner, 0>(query_box, wid_rect->get_right());
  bg::set<bg::max_corner, 1>(query_box, wid_rect->get_top());
}

/**
 * @brief Judging whether the rect is a jog of a wide rect by whether the two intersect
 *
 * @param result_rect
 * @param wid_rect
 * @return true
 * @return false
 */
bool JogSpacingCheck::isJogOfWidRect(DrcRect* result_rect, DrcRect* wid_rect)
{
  return DRCUtil::intersection(result_rect, wid_rect);
}

/**
 * @brief If the search result is wide rect or trigger rect itself, skip the check
 *
 * @param wid_rect
 * @param trigger_rect
 * @param result_rect
 * @return true
 * @return false
 */
bool JogSpacingCheck::skipCheck(DrcRect* wid_rect, DrcRect* trigger_rect, DrcRect* result_rect)
{
  return DRCUtil::isSameRect(result_rect, wid_rect) && DRCUtil::isSameRect(result_rect, trigger_rect);
}

/**
 * @brief Check whether the jog or neighbor rect meet jogSpacing requirements,
 *        and determine if it requires a jogtojog spacing check
 *
 * @param intercept_result_rect
 * @param check_rect
 * @param rect
 * @param jogs_need_to_check_jog2jog_spacing
 */
void JogSpacingCheck::checkSpacing_Vertical(DrcRect* intercept_result_rect, DrcRect* check_rect, DrcRect* rect,
                                            std::vector<DrcRect>& jogs_need_to_check_jog2jog_spacing)
{
  int layer_id = check_rect->get_layer_id();
  int long_jog_spacing = _rule->get_width_list()[layer_id].get_long_jog_spacing();
  int short_jog_spacing = _rule->get_short_jog_spacing();
  DrcRectangle<int> span_rect = DRCUtil::getSpanRectBetweenTwoRects(intercept_result_rect, check_rect);
  if (span_rect.get_rt_x() - span_rect.get_lb_x() > long_jog_spacing) {
    return;
  }
  // if (isSpanRectCovered(span_rect)) {
  //   return;
  // }
  int jog_width = _rule->get_jog_width();
  if (span_rect.get_rt_y() - span_rect.get_lb_y() > jog_width) {
    if (span_rect.get_rt_x() - span_rect.get_lb_x() < long_jog_spacing) {
      std::cout << "violation!" << std::endl;
      // add_spot();
    }
  } else {
    if (span_rect.get_rt_x() - span_rect.get_lb_x() < long_jog_spacing) {
      // if rect is a jog,need to check jogtojog-spacing
      if (isJog(intercept_result_rect, check_rect, rect)) {
        jogs_need_to_check_jog2jog_spacing.push_back(*intercept_result_rect);
      }
    }
    if (span_rect.get_rt_x() - span_rect.get_lb_x() < short_jog_spacing) {
      std::cout << "violation!" << std::endl;
      // add_spot();
    }
  }
}

void JogSpacingCheck::checkSpacing_Horizontal(DrcRect* intercept_result_rect, DrcRect* check_rect, DrcRect* rect,
                                              std::vector<DrcRect>& jogs_need_to_check_jog2jog_spacing)
{
  int layer_id = check_rect->get_layer_id();
  int long_jog_spacing = _rule->get_width_list()[layer_id].get_long_jog_spacing();
  int short_jog_spacing = _rule->get_short_jog_spacing();
  DrcRectangle<int> span_rect = DRCUtil::getSpanRectBetweenTwoRects(intercept_result_rect, check_rect);
  if (span_rect.get_rt_y() - span_rect.get_lb_y() > long_jog_spacing) {
    return;
  }
  // if (isSpanRectCovered(span_rect)) {
  //   return;
  // }
  int jog_width = _rule->get_jog_width();
  if (span_rect.get_rt_x() - span_rect.get_lb_x() > jog_width) {
    if (span_rect.get_rt_y() - span_rect.get_lb_y() < long_jog_spacing) {
      std::cout << "violation!" << std::endl;
      // add_spot();
    }
  } else {
    // Get all the jogs that need to be checked jog2jog spacing
    if (span_rect.get_rt_x() - span_rect.get_lb_x() < long_jog_spacing) {
      if (isJog(intercept_result_rect, check_rect, rect)) {
        jogs_need_to_check_jog2jog_spacing.push_back(*intercept_result_rect);
      }
    }
    if (span_rect.get_rt_y() - span_rect.get_lb_y() < short_jog_spacing) {
      std::cout << "violation!" << std::endl;
      // add_spot();
    }
  }
}

/**
 * @brief Use trigger rect to intercept the Result because the width
 *        of the jog is calculated from the jog after the intercept
 *
 * @param result_rect
 * @param trigger_rect
 * @param intercept_result_rect
 */
void JogSpacingCheck::interceptResultRect_Vertical(DrcRect* result_rect, DrcRect* trigger_rect, DrcRect& intercept_result_rect)
{
  intercept_result_rect = *result_rect;
  int lb_x = result_rect->get_left();
  int lb_y = std::max(result_rect->get_bottom(), trigger_rect->get_bottom());
  int rt_x = result_rect->get_right();
  int rt_y = std::min(result_rect->get_top(), trigger_rect->get_top());
  intercept_result_rect.set_coordinate(lb_x, lb_y, rt_x, rt_y);
}

void JogSpacingCheck::interceptResultRect_Horizontal(DrcRect* result_rect, DrcRect* trigger_rect, DrcRect& intercept_result_rect)
{
  intercept_result_rect = *result_rect;
  int lb_x = std::max(result_rect->get_left(), trigger_rect->get_left());
  int lb_y = result_rect->get_bottom();
  int rt_x = std::min(result_rect->get_right(), trigger_rect->get_right());
  int rt_y = result_rect->get_top();
  intercept_result_rect.set_coordinate(lb_x, lb_y, rt_x, rt_y);
}

/**
 * @brief Check the jogs between trigger rect and wide rect
 *         for jogSpacing and for jogtojog-Spacing violations
 *
 * @param wid_rect
 * @param trigger_rect
 */
void JogSpacingCheck::checkTriggerRect_Horizontal(DrcRect* wid_rect, DrcRect* trigger_rect)
{
  RTreeBox query_box = DRCUtil::getSpanBoxBetweenTwoRects(wid_rect, trigger_rect);
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  _region_query->queryInRoutingLayer(wid_rect->get_layer_id(), query_box, query_result);
  std::vector<DrcRect> jogs_need_to_check_jog2jog_spacing;
  for (auto& [rtree_box, result_rect] : query_result) {
    if (skipCheck(wid_rect, trigger_rect, result_rect)) {
      continue;
    }
    // Use trigger rect to intercept the result rect
    DrcRect intercept_result_rect;
    interceptResultRect_Horizontal(result_rect, trigger_rect, intercept_result_rect);
    if (isJogOfWidRect(result_rect, wid_rect)) {
      checkSpacing_Horizontal(&intercept_result_rect, trigger_rect, wid_rect, jogs_need_to_check_jog2jog_spacing);
    } else {
      checkSpacing_Horizontal(&intercept_result_rect, wid_rect, trigger_rect, jogs_need_to_check_jog2jog_spacing);
    }
  }
  checkJog2JogSpacing(jogs_need_to_check_jog2jog_spacing);
}

void JogSpacingCheck::checkTriggerRect_Vertical(DrcRect* wid_rect, DrcRect* trigger_rect)
{
  RTreeBox query_box = DRCUtil::getSpanBoxBetweenTwoRects(wid_rect, trigger_rect);
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  _region_query->queryInRoutingLayer(wid_rect->get_layer_id(), query_box, query_result);
  std::vector<DrcRect> jogs_need_to_check_jog2jog_spacing;
  for (auto& [rtree_box, result_rect] : query_result) {
    if (skipCheck(wid_rect, trigger_rect, result_rect)) {
      continue;
    }
    // Use trigger rect to intercept the result rect
    DrcRect intercept_result_rect;
    interceptResultRect_Vertical(result_rect, trigger_rect, intercept_result_rect);
    if (isJogOfWidRect(result_rect, wid_rect)) {
      checkSpacing_Vertical(&intercept_result_rect, trigger_rect, wid_rect, jogs_need_to_check_jog2jog_spacing);
    } else {
      checkSpacing_Vertical(&intercept_result_rect, wid_rect, trigger_rect, jogs_need_to_check_jog2jog_spacing);
    }
  }
  checkJog2JogSpacing(jogs_need_to_check_jog2jog_spacing);
}

/**
 * @brief Judging whether the rect is a jog by whether it intersect with trigger rect or wide rect
 *
 * @param result_rect
 * @param wid_rect
 * @param trigger_rect
 * @return true
 * @return false
 */
bool JogSpacingCheck::isJog(DrcRect* result_rect, DrcRect* wid_rect, DrcRect* trigger_rect)
{
  return DRCUtil::intersection(result_rect, wid_rect) || DRCUtil::intersection(result_rect, trigger_rect);
}

/**
 * @brief Check the wide-wire direction spacing between two jogs
 *
 * @param rect1
 * @param rect2
 */
void JogSpacingCheck::checkJog2JogSpacing(DrcRect& rect1, DrcRect& rect2)
{
  int jog2jog_spacing = _rule->get_jog_to_jog_spacing();
  if (_wid_rect_dir_is_horizontal) {
    int wire_dir_spacing = std::max(rect1.get_left() - rect2.get_right(), rect2.get_left() - rect1.get_right());
    if (wire_dir_spacing < jog2jog_spacing) {
      std::cout << "jog2jog vio" << std::endl;
    }
  } else {
    int wire_dir_spacing = std::max(rect1.get_bottom() - rect2.get_top(), rect2.get_bottom() - rect1.get_top());
    if (wire_dir_spacing < jog2jog_spacing) {
      std::cout << "jog2jog vio" << std::endl;
    }
  }
}

/**
 * @brief Check the wide-wire direction spacing between any two jogs in list of jogs need to check jogtojog-spacing
 *
 * @param jogs_need_to_check_jog2jog_spacing
 */
void JogSpacingCheck::checkJog2JogSpacing(std::vector<DrcRect>& jogs_need_to_check_jog2jog_spacing)
{
  int size = jogs_need_to_check_jog2jog_spacing.size();

  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      checkJog2JogSpacing(jogs_need_to_check_jog2jog_spacing[i], jogs_need_to_check_jog2jog_spacing[j]);
    }
  }
}

void JogSpacingCheck::checkJogSpacing_up(DrcRect* wid_rect)
{
  RTreeBox query_box;
  getParallelQueryBox_up(query_box, wid_rect);
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  _region_query->queryInRoutingLayer(wid_rect->get_layer_id(), query_box, query_result);
  // find trigger rect
  for (auto& [rtree_box, result_rect] : query_result) {
    if (isTriggerRect(wid_rect, result_rect)) {
      // find a jog or neighbor
      checkTriggerRect_Horizontal(wid_rect, result_rect);
    }
  }
}

void JogSpacingCheck::checkJogSpacing_down(DrcRect* wid_rect)
{
  RTreeBox query_box;
  getParallelQueryBox_down(query_box, wid_rect);
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  _region_query->queryInRoutingLayer(wid_rect->get_layer_id(), query_box, query_result);
  // find trigger rect
  for (auto& [rtree_box, result_rect] : query_result) {
    if (isTriggerRect(wid_rect, result_rect)) {
      // find a jog or neighbor
      checkTriggerRect_Horizontal(wid_rect, result_rect);
    }
  }
}

void JogSpacingCheck::checkJogSpacing_left(DrcRect* wid_rect)
{
  RTreeBox query_box;
  getParallelQueryBox_left(query_box, wid_rect);
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  _region_query->queryInRoutingLayer(wid_rect->get_layer_id(), query_box, query_result);
  // find trigger rect
  for (auto& [rtree_box, result_rect] : query_result) {
    if (isTriggerRect(wid_rect, result_rect)) {
      // find a jog or neighbor
      checkTriggerRect_Vertical(wid_rect, result_rect);
    }
  }
}

void JogSpacingCheck::checkJogSpacing_right(DrcRect* wid_rect)
{
  RTreeBox query_box;
  getParallelQueryBox_right(query_box, wid_rect);
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  _region_query->queryInRoutingLayer(wid_rect->get_layer_id(), query_box, query_result);
  // find trigger rect
  for (auto& [rtree_box, result_rect] : query_result) {
    if (isTriggerRect(wid_rect, result_rect)) {
      // find a jog or neighbor
      checkTriggerRect_Vertical(wid_rect, result_rect);
    }
  }
}

void JogSpacingCheck::checkJogSpacing(DrcRect* wid_rect)
{
  if (_wid_rect_dir_is_horizontal) {
    checkJogSpacing_up(wid_rect);
    checkJogSpacing_down(wid_rect);
  } else {
    checkJogSpacing_left(wid_rect);
    checkJogSpacing_right(wid_rect);
  }
}

}  // namespace idrc
