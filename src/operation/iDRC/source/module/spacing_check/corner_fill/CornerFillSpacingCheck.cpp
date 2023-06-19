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
#include "CornerFillSpacingCheck.hpp"

#include "DRCUtil.h"

namespace idrc {

bool CornerFillSpacingCheck::check(DrcNet* target_net)
{
  checkCornerFillSpacing(target_net);
  return _check_result;
}

bool CornerFillSpacingCheck::check(DrcPoly* poly)
{
  checkCornerFillSpacing(poly);
  return _check_result;
}

void CornerFillSpacingCheck::checkCornerFillSpacing(DrcNet* target_net)
{
  _corner_fill_rect.set_net_id(target_net->get_net_id());
  for (auto& [layer_id, target_polys] : target_net->get_route_polys_list()) {
    for (auto& target_poly : target_polys) {
      checkCornerFillSpacing(target_poly.get());
    }
  }
}

void CornerFillSpacingCheck::checkCornerFillSpacing(DrcPoly* target_poly)
{
  int layer_id = target_poly->get_layer_id();
  _rule = _tech->get_drc_routing_layer_list()[layer_id]->get_lef58_corner_fill_spacing_rule();

  if (_rule.get() == nullptr) {
    return;
  }
  // A polygon with holes has multiple sets of edges
  for (auto& edges : target_poly->getEdges()) {
    for (auto& edge : edges) {
      if (!isConCaveCornerTriggerMet(edge.get())) {
        continue;
      }
      if (!isLengthTriggerMet(edge.get())) {
        continue;
      }

      if (!isEOLTriggerMet(edge.get())) {
        continue;
      }

      checkCornerFillSpacing(edge.get());
    }
  }
}

void CornerFillSpacingCheck::getCornerFillRect(DrcEdge* edge)
{
  int lb_x, lb_y, rt_x, rt_y;
  auto next_edge = edge->getNextEdge();
  if (edge->isHorizontal()) {
    lb_x = edge->get_min_x();
    rt_x = edge->get_max_x();
    lb_y = next_edge->get_min_y();
    rt_y = next_edge->get_max_y();
  } else {
    lb_x = next_edge->get_min_x();
    rt_x = next_edge->get_max_x();
    lb_y = edge->get_min_y();
    rt_y = edge->get_max_y();
  }
  _corner_fill_rect.set_coordinate(lb_x, lb_y, rt_x, rt_y);
}

void CornerFillSpacingCheck::getQueryBox(DrcEdge* edge, RTreeBox& query_box)
{
  int corner_fill_spacing = _rule->get_spacing();
  if (edge->get_edge_dir() == EdgeDirection::kEast) {
    bg::set<bg::min_corner, 0>(query_box, _corner_fill_rect.get_left() - corner_fill_spacing);
    bg::set<bg::min_corner, 1>(query_box, _corner_fill_rect.get_bottom() - corner_fill_spacing);
    bg::set<bg::max_corner, 0>(query_box, _corner_fill_rect.get_right());
    bg::set<bg::max_corner, 1>(query_box, _corner_fill_rect.get_top());
  } else if (edge->get_edge_dir() == EdgeDirection::kWest) {
    bg::set<bg::min_corner, 0>(query_box, _corner_fill_rect.get_left());
    bg::set<bg::min_corner, 1>(query_box, _corner_fill_rect.get_bottom());
    bg::set<bg::max_corner, 0>(query_box, _corner_fill_rect.get_right() + corner_fill_spacing);
    bg::set<bg::max_corner, 1>(query_box, _corner_fill_rect.get_top() + corner_fill_spacing);
  } else if (edge->get_edge_dir() == EdgeDirection::kNorth) {
    bg::set<bg::min_corner, 0>(query_box, _corner_fill_rect.get_left());
    bg::set<bg::min_corner, 1>(query_box, _corner_fill_rect.get_bottom() - corner_fill_spacing);
    bg::set<bg::max_corner, 0>(query_box, _corner_fill_rect.get_right() + corner_fill_spacing);
    bg::set<bg::max_corner, 1>(query_box, _corner_fill_rect.get_top());
  } else {  // S
    bg::set<bg::min_corner, 0>(query_box, _corner_fill_rect.get_left() - corner_fill_spacing);
    bg::set<bg::min_corner, 1>(query_box, _corner_fill_rect.get_bottom());
    bg::set<bg::max_corner, 0>(query_box, _corner_fill_rect.get_right());
    bg::set<bg::max_corner, 1>(query_box, _corner_fill_rect.get_top() + corner_fill_spacing);
  }
}

bool CornerFillSpacingCheck::intersectionExceptJustEdgeTouch(RTreeBox* query_rect, DrcRect* result_rect)
{
  return DRCUtil::intersection(query_rect, result_rect, false);
}

bool CornerFillSpacingCheck::isSameNetRectConnect(DrcRect* result_rect)
{
  return DRCUtil::intersection(&_corner_fill_rect, result_rect, true) && (_corner_fill_rect.get_net_id() == result_rect->get_net_id());
}

bool CornerFillSpacingCheck::skipCheck(DrcRect* result_rect)
{
  //跳过两种情况：同一个矩形，同一个net且相交
  return (DRCUtil::isSameRect(result_rect, &_corner_fill_rect)) || isSameNetRectConnect(result_rect);
}

bool CornerFillSpacingCheck::checkShort(DrcRect* result_rect)
{
  if ((DRCUtil::intersection(&_corner_fill_rect, result_rect, true)) && (_corner_fill_rect.get_net_id() != result_rect->get_net_id())) {
    if (_interact_with_op) {
      _check_result = false;
      return true;
    } else {
      std::cout << "cornerfill spacing vio " << std::endl;
    }
  }
  return false;
}

bool CornerFillSpacingCheck::isParallelOverlap(DrcRect* result_rect)
{
  return DRCUtil::isParallelOverlap(&_corner_fill_rect, result_rect);
}

void CornerFillSpacingCheck::checkCornerSpacing(DrcRect* result_rect)
{
  int required_spacing = _rule->get_spacing();
  int distanceX = std::min(std::abs(_corner_fill_rect.get_left() - result_rect->get_right()),
                           std::abs(_corner_fill_rect.get_right() - result_rect->get_left()));
  int distanceY = std::min(std::abs(_corner_fill_rect.get_bottom() - result_rect->get_top()),
                           std::abs(_corner_fill_rect.get_top() - result_rect->get_bottom()));
  if (required_spacing * required_spacing > distanceX * distanceX + distanceY * distanceY) {
    if (_interact_with_op) {
      _region_query->addViolation(ViolationType::kCornerFillingSpacing);
      addSpot(result_rect);
      _check_result = false;
    } else {
      std::cout << "cornerfill spacing vio " << std::endl;
    }
  }
}

void CornerFillSpacingCheck::checkXYSpacing(DrcRect* result_rect)
{
  RTreeBox span_box = DRCUtil::getSpanBoxBetweenTwoRects(&_corner_fill_rect, result_rect);
  // bool isHorizontalParallelOverlap = false;
  int spacing = -1;
  int lb_x = span_box.min_corner().get<0>();
  int lb_y = span_box.min_corner().get<1>();
  int rt_x = span_box.max_corner().get<0>();
  int rt_y = span_box.max_corner().get<1>();
  if (DRCUtil::isHorizontalParallelOverlap(&_corner_fill_rect, result_rect)) {
    // isHorizontalParallelOverlap = true;
    spacing = std::abs(rt_y - lb_y);
  } else {
    spacing = std::abs(rt_x - lb_x);
  }

  if (spacing < _rule->get_spacing()) {
    if (_interact_with_op) {
      _region_query->addViolation(ViolationType::kCornerFillingSpacing);
      addSpot(result_rect);
      _check_result = false;
    } else {
      std::cout << "spacing vio " << std::endl;
    }
  }
}

void CornerFillSpacingCheck::addSpot(DrcRect* result_rect)
{
  auto box = DRCUtil::getSpanBoxBetweenTwoRects(&_corner_fill_rect, result_rect);
  DrcViolationSpot* spot = new DrcViolationSpot();
  int layer_id = result_rect->get_layer_id();
  spot->set_layer_id(layer_id);
  spot->set_layer_name(_tech->getCutLayerNameById(layer_id));
  spot->set_net_id(_corner_fill_rect.get_net_id());
  spot->set_vio_type(ViolationType::kCornerFillingSpacing);
  spot->setCoordinate(box.min_corner().x(), box.min_corner().y(), box.max_corner().x(), box.max_corner().y());
  _region_query->_metal_corner_fill_spacing_spot_list.emplace_back(spot);
}

void CornerFillSpacingCheck::checkSpacing(DrcRect* result_rect)
{
  RTreeBox span_box = DRCUtil::getSpanBoxBetweenTwoRects(&_corner_fill_rect, result_rect);
  std::vector<std::pair<RTreeBox, DrcRect*>> span_box_query_result;
  _region_query->queryInRoutingLayer(result_rect->get_layer_id(), span_box, span_box_query_result);
  if (!span_box_query_result.empty()) {
    return;
  }
  if (!isParallelOverlap(result_rect)) {
    // case no Parallel Overlap between two rect ,need check corner spacing
    // if corner spacing is not meet require_spacing,it is a violation
    //如果两个矩形不存在平行交叠则检查角间距
    checkCornerSpacing(result_rect);
  } else {
    // There is  Parallel Overlap between two rect
    // need check span box is covered by exited rect
    //存在平行交叠检查X或Y方向上的间距
    checkXYSpacing(result_rect);
  }
}

void CornerFillSpacingCheck::checkSpacingToRect(DrcRect* result_rect)
{
  if (checkShort(result_rect)) {
    return;
  }
  checkSpacing(result_rect);
}

void CornerFillSpacingCheck::checkCornerFillSpacing(DrcEdge* edge)
{
  getCornerFillRect(edge);

  RTreeBox query_box;
  getQueryBox(edge, query_box);
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  _region_query->queryInRoutingLayer(edge->get_layer_id(), query_box, query_result);
  for (auto& [rtree_box, result_rect] : query_result) {
    if (skipCheck(result_rect)) {
      continue;
    }
    if (intersectionExceptJustEdgeTouch(&query_box, result_rect)) {
      checkSpacingToRect(result_rect);
    }
  }
}

bool CornerFillSpacingCheck::isEdgeEOL(DrcEdge* edge)
{
  // EOL edge must first meet the length requirement
  if (edge->getLength() >= _rule->get_eol_width()) {
    return false;
  }
  EdgeDirection pre_edge_dir = edge->getPreEdge()->get_edge_dir();
  EdgeDirection next_edge_dir = edge->getNextEdge()->get_edge_dir();
  if (edge->get_edge_dir() == EdgeDirection::kNone || pre_edge_dir == EdgeDirection::kNone || next_edge_dir == EdgeDirection::kNone) {
    std::cout << "Error : Can not judge eol for a none dir edge!" << std::endl;
    return false;
  }
  // EOL edge must satisfy that the adjacent edge must have a specific orientation
  switch (edge->get_edge_dir()) {
    case EdgeDirection::kNone:
      break;
    case EdgeDirection::kEast:
      return (pre_edge_dir == EdgeDirection::kSouth) && (next_edge_dir == EdgeDirection::kNorth);
    case EdgeDirection::kWest:
      return (pre_edge_dir == EdgeDirection::kNorth) && (next_edge_dir == EdgeDirection::kSouth);
    case EdgeDirection::kNorth:
      return (pre_edge_dir == EdgeDirection::kEast) && (next_edge_dir == EdgeDirection::kWest);
    case EdgeDirection::kSouth:
      return (pre_edge_dir == EdgeDirection::kWest) && (next_edge_dir == EdgeDirection::kEast);
  }
  return false;
}

bool CornerFillSpacingCheck::isEOLTriggerMet(DrcEdge* edge)
{
  DrcEdge* adj_edge;
  if (_is_edge_length2) {
    adj_edge = edge->getPreEdge();
  } else {
    adj_edge = edge->getNextEdge()->getNextEdge();
  }
  if (isEdgeEOL(adj_edge)) {
    return true;
  }
  return false;
}

bool CornerFillSpacingCheck::isLengthTriggerMet(DrcEdge* edge)
{
  int length1 = _rule->get_edge_length1();
  int length2 = _rule->get_edge_length2();
  auto next_edge = edge->getNextEdge();
  if (edge->getLength() < length1 && next_edge->getLength() < length2) {
    _is_edge_length2 = false;
    return true;
  }
  if (edge->getLength() < length2 && next_edge->getLength() < length1) {
    _is_edge_length2 = true;
    return true;
  }
  return false;
}

bool CornerFillSpacingCheck::isConCaveCornerTriggerMet(DrcEdge* edge)
{
  return DRCUtil::isCornerConCave(edge);
}

void CornerFillSpacingCheck::getScopeRect(DrcEdge* edge, DrcRect* scope_rect)
{
  scope_rect->set_layer_id(edge->get_layer_id());
  scope_rect->set_scope_owner(edge);
  scope_rect->setScopeType(ScopeType::CornerFill);
  // scope_rect->set_is_max_scope();
  int corner_fill_spacing = _rule->get_spacing();
  if (edge->get_edge_dir() == EdgeDirection::kEast) {
    int lb_x = _corner_fill_rect.get_left() - corner_fill_spacing;
    int lb_y = _corner_fill_rect.get_bottom() - corner_fill_spacing;
    int rt_x = _corner_fill_rect.get_right();
    int rt_y = _corner_fill_rect.get_top();
    scope_rect->set_coordinate(lb_x, lb_y, rt_x, rt_y);
  } else if (edge->get_edge_dir() == EdgeDirection::kWest) {
    int lb_x = _corner_fill_rect.get_left();
    int lb_y = _corner_fill_rect.get_bottom();
    int rt_x = _corner_fill_rect.get_right() + corner_fill_spacing;
    int rt_y = _corner_fill_rect.get_top() + corner_fill_spacing;
    scope_rect->set_coordinate(lb_x, lb_y, rt_x, rt_y);
  } else if (edge->get_edge_dir() == EdgeDirection::kNorth) {
    int lb_x = _corner_fill_rect.get_left();
    int lb_y = _corner_fill_rect.get_bottom() - corner_fill_spacing;
    int rt_x = _corner_fill_rect.get_right() + corner_fill_spacing;
    int rt_y = _corner_fill_rect.get_top();
    scope_rect->set_coordinate(lb_x, lb_y, rt_x, rt_y);
  } else {  // S
    int lb_x = _corner_fill_rect.get_left() - corner_fill_spacing;
    int lb_y = _corner_fill_rect.get_bottom();
    int rt_x = _corner_fill_rect.get_right();
    int rt_y = _corner_fill_rect.get_top() + corner_fill_spacing;
    scope_rect->set_coordinate(lb_x, lb_y, rt_x, rt_y);
  }
}

void CornerFillSpacingCheck::getScopeOfEdge(DrcEdge* edge, std::vector<DrcRect*>& max_scope_list)
{
  getCornerFillRect(edge);
  DrcRect* scope_rect = new DrcRect();
  getScopeRect(edge, scope_rect);
  max_scope_list.push_back(scope_rect);
}

void CornerFillSpacingCheck::getScope(DrcPoly* target_poly, std::vector<DrcRect*>& max_scope_list)
{
  int layer_id = target_poly->get_layer_id();
  _rule = _tech->get_drc_routing_layer_list()[layer_id]->get_lef58_corner_fill_spacing_rule();

  if (_rule.get() == nullptr) {
    return;
  }
  // A polygon with holes has multiple sets of edges
  for (auto& edges : target_poly->getEdges()) {
    for (auto& edge : edges) {
      if (!isConCaveCornerTriggerMet(edge.get())) {
        continue;
      }
      if (!isLengthTriggerMet(edge.get())) {
        continue;
      }

      if (!isEOLTriggerMet(edge.get())) {
        continue;
      }
      getScopeOfEdge(edge.get(), max_scope_list);
    }
  }
}

void CornerFillSpacingCheck::addScope(DrcPoly* target_poly, RegionQuery* rq)
{
  int layer_id = target_poly->get_layer_id();
  _rule = _tech->get_drc_routing_layer_list()[layer_id]->get_lef58_corner_fill_spacing_rule();

  if (_rule.get() == nullptr) {
    return;
  }
  // A polygon with holes has multiple sets of edges
  for (auto& edges : target_poly->getEdges()) {
    for (auto& edge : edges) {
      if (!isConCaveCornerTriggerMet(edge.get())) {
        continue;
      }
      if (!isLengthTriggerMet(edge.get())) {
        continue;
      }

      if (!isEOLTriggerMet(edge.get())) {
        continue;
      }
      getCornerFillRect(edge.get());
      DrcRect* scope_rect = new DrcRect();
      getScopeRect(edge.get(), scope_rect);

      DrcRect* _corner_fill_scope = new DrcRect(_corner_fill_rect);
      target_poly->addScope(_corner_fill_scope);
      rq->addScopeToMaxScopeRTree(_corner_fill_scope);
      rq->addScopeToMinScopeRTree(_corner_fill_scope);
    }
  }
}
}  // namespace idrc