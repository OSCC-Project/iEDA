#include "EOLSpacingCheck.hpp"

#include "BoostType.h"
#include "DrcRules.hpp"

namespace idrc {

// operation api
bool EOLSpacingCheck::check(DrcNet* target_net)
{
  checkEOLSpacing(target_net);
  return _check_result;
}

bool EOLSpacingCheck::check(DrcPoly* poly)
{
  checkEOLSpacing(poly);
  return _check_result;
}

bool EOLSpacingCheck::check(void* scope_owner, DrcRect* rect)
{
  DrcEdge* edge = static_cast<DrcEdge*>(scope_owner);
  int layer_id = edge->get_layer_id();
  _rules = _tech->get_drc_routing_layer_list()[layer_id]->get_lef58_eol_spacing_rule_list();
  return checkEOLSpacing_api(edge, rect);
}

bool EOLSpacingCheck::checkEOLSpacing_api(DrcEdge* edge, DrcRect* rect)
{
  int size = _rules.size();
  for (_rule_index = 0; _rule_index < size; _rule_index++) {
    // check if rule is triggered
    bool trigger = true;
    if (_rules[_rule_index]->get_adj_edge_length().has_value()) {
      // Check whether the adjacent edge meets the length requirement
      // [MAXLENGTH maxLength | MINLENGTH minLength [TWOSIDES]] [EQUALRECTWIDTH]
      if (!isAdjEdgeConsMet(edge)) {
        trigger = false;
        continue;
      }
    }
    if (_rules[_rule_index]->get_parallel_edge().has_value()) {
      // Check whether parallel edge requirements are met
      // [PARALLELEDGE [SUBTRACTEOLWIDTH] parSpace WITHIN parWithin [MINLENGTH minLength] [TWOEDGES]]
      if (!isPRLConsMet(edge)) {
        trigger = false;
        continue;
      }
    }
    if (_rules[_rule_index]->get_enclose_cut().has_value()) {
      // Check whether the neighboring cut requirements are met
      // [ENCLOSECUT [BELOW | ABOVE] encloseDist CUTSPACING cutToMetalSpace] [ALLCUT]
      if (!isCutConsMet(edge)) {
        trigger = false;
        continue;
      }
    }
    // check rule only when all cons are met;
    if (trigger) {
      return checkEOLSpacingHelper_api(edge, rect);
    }
  }
  return true;
}

void EOLSpacingCheck::getPreciseScope(DrcEdge* edge, DrcRect* precise_scope)
{
  if (edge->get_edge_dir() == EdgeDirection::kNone) {
    std::cout << "Error : Can not genenrate a QueryBox for a none dir edge!" << std::endl;
    return;
  }
  int eolWithin = _rules[_rule_index]->get_eol_within().value();
  int eolSpace = _rules[_rule_index]->get_eol_space();
  if (edge->get_edge_dir() == EdgeDirection::kEast) {
    int lb_x = edge->get_begin_x() - eolWithin;
    int lb_y = edge->get_begin_y() - eolSpace;
    int rt_x = edge->get_end_x() + eolWithin;
    int rt_y = edge->get_end_y();
    precise_scope->set_coordinate(lb_x, lb_y, rt_x, rt_y);
  } else if (edge->get_edge_dir() == EdgeDirection::kWest) {
    int lb_x = edge->get_end_x() - eolWithin;
    int lb_y = edge->get_end_y();
    int rt_x = edge->get_begin_x() + eolWithin;
    int rt_y = edge->get_begin_y() + eolSpace;
    precise_scope->set_coordinate(lb_x, lb_y, rt_x, rt_y);
  } else if (edge->get_edge_dir() == EdgeDirection::kNorth) {
    int lb_x = edge->get_begin_x();
    int lb_y = edge->get_begin_y() - eolWithin;
    int rt_x = edge->get_end_x() + eolSpace;
    int rt_y = edge->get_end_y() + eolWithin;
    precise_scope->set_coordinate(lb_x, lb_y, rt_x, rt_y);
  } else {  // S
    int lb_x = edge->get_end_x() - eolSpace;
    int lb_y = edge->get_end_y() - eolWithin;
    int rt_x = edge->get_begin_x();
    int rt_y = edge->get_begin_y() + eolWithin;
    precise_scope->set_coordinate(lb_x, lb_y, rt_x, rt_y);
  }
}

bool EOLSpacingCheck::checkEOLSpacingHelper_api(DrcEdge* edge, DrcRect* rect)
{
  DrcRect precise_scope;
  // getCheckRuleQueryBox(edge, precise_scope);
  getPreciseScope(edge, &precise_scope);
  if (skipCheck(rect, edge, &precise_scope)) {
    return true;
  }
  if (_rules[_rule_index]->get_parallel_edge()->is_same_metal()) {
    if (!isSameMetalMet(DRCUtil::getRTreeBox(rect), edge)) {
      return true;
    }
  }
  return false;
}

void EOLSpacingCheck::checkEOLSpacing(DrcNet* target_net)
{
  for (auto& [layer_id, target_polys] : target_net->get_route_polys_list()) {
    for (auto& target_poly : target_polys) {
      checkEOLSpacing(target_poly.get());
    }
  }
}

void EOLSpacingCheck::checkEOLSpacing(DrcPoly* target_poly)
{
  // test
  int layer_id = target_poly->get_layer_id();
  _rules = _tech->get_drc_routing_layer_list()[layer_id]->get_lef58_eol_spacing_rule_list();

  if (_rules.empty()) {
    return;
  }
  // A polygon with holes has multiple sets of edges
  for (auto& edges : target_poly->getEdges()) {
    for (auto& edge : edges) {
      if (_rules[0]->get_end_to_end().has_value()) {
        //已经检出的END2END violation的边可能会和EOL中有重复
        checkEOLSpacingEnd2End(edge.get());
      }
      checkEOLSpacing(edge.get());
    }
  }
}

bool EOLSpacingCheck::isTwoEdgeOppsite(DrcEdge* edge1, DrcEdge* edge2)
{
  if (edge1->isHorizontal() && edge2->isHorizontal()) {
    return true;
  }
  if (edge1->isVertical() && edge2->isVertical()) {
    return true;
  }
  return false;
}

bool EOLSpacingCheck::isEdgeRectInterectTargetEdge(DrcEdge* edge, DrcEdge* target_edge)
{
  BoostSegment pre_edge = DRCUtil::getBoostSegment(edge->getPreEdge());
  BoostSegment next_edge = DRCUtil::getBoostSegment(edge->getNextEdge());
  BoostSegment target_rt_edge = DRCUtil::getBoostSegment(target_edge);

  // edge intersects with rect edge;
  if (bp::intersects(pre_edge, target_rt_edge) || bp::intersects(next_edge, target_rt_edge)) {
    return true;
  }
  return false;
}

void EOLSpacingCheck::checkEOLSpacingEnd2End(DrcEdge* edge)
{
  if (!isEdgeEOL(edge, true)) {
    return;
  }
  // [OTHERENDWIDTH]
  if (_rules[0]->get_end_to_end()->get_other_end_width().has_value()) {
    std::cout << "Error : the OTHERENDWIDTH field is not supported now!" << std::endl;
  }

  RTreeBox query_box;
  std::vector<std::pair<RTreeSegment, DrcEdge*>> query_result;

  getEnd2EndQueryBox(query_box, edge);

  // Queries the directed edges in query_box region
  _region_query->queryEdgeInRoutingLayer(edge->get_layer_id(), query_box, query_result);

  for (auto& [boost_seg, result_edge] : query_result) {
    //  Rule out three scenarios:
    // 1) Result edge is not EOL
    // 2) The search result edge is not parallel to the target edge direction
    // 3) The adjacent edge of the search result intersects the target edge
    if (!isEdgeEOL(result_edge, true) || (!isTwoEdgeOppsite(edge, result_edge)) || isEdgeRectInterectTargetEdge(edge, result_edge)) {
      continue;
    }
    if (!DRCUtil::intersection(result_edge, query_box, false)) {
      continue;
    }
    auto span_box = DRCUtil::getSpanBoxBetweenTwoEdges(edge, result_edge);
    std::vector<std::pair<RTreeBox, DrcRect*>> span_box_query_result;
    _region_query->queryIntersectsInRoutingLayer(edge->get_layer_id(), span_box, span_box_query_result);
    // _region_query->queryEdgeInRoutingLayer(routingLayerId, span_box, span_box_query_result);
    if (!span_box_query_result.empty()) {
      continue;
    }
    if (_interact_with_op) {
      _region_query->addMetalEOLSpacingViolation(edge->get_layer_id(), span_box);
      // addE2ESpot(edge, result_edge);
      _check_result = false;
    } else {
      storeEnd2EndViolationResult(result_edge, edge);
    }
  }
}

void EOLSpacingCheck::addE2ESpot(DrcEdge* edge, DrcEdge* result_edge)
{
  auto box = DRCUtil::getSpanBoxBetweenTwoEdges(edge, result_edge);
  DrcViolationSpot* spot = new DrcViolationSpot();
  int layer_id = edge->get_layer_id();
  spot->set_layer_id(layer_id);
  spot->set_layer_name(_tech->getRoutingLayerNameById(layer_id));
  spot->set_vio_type(ViolationType::kEOLSpacing);
  spot->setCoordinate(box.min_corner().x(), box.min_corner().y(), box.max_corner().x(), box.max_corner().y());
  _region_query->_metal_eol_spacing_spot_list.emplace_back(spot);
}

void EOLSpacingCheck::storeEnd2EndViolationResult(DrcEdge* result_edge, DrcEdge* edge)
{
  DrcSpot spot;
  spot.set_violation_type(ViolationType::kEnd2EndEOLSpacing);
  spot.add_spot_edge(result_edge);
  spot.add_spot_edge(edge);
  int layer_id = edge->get_layer_id();
  _routing_layer_to_e2e_spacing_spots_list[layer_id].emplace_back(spot);
}

void EOLSpacingCheck::getEnd2EndQueryBox(RTreeBox& query_box, DrcEdge* edge)
{
  if (edge->get_edge_dir() == EdgeDirection::kNone) {
    std::cout << "Error : Can not genenrate a QueryBox for a none dir edge!" << std::endl;
    return;
  }
  int eolWithin = _rules[0]->get_eol_within().value();
  int eolSpace = _rules[0]->get_end_to_end()->get_end_to_end_space();
  if (edge->get_edge_dir() == EdgeDirection::kEast) {
    bg::set<bg::min_corner, 0>(query_box, edge->get_begin_x() - eolWithin);
    bg::set<bg::min_corner, 1>(query_box, edge->get_begin_y() - eolSpace);
    bg::set<bg::max_corner, 0>(query_box, edge->get_end_x() + eolWithin);
    bg::set<bg::max_corner, 1>(query_box, edge->get_end_y());
  } else if (edge->get_edge_dir() == EdgeDirection::kWest) {
    bg::set<bg::min_corner, 0>(query_box, edge->get_end_x() - eolWithin);
    bg::set<bg::min_corner, 1>(query_box, edge->get_end_y());
    bg::set<bg::max_corner, 0>(query_box, edge->get_begin_x() + eolWithin);
    bg::set<bg::max_corner, 1>(query_box, edge->get_begin_y() + eolSpace);
  } else if (edge->get_edge_dir() == EdgeDirection::kNorth) {
    bg::set<bg::min_corner, 0>(query_box, edge->get_begin_x());
    bg::set<bg::min_corner, 1>(query_box, edge->get_begin_y() - eolWithin);
    bg::set<bg::max_corner, 0>(query_box, edge->get_end_x() + eolSpace);
    bg::set<bg::max_corner, 1>(query_box, edge->get_end_y() + eolWithin);
  } else {  // S
    bg::set<bg::min_corner, 0>(query_box, edge->get_end_x() - eolSpace);
    bg::set<bg::min_corner, 1>(query_box, edge->get_end_y() - eolWithin);
    bg::set<bg::max_corner, 0>(query_box, edge->get_begin_x());
    bg::set<bg::max_corner, 1>(query_box, edge->get_begin_y() + eolWithin);
  }
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

bool EOLSpacingCheck::isEdgeEOL(DrcEdge* edge, bool is_end2end)
{
  if (is_end2end) {
    if (edge->getLength() >= _rules[0]->get_eol_width()) {
      return false;
    }
  } else {
    if (edge->getLength() >= _rules[_rule_index]->get_eol_width()) {
      return false;
    }
  }

  // EOL edge must first meet the length requirement
  EdgeDirection pre_edge_dir = edge->getPreEdge()->get_edge_dir();
  EdgeDirection next_edge_dir = edge->getNextEdge()->get_edge_dir();
  if (edge->get_edge_dir() == EdgeDirection::kNone || pre_edge_dir == EdgeDirection::kNone || next_edge_dir == EdgeDirection::kNone) {
    std::cout << "Error : Can not judge eol for a none dir edge!" << std::endl;
    return false;
  }

  // EOL edge must satisfy that the adjacent edge must have a specific orientation
  switch (edge->get_edge_dir()) {
    case EdgeDirection::kEast:
      return (pre_edge_dir == EdgeDirection::kSouth) && (next_edge_dir == EdgeDirection::kNorth);
    case EdgeDirection::kWest:
      return (pre_edge_dir == EdgeDirection::kNorth) && (next_edge_dir == EdgeDirection::kSouth);
    case EdgeDirection::kNorth:
      return (pre_edge_dir == EdgeDirection::kEast) && (next_edge_dir == EdgeDirection::kWest);
    case EdgeDirection::kSouth:
      return (pre_edge_dir == EdgeDirection::kWest) && (next_edge_dir == EdgeDirection::kEast);
    case EdgeDirection::kNone:
      return false;
  }
  return false;
}

void EOLSpacingCheck::checkEOLSpacing(DrcEdge* edge)
{
  // checkHasRoutingShape();
  int size = _rules.size();
  for (_rule_index = 0; _rule_index < size; _rule_index++) {
    if (!isEdgeEOL(edge, false)) {
      return;
    }
    // check if rule is triggered
    bool trigger = true;
    if (_rules[_rule_index]->get_adj_edge_length().has_value()) {
      // Check whether the adjacent edge meets the length requirement
      // [MAXLENGTH maxLength | MINLENGTH minLength [TWOSIDES]] [EQUALRECTWIDTH]
      if (!isAdjEdgeConsMet(edge)) {
        trigger = false;
        continue;
      }
    }
    if (_rules[_rule_index]->get_parallel_edge().has_value()) {
      // Check whether parallel edge requirements are met
      // [PARALLELEDGE [SUBTRACTEOLWIDTH] parSpace WITHIN parWithin [MINLENGTH minLength] [TWOEDGES]]
      if (!isPRLConsMet(edge)) {
        trigger = false;
        continue;
      }
    }
    if (_rules[_rule_index]->get_enclose_cut().has_value()) {
      // Check whether the neighboring cut requirements are met
      // [ENCLOSECUT [BELOW | ABOVE] encloseDist CUTSPACING cutToMetalSpace] [ALLCUT]
      if (!isCutConsMet(edge)) {
        trigger = false;
        continue;
      }
    }
    // check rule only when all cons are met;
    if (trigger) {
      checkEOLSpacingHelper(edge);
    }
  }
}

bool EOLSpacingCheck::isAdjEdgeConsMet(DrcEdge* edge)
{
  //***********************************************************
  // LEF 5.7 Reference:
  // {MAXLENGTH maxLength}
  // Indicates that if the EOL is more than maxLength along both
  // sides, the rule does not apply
  //**********************************************************
  if (_rules[_rule_index]->get_adj_edge_length()->get_max_length().has_value()) {
    int max_length = _rules[_rule_index]->get_adj_edge_length()->get_max_length().value();
    if ((edge->getPreEdge()->getLength() > max_length) && (edge->getNextEdge()->getLength() > max_length)) {
      return false;
    }
  }
  //***********************************************************
  // LEF 5.7 Reference:
  // {MAXLENGTH maxLength}
  // Indicates that if the EOL is more than maxLength along both
  // sides, the rule does not apply
  //**********************************************************
  if (_rules[_rule_index]->get_adj_edge_length()->get_max_length().has_value()) {
    int min_length = _rules[_rule_index]->get_adj_edge_length()->get_min_length().value();
    //***********************************************************
    // LEF 5.7 Reference:
    // {TWOSIDES}
    // Indicates that the rule applies only when the EOL length is
    // greater than and equal to minLength along both sides. In
    // other words, if the EOL length is less than minLength along
    // any one side, the rule does not apply
    //**********************************************************
    if (_rules[_rule_index]->get_adj_edge_length()->is_two_sides()) {
      if ((edge->getPreEdge()->getLength() < min_length) || (edge->getNextEdge()->getLength() < min_length)) {
        return false;
      }
    } else {
      if ((edge->getPreEdge()->getLength() < min_length) && (edge->getNextEdge()->getLength() < min_length)) {
        return false;
      }
    }
  }
  return true;
}

bool EOLSpacingCheck::checkExistPRLEdge_SubWidth(DrcEdge* edge)
{
  // TODO
  return true;
}

bool EOLSpacingCheck::isPRLConsMet(DrcEdge* edge)
{
  //***********************************************************
  // LEF 5.7 Reference:
  // MINLENGTH
  // Indicates that if the EOL length is less than minLength, then
  // any parallel-edge is ignored, and the rule does not apply
  //**********************************************************
  if (_rules[_rule_index]->get_parallel_edge()->get_min_length().has_value()) {
    int min_length = _rules[_rule_index]->get_parallel_edge()->get_min_length().value();
    if (edge->getPreEdge()->getLength() < min_length || edge->getNextEdge()->getLength() < min_length) {
      return false;
    }
  }
  // TODO:SUBTRACTEOLWIDTH
  if (_rules[_rule_index]->get_parallel_edge()->is_subtract_eol_width() == true) {
    // std::cout << "Error: SUBTRACTEOLWIDTH is not supported!" << std::endl;
    if (checkExistPRLEdge_SubWidth(edge)) {
      return true;
    }
  } else {
    bool is_left_has_prl_edge = checkExistPRLEdge(edge, false);
    bool is_right_has_prl_edge = checkExistPRLEdge(edge, true);
    // check whether edge has PRL

    //***********************************************************
    // LEF 5.7 Reference:
    // [TWOEDGES]
    // If TWOEDGES is specified, the EOL rule applies only if there are
    // two parallel edges on each side of the EOL edge that meet the
    // PARALLELEDGE.
    //**********************************************************

    if ((!_rules[_rule_index]->get_parallel_edge()->is_two_edges()) && (is_left_has_prl_edge || is_right_has_prl_edge)) {
      return true;
    }
    if ((_rules[_rule_index]->get_parallel_edge()->is_two_edges()) && is_left_has_prl_edge && is_right_has_prl_edge) {
      return true;
    }
  }

  return false;
}

//***********************************************************
// LEF 5.7 Reference:
// ENCLOSECUT
// Indicates that the rule only applies if there is a cut below or
// above this metal that is less than encloseDist away from the
// EOL edge, and the the cut-edge to metal-edge space beyond
// the EOL edge is less than cutToMetalSpace. If there is
// more than one cut connecting the same metal shapes above
// and below, only one cut needs to meet this rule.
//**********************************************************
bool EOLSpacingCheck::isCutConsMet(DrcEdge* edge)
{
  if (!_rules[_rule_index]->get_enclose_cut()->is_all_cuts()) {
    std::cout << "Error: Cut Class is not supported!" << std::endl;
    return false;
  }
  //***********************************************************
  // LEF 5.7 Reference:
  // [BELOW | ABOVE]
  // If you specify BELOW, encloseDist and
  // cutToMetalSpace are checked for the cut layer below this
  // routing layer. If you specify ABOVE, they are checked for the cut
  // layer above this routing layer. If you specify neither, the rule
  // applies to both adjacent cut layers.
  //**********************************************************
  if (_rules[_rule_index]->get_enclose_cut()->get_direction() == DBEol::Direction::kNone) {
    if (isCutConsMetOneDir(edge, true) && isCutConsMetOneDir(edge, false)) {
      return true;
    }
  } else if (_rules[_rule_index]->get_enclose_cut()->get_direction() == DBEol::Direction::kBelow) {
    if (isCutConsMetOneDir(edge, true)) {
      return true;
    }
  } else {
    if (isCutConsMetOneDir(edge, false)) {
      return true;
    }
  }
  return false;
}

/**
 * @brief generate a CutConstrain QueryBox for an edge
 *
 * @param rule
 * @param query_box
 * @param edge
 */
void EOLSpacingCheck::getCutEncloseDistQueryBox(RTreeBox& query_box, DrcEdge* edge)
{
  int low_x = edge->get_begin_x();
  int low_y = edge->get_begin_y();
  int high_x = edge->get_end_x();
  int high_y = edge->get_end_y();
  int encloseDist = _rules[_rule_index]->get_enclose_cut()->get_enclose_dist();
  if (edge->get_edge_dir() == EdgeDirection::kNone) {
    std::cout << "Error : Can not genenrate a QueryBox for a none dir edge!" << std::endl;
    return;
  }
  if (edge->get_edge_dir() == EdgeDirection::kEast) {
    bg::set<bg::min_corner, 0>(query_box, low_x);
    bg::set<bg::min_corner, 1>(query_box, low_y);
    bg::set<bg::max_corner, 0>(query_box, high_x);
    bg::set<bg::max_corner, 1>(query_box, high_y + encloseDist);
  } else if (edge->get_edge_dir() == EdgeDirection::kWest) {
    bg::set<bg::min_corner, 0>(query_box, high_x);
    bg::set<bg::min_corner, 1>(query_box, high_y - encloseDist);
    bg::set<bg::max_corner, 0>(query_box, low_x);
    bg::set<bg::max_corner, 1>(query_box, low_y);
  } else if (edge->get_edge_dir() == EdgeDirection::kNorth) {
    bg::set<bg::min_corner, 0>(query_box, low_x - encloseDist);
    bg::set<bg::min_corner, 1>(query_box, low_y);
    bg::set<bg::max_corner, 0>(query_box, high_x);
    bg::set<bg::max_corner, 1>(query_box, high_y);
  } else {
    bg::set<bg::min_corner, 0>(query_box, high_x);
    bg::set<bg::min_corner, 1>(query_box, high_y);
    bg::set<bg::max_corner, 0>(query_box, low_x + encloseDist);
    bg::set<bg::max_corner, 1>(query_box, low_y);
  }
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

void EOLSpacingCheck::getCutToMetalSpaceQueryBox(RTreeBox& query_box, DrcEdge* edge, DrcRect* cut_rect)
{
  int low_x = edge->get_begin_x();
  int low_y = edge->get_begin_y();
  int cut2metal_space = _rules[_rule_index]->get_enclose_cut()->get_cut_to_metal_space();
  int enclose_dist = _rules[_rule_index]->get_enclose_cut()->get_enclose_dist();
  if (edge->get_edge_dir() == EdgeDirection::kNone) {
    std::cout << "Error : Can not genenrate a QueryBox for a none dir edge!" << std::endl;
    return;
  }
  if (edge->get_edge_dir() == EdgeDirection::kWest) {
    bg::set<bg::min_corner, 0>(query_box, cut_rect->get_left());
    bg::set<bg::min_corner, 1>(query_box, low_y);
    bg::set<bg::max_corner, 0>(query_box, cut_rect->get_right());
    bg::set<bg::max_corner, 1>(query_box, low_y + cut2metal_space - enclose_dist);
  } else if (edge->get_edge_dir() == EdgeDirection::kEast) {
    bg::set<bg::min_corner, 0>(query_box, cut_rect->get_left());
    bg::set<bg::min_corner, 1>(query_box, low_y - cut2metal_space + enclose_dist);
    bg::set<bg::max_corner, 0>(query_box, cut_rect->get_right());
    bg::set<bg::max_corner, 1>(query_box, low_y);
  } else if (edge->get_edge_dir() == EdgeDirection::kSouth) {
    bg::set<bg::min_corner, 0>(query_box, low_x - cut2metal_space + enclose_dist);
    bg::set<bg::min_corner, 1>(query_box, cut_rect->get_bottom());
    bg::set<bg::max_corner, 0>(query_box, low_x);
    bg::set<bg::max_corner, 1>(query_box, cut_rect->get_top());
  } else {
    bg::set<bg::min_corner, 0>(query_box, low_x);
    bg::set<bg::min_corner, 1>(query_box, cut_rect->get_bottom());
    bg::set<bg::max_corner, 0>(query_box, low_x + cut2metal_space - enclose_dist);
    bg::set<bg::max_corner, 1>(query_box, cut_rect->get_top());
  }
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

bool EOLSpacingCheck::isCutConsMetOneDir(DrcEdge* edge, bool is_below)
{
  RTreeBox query_box;
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  getCutEncloseDistQueryBox(query_box, edge);
  // if only edge touch,not met
  if (is_below) {
    _region_query->queryInCutLayer(edge->get_layer_id(), query_box, query_result);
  } else {
    _region_query->queryInCutLayer(edge->get_layer_id() + 1, query_box, query_result);
  }
  if (!query_result.empty()) {
    for (auto& [boost_rect, cut_rect] : query_result) {
      // Check whether the distance between the cut and the metal in the direction of the edge is less than the specified value;
      if (checkCutToMetalSpace(edge, cut_rect)) {
        return true;
      }
    }
  }
  return false;
}

bool EOLSpacingCheck::checkCutToMetalSpace(DrcEdge* edge, DrcRect* cut)
{
  RTreeBox query_box;
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  getCutToMetalSpaceQueryBox(query_box, edge, cut);
  _region_query->queryInRoutingLayer(edge->get_layer_id(), query_box, query_result);
  //     If there is
  // more than one cut connecting the same metal shapes above
  // and below, only one cut needs to meet this rule
  for (auto& [rt_rect, result_rect] : query_result) {
    // is metal; not only edge touch;
    BoostRect rect1 = DRCUtil::getBoostRect(result_rect);
    BoostRect rect2 = DRCUtil::getBoostRect(query_box);
    // Skip two cases:
    // 1) The query result rectangle is not metal
    // 2) The query result intersects only the edge of the query area
    if ((!result_rect->is_fixed()) && isOnlyEdgeTouch(rect1, rect2)) {
      continue;
    }
    return true;
  }

  return false;
}

bool EOLSpacingCheck::isOnlyEdgeTouch(BoostRect result_rect, BoostRect target_rect)
{
  if (!bp::intersects(result_rect, target_rect, false)) {
    return true;
  }
  return false;
}

bool EOLSpacingCheck::checkExistPRLEdge(DrcEdge* edge, bool is_dir_right)
{
  RTreeBox query_box;
  getPRLQueryBox(edge, is_dir_right, query_box);
  BoostRect query_box_boost = DRCUtil::getBoostRect(query_box);
  std::vector<std::pair<RTreeSegment, DrcEdge*>> query_result;
  _region_query->queryEdgeInRoutingLayer(edge->get_layer_id(), query_box, query_result);
  for (auto& [rt_segment, seg] : query_result) {
    // skip if seg is not prl
    if ((seg->isHorizontal() && edge->isHorizontal()) || (seg->isVertical() && edge->isVertical())) {
      continue;
    }
    // skip if only edge touch(no area overlap)
    BoostRect ext_prl_rect;
    getExtPrlEdgeRect(seg, ext_prl_rect);
    if (!bp::intersects(query_box_boost, ext_prl_rect, false)) {
      continue;
    }
    // samematal
    if (_rules[_rule_index]->get_parallel_edge()->is_same_metal()) {
      if (edge->isHorizontal()) {
        if (seg->get_max_y() >= query_box.max_corner().y() && seg->get_min_y() <= query_box.min_corner().y()) {
          continue;
        }
      } else {
        if (seg->get_max_x() >= query_box.max_corner().x() && seg->get_min_x() <= query_box.min_corner().x()) {
          continue;
        }
      }
    }
    return true;
  }
  return false;
}

void EOLSpacingCheck::getExtPrlEdgeRect(DrcEdge* edge, BoostRect& rect)
{
  if (edge->get_edge_dir() == EdgeDirection::kNone) {
    std::cout << "Error : Can not genenrate a ExtBox for a none dir edge!" << std::endl;
    return;
  }
  if (edge->get_edge_dir() == EdgeDirection::kEast) {
    bp::xl(rect, edge->get_begin_x());
    bp::yl(rect, edge->get_begin_y());
    bp::xh(rect, edge->get_end_x());
    bp::yh(rect, edge->get_end_y() + 1);
  } else if (edge->get_edge_dir() == EdgeDirection::kWest) {
    bp::xl(rect, edge->get_end_x());
    bp::yl(rect, edge->get_end_y() - 1);
    bp::xh(rect, edge->get_begin_x());
    bp::yh(rect, edge->get_begin_y());
  } else if (edge->get_edge_dir() == EdgeDirection::kNorth) {
    bp::xl(rect, edge->get_begin_x() - 1);
    bp::yl(rect, edge->get_begin_y());
    bp::xh(rect, edge->get_end_x());
    bp::yh(rect, edge->get_end_y());
  } else {  // S
    bp::xl(rect, edge->get_end_x());
    bp::yl(rect, edge->get_end_y());
    bp::xh(rect, edge->get_begin_x() + 1);
    bp::yh(rect, edge->get_begin_y());
  }
}

void EOLSpacingCheck::getPRLQueryBox(DrcEdge* edge, bool is_dir_right, RTreeBox& query_box)
{
  if (edge->get_edge_dir() == EdgeDirection::kNone) {
    std::cout << "Error : Can not genenrate a QueryBox for a none dir edge!" << std::endl;
    return;
  }
  int point_x, point_y;
  int eolWithin = _rules[_rule_index]->get_eol_within().value();
  int parWithin = _rules[_rule_index]->get_parallel_edge()->get_par_within();
  int parSpace = _rules[_rule_index]->get_parallel_edge()->get_par_space();

  if (is_dir_right) {
    point_x = edge->get_begin_x();
    point_y = edge->get_begin_y();
    if (edge->get_edge_dir() == EdgeDirection::kEast) {
      bg::set<bg::min_corner, 0>(query_box, point_x - parSpace);
      bg::set<bg::min_corner, 1>(query_box, point_y - eolWithin);
      bg::set<bg::max_corner, 0>(query_box, point_x);
      bg::set<bg::max_corner, 1>(query_box, point_y + parWithin);
    } else if (edge->get_edge_dir() == EdgeDirection::kWest) {
      bg::set<bg::min_corner, 0>(query_box, point_x);
      bg::set<bg::min_corner, 1>(query_box, point_y - parWithin);
      bg::set<bg::max_corner, 0>(query_box, point_x + parSpace);
      bg::set<bg::max_corner, 1>(query_box, point_y + eolWithin);
    } else if (edge->get_edge_dir() == EdgeDirection::kNorth) {
      bg::set<bg::min_corner, 0>(query_box, point_x - parWithin);
      bg::set<bg::min_corner, 1>(query_box, point_y - parSpace);
      bg::set<bg::max_corner, 0>(query_box, point_x + eolWithin);
      bg::set<bg::max_corner, 1>(query_box, point_y);
    } else {  // S
      bg::set<bg::min_corner, 0>(query_box, point_x - eolWithin);
      bg::set<bg::min_corner, 1>(query_box, point_y);
      bg::set<bg::max_corner, 0>(query_box, point_x + parWithin);
      bg::set<bg::max_corner, 1>(query_box, point_y + parSpace);
    }
  } else {
    point_x = edge->get_end_x();
    point_y = edge->get_end_y();
    if (edge->get_edge_dir() == EdgeDirection::kEast) {
      bg::set<bg::min_corner, 0>(query_box, point_x);
      bg::set<bg::min_corner, 1>(query_box, point_y - eolWithin);
      bg::set<bg::max_corner, 0>(query_box, point_x + parSpace);
      bg::set<bg::max_corner, 1>(query_box, point_y + parWithin);
    } else if (edge->get_edge_dir() == EdgeDirection::kWest) {
      bg::set<bg::min_corner, 0>(query_box, point_x - parSpace);
      bg::set<bg::min_corner, 1>(query_box, point_y - parWithin);
      bg::set<bg::max_corner, 0>(query_box, point_x);
      bg::set<bg::max_corner, 1>(query_box, point_y + eolWithin);
    } else if (edge->get_edge_dir() == EdgeDirection::kNorth) {
      bg::set<bg::min_corner, 0>(query_box, point_x - parWithin);
      bg::set<bg::min_corner, 1>(query_box, point_y);
      bg::set<bg::max_corner, 0>(query_box, point_x + eolWithin);
      bg::set<bg::max_corner, 1>(query_box, point_y + parSpace);
    } else {  // S
      bg::set<bg::min_corner, 0>(query_box, point_x - eolWithin);
      bg::set<bg::min_corner, 1>(query_box, point_y - parSpace);
      bg::set<bg::max_corner, 0>(query_box, point_x + parWithin);
      bg::set<bg::max_corner, 1>(query_box, point_y);
    }
  }
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

void EOLSpacingCheck::getCheckRuleQueryBox(DrcEdge* edge, RTreeBox& query_box)
{
  if (edge->get_edge_dir() == EdgeDirection::kNone) {
    std::cout << "Error : Can not genenrate a QueryBox for a none dir edge!" << std::endl;
    return;
  }
  int eolWithin = _rules[_rule_index]->get_eol_within().value();
  int eolSpace = _rules[_rule_index]->get_eol_space();
  if (edge->get_edge_dir() == EdgeDirection::kEast) {
    bg::set<bg::min_corner, 0>(query_box, edge->get_begin_x() - eolWithin);
    bg::set<bg::min_corner, 1>(query_box, edge->get_begin_y() - eolSpace);
    bg::set<bg::max_corner, 0>(query_box, edge->get_end_x() + eolWithin);
    bg::set<bg::max_corner, 1>(query_box, edge->get_end_y());
  } else if (edge->get_edge_dir() == EdgeDirection::kWest) {
    bg::set<bg::min_corner, 0>(query_box, edge->get_end_x() - eolWithin);
    bg::set<bg::min_corner, 1>(query_box, edge->get_end_y());
    bg::set<bg::max_corner, 0>(query_box, edge->get_begin_x() + eolWithin);
    bg::set<bg::max_corner, 1>(query_box, edge->get_begin_y() + eolSpace);
  } else if (edge->get_edge_dir() == EdgeDirection::kNorth) {
    bg::set<bg::min_corner, 0>(query_box, edge->get_begin_x());
    bg::set<bg::min_corner, 1>(query_box, edge->get_begin_y() - eolWithin);
    bg::set<bg::max_corner, 0>(query_box, edge->get_end_x() + eolSpace);
    bg::set<bg::max_corner, 1>(query_box, edge->get_end_y() + eolWithin);
  } else {  // S
    bg::set<bg::min_corner, 0>(query_box, edge->get_end_x() - eolSpace);
    bg::set<bg::min_corner, 1>(query_box, edge->get_end_y() - eolWithin);
    bg::set<bg::max_corner, 0>(query_box, edge->get_begin_x());
    bg::set<bg::max_corner, 1>(query_box, edge->get_begin_y() + eolWithin);
  }
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

void EOLSpacingCheck::checkEOLSpacingHelper(DrcEdge* edge)
{
  // int query_spacing = _rules[_rule_index]->get_eol_space();
  // int check_spacing;
  RTreeBox query_box;
  getCheckRuleQueryBox(edge, query_box);

  DrcRect* query_box_drc_rect = new DrcRect();

  query_box_drc_rect->set_coordinate(query_box.min_corner().x(), query_box.min_corner().y(), query_box.max_corner().x(),
                                     query_box.max_corner().y());

  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;

  _region_query->queryInRoutingLayer(edge->get_layer_id(), query_box, query_result);

  for (auto& [rtree_box, result_rect] : query_result) {
    // *************LEF/DEF 5.7 Language Reference********************
    // LEF Syntax:
    //     For an end-to-end situation when
    // the parallel run length is greater than 0 between two EOL
    // edges, with eolWithin extension on the EOL edge being
    // checked, endToEndSpace is applied
    // ****************************************************************
    // means end2end_condition has higher priority
    if (skipCheck(result_rect, edge, query_box_drc_rect)) {
      continue;
    }
    if (_rules[_rule_index]->get_parallel_edge()->is_same_metal()) {
      if (!isSameMetalMet(rtree_box, edge)) {
        continue;
      }
    }
    auto span_box = DRCUtil::getSpanBoxBetweenEdgeAndRect(edge, result_rect);
    std::vector<std::pair<RTreeBox, DrcRect*>> span_box_query_result;
    _region_query->queryIntersectsInRoutingLayer(edge->get_layer_id(), span_box, span_box_query_result);
    // _region_query->queryEdgeInRoutingLayer(routingLayerId, span_box, span_box_query_result);
    if (!span_box_query_result.empty()) {
      continue;
    }
    // addSpot();
    if (_interact_with_op) {
      _region_query->addMetalEOLSpacingViolation(edge->get_layer_id(), span_box);
      // addSpot(edge, result_rect);
      _check_result = false;
    } else {
      storeEOLViolationResult(result_rect, edge);
    }
  }
}

void EOLSpacingCheck::addSpot(DrcEdge* target_edge, DrcRect* result_rect)
{
  auto box = DRCUtil::getSpanBoxBetweenEdgeAndRect(target_edge, result_rect);
  DrcViolationSpot* spot = new DrcViolationSpot();
  int layer_id = target_edge->get_layer_id();
  spot->set_layer_id(layer_id);
  spot->set_layer_name(_tech->getRoutingLayerNameById(layer_id));
  spot->set_vio_type(ViolationType::kEOLSpacing);
  spot->setCoordinate(box.min_corner().x(), box.min_corner().y(), box.max_corner().x(), box.max_corner().y());
  _region_query->_metal_eol_spacing_spot_list.emplace_back(spot);
}

bool EOLSpacingCheck::isSameMetalMet(RTreeBox result_rect, DrcEdge* edge)
{
  // check left
  RTreeBox query_box;
  getPRLQueryBox(edge, true, query_box);
  BoostRect query_box_boost = DRCUtil::getBoostRect(query_box);
  std::vector<std::pair<RTreeSegment, DrcEdge*>> query_result;
  _region_query->queryEdgeInRoutingLayer(edge->get_layer_id(), query_box, query_result);
  for (auto& [rt_segment, seg] : query_result) {
    // skip if seg is not prl
    if ((seg->isHorizontal() && edge->isHorizontal()) || (seg->isVertical() && edge->isVertical())) {
      continue;
    }
    // skip if only edge touch(no area overlap)
    BoostRect ext_prl_rect;
    getExtPrlEdgeRect(seg, ext_prl_rect);
    if (!bp::intersects(query_box_boost, ext_prl_rect, false)) {
      continue;
    }
    // check if prl rect and result rect is same metal
    BoostRect result_rect_boost = DRCUtil::getBoostRect(result_rect);
    if (bp::intersects(result_rect_boost, ext_prl_rect, false)) {
      return true;
    }
  }
  // check right
  query_result.clear();
  getPRLQueryBox(edge, false, query_box);
  query_box_boost = DRCUtil::getBoostRect(query_box);
  _region_query->queryEdgeInRoutingLayer(edge->get_layer_id(), query_box, query_result);
  for (auto& [rt_segment, seg] : query_result) {
    // skip if seg is not prl
    if ((seg->isHorizontal() && edge->isHorizontal()) || (seg->isVertical() && edge->isVertical())) {
      continue;
    }
    // skip if only edge touch(no area overlap)
    BoostRect ext_prl_rect;
    getExtPrlEdgeRect(seg, ext_prl_rect);
    if (!bp::intersects(query_box_boost, ext_prl_rect, false)) {
      continue;
    }
    // check if prl rect and result rect is same metal
    BoostRect result_rect_boost = DRCUtil::getBoostRect(result_rect);
    if (bp::intersects(result_rect_boost, ext_prl_rect, false)) {
      return true;
    }
  }
  return false;
}

void EOLSpacingCheck::storeEOLViolationResult(DrcRect* rect, DrcEdge* edge)
{
  DrcSpot spot;
  spot.set_violation_type(ViolationType::kEOLSpacing);
  spot.add_eol_vio(rect, edge);
  int layer_id = edge->get_layer_id();
  _routing_layer_to_eol_spacing_spots_list[layer_id].emplace_back(spot);
}

//排除仅边缘交叠
// return ture: 交叠且不仅仅是边；
// return false: 边界交叠以及没有交叠；
bool EOLSpacingCheck::intersectionExceptJustEdgeTouch(DrcRect* result_rect, DrcRect* query_box)
{
  return DRCUtil::intersection(query_box, result_rect, false);
}

bool EOLSpacingCheck::skipCheck(DrcRect* result_rect, DrcEdge* edge, DrcRect* query_box)
{
  // Skip two cases:
  // 1) The query result rectangle intersects with the query area only at the edge
  // 2) The query result rectangle intersects the target edge
  if (!intersectionExceptJustEdgeTouch(result_rect, query_box) || DRCUtil::intersectionWithEdge(result_rect, edge)) {
    return true;
  } else {
    return false;
  }
}

bool EOLSpacingCheck::isTwoEOLHasPrlLength(DrcEdge* result_edge, DrcEdge* edge)
{
  // check dir is prl
  auto result_edge_dir = result_edge->get_edge_dir();
  auto target_edge_dir = edge->get_edge_dir();
  if (result_edge_dir == EdgeDirection::kEast || result_edge_dir == EdgeDirection::kWest) {
    if ((target_edge_dir == EdgeDirection::kNorth) || (target_edge_dir == EdgeDirection::kSouth)) {
      return false;
    } else {
      int result_edge_max_x = std::max(result_edge->get_begin_x(), result_edge->get_end_x());
      int result_edge_min_x = std::min(result_edge->get_begin_x(), result_edge->get_end_x());
      int target_edge_max_x = std::max(edge->get_begin_x(), edge->get_end_x());
      int target_edge_min_x = std::min(edge->get_begin_x(), edge->get_end_x());
      if (!((result_edge_max_x <= target_edge_min_x) || (result_edge_min_x >= target_edge_max_x))) {
        return true;
      }
    }
  } else if (result_edge_dir == EdgeDirection::kNorth || result_edge_dir == EdgeDirection::kSouth) {
    if ((target_edge_dir == EdgeDirection::kEast) || (target_edge_dir == EdgeDirection::kWest)) {
      return false;
    } else {
      int result_edge_max_y = std::max(result_edge->get_begin_y(), result_edge->get_end_y());
      int result_edge_min_y = std::min(result_edge->get_begin_y(), result_edge->get_end_y());
      int target_edge_max_y = std::max(edge->get_begin_y(), edge->get_end_y());
      int target_edge_min_y = std::min(edge->get_begin_y(), edge->get_end_y());
      if (!((result_edge_max_y <= target_edge_min_y) || (result_edge_min_y >= target_edge_max_y))) {
        return true;
      }
    }
  }
  return false;
}

int EOLSpacingCheck::get_eol_violation_num()
{
  int count = 0;
  for (auto& [layerId, short_spot_list] : _routing_layer_to_eol_spacing_spots_list) {
    count += static_cast<int>(short_spot_list.size());
  }
  return count;
}

int EOLSpacingCheck::get_e2e_violation_num()
{
  int count = 0;
  for (auto& [layerId, spacing_spot_list] : _routing_layer_to_e2e_spacing_spots_list) {
    count += spacing_spot_list.size();
  }
  return count;
}

// DrcAPI
void EOLSpacingCheck::getScope(DrcPoly* target_poly, std::vector<DrcRect*>& max_scope_list, bool is_max)
{
  int layer_id = target_poly->get_layer_id();
  _rules = _tech->get_drc_routing_layer_list()[layer_id]->get_lef58_eol_spacing_rule_list();

  if (_rules.empty()) {
    return;
  }
  // A polygon with holes has multiple sets of edges
  for (auto& edges : target_poly->getEdges()) {
    for (auto& edge : edges) {
      if (is_max) {
        _rule_index = _rules.size() - 1;
        if (isEdgeEOL(edge.get(), false)) {
          DrcRect* eol_scope = new DrcRect();
          getEOLSpacingScopeRect(eol_scope, edge.get(), is_max);
          max_scope_list.push_back(eol_scope);
        }
        --_rule_index;
        if (isEdgeEOL(edge.get(), false)) {
          DrcRect* eol_scope = new DrcRect();
          getEOLSpacingScopeRect(eol_scope, edge.get(), is_max);
          max_scope_list.push_back(eol_scope);
        }
      } else {
        _rule_index = 0;
        if (isEdgeEOL(edge.get(), false)) {
          DrcRect* eol_scope = new DrcRect();
          getEOLSpacingScopeRect(eol_scope, edge.get(), is_max);
          max_scope_list.push_back(eol_scope);
        }
      }
    }
  }
}

void EOLSpacingCheck::addScope(DrcPoly* target_poly, bool is_max, RegionQuery* rq)
{
  int layer_id = target_poly->get_layer_id();
  _rules = _tech->get_drc_routing_layer_list()[layer_id]->get_lef58_eol_spacing_rule_list();

  if (_rules.empty()) {
    return;
  }
  // A polygon with holes has multiple sets of edges
  for (auto& edges : target_poly->getEdges()) {
    for (auto& edge : edges) {
      if (is_max) {
        _rule_index = _rules.size() - 1;
        if (isEdgeEOL(edge.get(), false)) {
          DrcRect* eol_scope = new DrcRect();
          getEOLSpacingScopeRect(eol_scope, edge.get(), is_max);
          target_poly->addScope(eol_scope);
          rq->addScopeToMaxScopeRTree(eol_scope);
        }
        --_rule_index;
        if (isEdgeEOL(edge.get(), false)) {
          DrcRect* eol_scope = new DrcRect();
          getEOLSpacingScopeRect(eol_scope, edge.get(), is_max);
          target_poly->addScope(eol_scope);
          rq->addScopeToMaxScopeRTree(eol_scope);
        }
      } else {
        _rule_index = 0;
        if (isEdgeEOL(edge.get(), false)) {
          DrcRect* eol_scope = new DrcRect();
          getEOLSpacingScopeRect(eol_scope, edge.get(), is_max);
          target_poly->addScope(eol_scope);
          rq->addScopeToMinScopeRTree(eol_scope);
        }
      }
    }
  }
}

void EOLSpacingCheck::getEOLSpacingScopeRect(DrcRect* eol_scope, DrcEdge* edge, bool is_max)
{
  eol_scope->setScopeType(ScopeType::EOL);
  eol_scope->set_is_max_scope(is_max);
  eol_scope->set_scope_owner(edge);
  int layer_id = edge->get_layer_id();
  eol_scope->set_layer_id(layer_id);
  int eolWithin = _rules[_rule_index]->get_eol_within().value();
  int eolSpace = _rules[_rule_index]->get_eol_space();
  if (edge->get_edge_dir() == EdgeDirection::kEast) {
    int lb_x = edge->get_begin_x() - eolWithin;
    int lb_y = edge->get_begin_y() - eolSpace;
    int rt_x = edge->get_end_x() + eolWithin;
    int rt_y = edge->get_end_y();
    eol_scope->set_coordinate(lb_x, lb_y, rt_x, rt_y);
  } else if (edge->get_edge_dir() == EdgeDirection::kWest) {
    int lb_x = edge->get_end_x() - eolWithin;
    int lb_y = edge->get_end_y();
    int rt_x = edge->get_begin_x() + eolWithin;
    int rt_y = edge->get_begin_y() + eolSpace;
    eol_scope->set_coordinate(lb_x, lb_y, rt_x, rt_y);
  } else if (edge->get_edge_dir() == EdgeDirection::kNorth) {
    int lb_x = edge->get_begin_x();
    int lb_y = edge->get_begin_y() - eolWithin;
    int rt_x = edge->get_end_x() + eolSpace;
    int rt_y = edge->get_end_y() + eolWithin;
    eol_scope->set_coordinate(lb_x, lb_y, rt_x, rt_y);
  } else {  // S
    int lb_x = edge->get_end_x() - eolSpace;
    int lb_y = edge->get_end_y() - eolWithin;
    int rt_x = edge->get_begin_x();
    int rt_y = edge->get_begin_y() + eolWithin;
    eol_scope->set_coordinate(lb_x, lb_y, rt_x, rt_y);
  }
}

}  // namespace idrc
