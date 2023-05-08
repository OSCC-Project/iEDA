#include "NotchSpacingCheck.hpp"

namespace idrc {

// operation api
bool NotchSpacingCheck::check(DrcNet* target_net)
{
  checkNotchSpacing(target_net);
  return _check_result;
}

bool NotchSpacingCheck::check(DrcPoly* target_poly)
{
  checkNotchSpacing(target_poly);
  return _check_result;
}

void NotchSpacingCheck::checkNotchSpacing(DrcNet* target_net)
{
  for (auto& [layer_id, target_polys] : target_net->get_route_polys_list()) {
    for (auto& target_poly : target_polys) {
      checkNotchSpacing(target_poly.get());
    }
  }
}

bool NotchSpacingCheck::isEdgeNotchBottom(DrcEdge* edge)
{
  DrcEdge* pre_edge = edge->getPreEdge();
  DrcEdge* next_edge = edge->getNextEdge();
  EdgeDirection pre_edge_dir = pre_edge->get_edge_dir();
  EdgeDirection next_edge_dir = next_edge->get_edge_dir();
  if (edge->get_edge_dir() == EdgeDirection::kNone || pre_edge_dir == EdgeDirection::kNone || next_edge_dir == EdgeDirection::kNone) {
    std::cout << "Error : Can not identify notch for a none dir edge!" << std::endl;
    return false;
  }
  switch (edge->get_edge_dir()) {
    case EdgeDirection::kNone:
      return false;
    case EdgeDirection::kEast:
      if ((pre_edge_dir == EdgeDirection::kNorth) && (next_edge_dir == EdgeDirection::kSouth)) {
        // Exclude the case of polygon holes
        if (pre_edge->getPreEdge() != next_edge->getPreEdge()) {
          return true;
        }
      }
      return false;
    case EdgeDirection::kWest:
      if ((pre_edge_dir == EdgeDirection::kSouth) && (next_edge_dir == EdgeDirection::kNorth)) {
        if (pre_edge->getPreEdge() != next_edge->getPreEdge()) {
          return true;
        }
      }
      return false;
    case EdgeDirection::kNorth:
      if ((pre_edge_dir == EdgeDirection::kWest) && (next_edge_dir == EdgeDirection::kEast)) {
        if (pre_edge->getPreEdge() != next_edge->getPreEdge()) {
          return true;
        }
      }
      return false;
    case EdgeDirection::kSouth:
      if ((pre_edge_dir == EdgeDirection::kEast) && (next_edge_dir == EdgeDirection::kWest)) {
        if (pre_edge->getPreEdge() != next_edge->getPreEdge()) {
          return true;
        }
      }
      return false;
  }
  return false;
}

void NotchSpacingCheck::checkNotchSpacing(DrcPoly* target_poly)
{
  // test
  int layer_id = target_poly->get_layer_id();
  _lef58_notch_spacing_rule = _tech->get_drc_routing_layer_list()[layer_id]->get_lef58_notch_spacing_rule();
  _notch_spacing_rule = _tech->get_drc_routing_layer_list()[layer_id]->get_notch_spacing_rule();

  // a routing layer cant has more than one notch rule
  if (_notch_spacing_rule.exist() && _lef58_notch_spacing_rule) {
    std::cout << "Error : Routing layer " << layer_id << " has more than one notch rule!" << std::endl;
  }
  // if layer dont have notch rule,skip check
  if ((!_notch_spacing_rule.exist()) && (!_lef58_notch_spacing_rule)) {
    return;
  }
  // A polygon with holes has multiple sets of edges
  for (auto& edges : target_poly->getEdges()) {
    for (auto& edge : edges) {
      if (!isEdgeNotchBottom(edge.get())) {
        continue;
      }
      checkNotchSpacing(edge.get());
    }
  }
}

/**
 * @brief Check whether notch meets the shape requirements in the rule
 *
 * @param edge
 * @return true Meet the shape requirements of notch, check ;
 * @return false notch's shape requirements are not met, skip check ;
 */
bool NotchSpacingCheck::checkNotchShape(DrcEdge* edge)
{
  auto pre_edge = edge->getPreEdge();
  auto next_edge = edge->getNextEdge();
  int min_notch_length = _notch_spacing_rule.get_notch_length();
  // NOTCHLENGTH minNotchLength
  // LEF5.8 Rerferrence
  //**********************************************************************
  // Indicates that any notch with a notch length less than minNotchLength
  // must have notch spacing greater than or equal to minSpacing.
  //**********************************************************************
  if (pre_edge->getLength() < min_notch_length || next_edge->getLength() < min_notch_length) {
    return true;
  }
  return false;
}

void NotchSpacingCheck::getEdgeExtQueryBox(DrcEdge* edge, RTreeBox& rect)
{
  if (edge->get_edge_dir() == EdgeDirection::kNone) {
    std::cout << "Error : Can not genenrate a ExtBox for a none dir edge!" << std::endl;
    return;
  }
  if (edge->get_edge_dir() == EdgeDirection::kEast) {
    bg::set<bg::min_corner, 0>(rect, edge->get_begin_x() + 1);
    bg::set<bg::min_corner, 1>(rect, edge->get_begin_y());
    bg::set<bg::max_corner, 0>(rect, edge->get_end_x() - 1);
    bg::set<bg::max_corner, 1>(rect, edge->get_end_y() + 1);
  } else if (edge->get_edge_dir() == EdgeDirection::kWest) {
    bg::set<bg::min_corner, 0>(rect, edge->get_begin_x() + 1);
    bg::set<bg::min_corner, 1>(rect, edge->get_begin_y() - 1);
    bg::set<bg::max_corner, 0>(rect, edge->get_end_x() - 1);
    bg::set<bg::max_corner, 1>(rect, edge->get_end_y());
  } else if (edge->get_edge_dir() == EdgeDirection::kNorth) {
    bg::set<bg::min_corner, 0>(rect, edge->get_begin_x() - 1);
    bg::set<bg::min_corner, 1>(rect, edge->get_begin_y() + 1);
    bg::set<bg::max_corner, 0>(rect, edge->get_end_x());
    bg::set<bg::max_corner, 1>(rect, edge->get_end_y() - 1);
  } else {  // S
    bg::set<bg::min_corner, 0>(rect, edge->get_begin_x());
    bg::set<bg::min_corner, 1>(rect, edge->get_begin_y() + 1);
    bg::set<bg::max_corner, 0>(rect, edge->get_end_x() + 1);
    bg::set<bg::max_corner, 1>(rect, edge->get_end_y() - 1);
  }
}

int NotchSpacingCheck::getRectOfEdgeMaxWidth(DrcEdge* edge)
{
  int max_width = 0;
  RTreeBox ext_query_box;
  getEdgeExtQueryBox(edge, ext_query_box);

  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  _region_query->queryInRoutingLayer(edge->get_layer_id(), ext_query_box, query_result);
  for (auto& [rtree_box, result_rect] : query_result) {
    int cur_rect_width = getRectWidth(result_rect, edge);
    if (cur_rect_width > max_width) {
      max_width = cur_rect_width;
    }
  }
  return max_width;
}

int NotchSpacingCheck::getRectWidth(DrcRect* rect, DrcEdge* edge)
{
  if (edge->isVertical()) {
    return rect->get_right() - rect->get_left();
  } else {
    return rect->get_top() - rect->get_bottom();
  }
  return 0;
}

bool NotchSpacingCheck::checkLef58NotchSidesWidth(DrcEdge* pre_edge, DrcEdge* next_edge)
{
  int pre_rect_width = getRectOfEdgeMaxWidth(pre_edge);
  int next_rect_width = getRectOfEdgeMaxWidth(next_edge);
  int side_of_notch_width = _lef58_notch_spacing_rule->get_concave_ends_side_of_notch_width().value();
  if (pre_rect_width > side_of_notch_width || next_rect_width > side_of_notch_width) {
    return true;
  }
  return false;
}

/**
 * @brief Check whether notch meets the shape requirements in the rule
 *
 * @param edge
 * @return true Meet the shape requirements of notch, check spacing;
 * @return false notch's shape requirements are not met, skip check;
 */
bool NotchSpacingCheck::checkLef58NotchShape(DrcEdge* edge)
{
  auto pre_edge = edge->getPreEdge();
  auto next_edge = edge->getNextEdge();
  int min_notch_length = _lef58_notch_spacing_rule->get_min_notch_length();
  // CONCAVEENDS  sideOfNotchWidth
  // LEF5.8 Referrence
  //**************************************************************************
  // Specifies the minimum notch length spacing only applies if the width of
  // both sides of the notch must be less than or equal to sideOfNotchWidth.
  // In addition, one of the side edges with length less than minNotchLength
  // must be between two concave corners at the ends, and the length of the opposite
  // edge must be greater than or equal to minNotchLength.
  //**************************************************************************
  if (_lef58_notch_spacing_rule->get_concave_ends_side_of_notch_width().has_value()) {
    // the minimum notch length spacing only applies if the width of
    // both sides of the notch must be less than or equal to sideOfNotchWidth.
    if (checkLef58NotchSidesWidth(pre_edge, next_edge)) {
      return false;
    }
    if (pre_edge->getLength() < min_notch_length) {
      // one of the side edges with length less than minNotchLength
      // must be between two concave corners at the ends
      if (pre_edge->getPreEdge()->get_edge_dir() != edge->get_edge_dir()) {
        // the length of the opposite edge must be greater than or equal to minNotchLength.
        if (next_edge->getLength() >= min_notch_length) {
          return true;
        }
      }
    }
    if (next_edge->getLength() < min_notch_length) {
      if (next_edge->getNextEdge()->get_edge_dir() != edge->get_edge_dir()) {
        if (pre_edge->getLength() >= min_notch_length) {
          return true;
        }
      }
    }
  } else {
    if (pre_edge->getLength() < min_notch_length || next_edge->getLength() < min_notch_length) {
      return true;
    }
  }
  return false;
}

void NotchSpacingCheck::checkNotchSpacingRule(DrcEdge* edge)
{
  int check_spacing;
  if (isNotchRuleLef58()) {
    check_spacing = _lef58_notch_spacing_rule->get_min_spacing();
  } else {
    check_spacing = _notch_spacing_rule.get_min_spacing();
  }
  if (edge->getLength() < check_spacing) {
    if (_interact_with_op) {
      _region_query->addViolation(ViolationType::kNotchSpacing);
      addSpot(edge);
      _check_result = false;
    } else {
      std::cout << "Edge::" << edge->get_begin_x() << "," << edge->get_begin_y() << "NOTCH Violation!!!!!!!!!!!" << std::endl;
      // TODO
      // addSpot();
    }
  }
}

void NotchSpacingCheck::addSpot(DrcEdge* edge)
{
  DrcViolationSpot* spot = new DrcViolationSpot();
  int lb_x = 1e9, lb_y = 1e9, rt_x = 0, rt_y = 0;
  auto next_edge = edge->getNextEdge();
  auto pre_edge = edge->getPreEdge();

  if (edge->isHorizontal()) {
    lb_x = edge->get_min_x();
    rt_x = edge->get_max_x();
    if (std::abs(pre_edge->get_max_y() - edge->get_min_y()) > std::abs(next_edge->get_max_y() - edge->get_min_y())) {
      if (next_edge->get_max_y() - edge->get_min_y() > 0) {
        lb_y = edge->get_min_y();
        rt_y = next_edge->get_max_y();
      } else {
        lb_y = next_edge->get_max_y();
        rt_y = edge->get_min_y();
      }
    } else {
      if (pre_edge->get_max_y() - edge->get_min_y() > 0) {
        lb_y = edge->get_min_y();
        rt_y = pre_edge->get_max_y();
      } else {
        lb_y = pre_edge->get_max_y();
        rt_y = edge->get_min_y();
      }
    }
  }

  int layer_id = edge->get_layer_id();
  spot->set_layer_id(layer_id);
  spot->set_layer_name(_tech->getRoutingLayerNameById(layer_id));
  spot->set_vio_type(ViolationType::kNotchSpacing);
  spot->setCoordinate(lb_x, lb_y, rt_x, rt_y);
  _region_query->_metal_notch_spacing_spot_list.emplace_back(spot);
}

void NotchSpacingCheck::checkNotchSpacing(DrcEdge* edge)
{
  // check Notch side Length and width
  if (isNotchRuleLef58()) {
    if (!checkLef58NotchShape(edge)) {
      return;
    }
  } else {
    if (!checkNotchShape(edge)) {
      return;
    }
  }
  // TODO:
  // "ENDOFNOTCHWIDTH endOfNotchWidth NOTCHSPACING minNotchSpacing NOTCHLENGTH minNotchLength"
  // TODO:
  // LEF58:
  // "EXCEPTWITHIN lowExcludeSpacing highExcludeSpacing
  //  NOTCHWIDTH notchWidth
  //  WIDTH sideOfNotchWidth
  //  WITHIN within SPANLENGTH sideOfNotchSpanLength"
  checkNotchSpacingRule(edge);
}
}  // namespace idrc
