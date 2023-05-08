#include "EnclosureCheck.hpp"

#include "DRCUtil.h"
#include "DrcConflictGraph.h"
#include "DrcDesign.h"
#include "DrcRules.hpp"
#include "RegionQuery.h"
#include "Tech.h"

namespace idrc {

bool EnclosureCheck::check(DrcRect* target_rect)
{
  if (target_rect->get_owner_type() != RectOwnerType::kViaCut) {
    std::cout << "[Drc EnclosureCheck Warning] : Can not check enclosure for a rect is not cut,check skipped!" << std::endl;
    return false;
  }
  _check_result = true;
  checkEnclosure(target_rect);
  return _check_result;
}

/**
 * @brief Initializes the Enlosure violation list from the R tree
 *
 */
void EnclosureCheck::initEnclosureSpotListFromRtree()
{
  for (auto& [layerId, rtree] : _layer_to_violation_box_tree) {
    for (auto it = rtree.begin(); it != rtree.end(); ++it) {
      RTreeBox box = *it;
      DrcRectangle<int> rect = DRCUtil::getRectangleFromRTreeBox(box);

      DrcSpot spot;
      spot.set_violation_type(ViolationType::kEnclosure);

      DrcRect* drc_rect = new DrcRect();
      drc_rect->set_owner_type(RectOwnerType::kSpotMark);
      drc_rect->set_layer_id(layerId);
      drc_rect->set_rectangle(rect);

      spot.add_spot_rect(drc_rect);
      _cut_layer_to_enclosure_spots_list[layerId].emplace_back(spot);
    }
  }
}

/**
 * @brief Initializing the Enclosure Check module

 * @param config
 * @param tech
 */
void EnclosureCheck::init(DrcConfig* config, Tech* tech, RegionQuery* region_query)
{
  _config = config;
  _tech = tech;
  _region_query = region_query;
}

/**
 * @brief Check each line network Enclosure
 *
 * @param target_net
 */
void EnclosureCheck::checkEnclosure(DrcNet* target_net)
{
  for (auto& [layerId, routing_rect_list] : target_net->get_layer_to_cut_rects_map()) {
    for (auto target_rect : routing_rect_list) {
      checkEnclosure(target_rect);
    }
  }

  // initEnclosureSpotListFromRtree();
  // _layer_to_violation_box_tree.clear();
}

// /**
//  * @brief Check that the cut corresponds to the target rectangle and report the violation
//  *
//  * @param cutLayerId
//  * @param target_rect
//  * @param below_enclosure_rule_list
//  */
// void EnclosureCheck::checkBelowEnclosureRule(int cutLayerId, DrcRect* target_rect, std::vector<EnclosureRule*> below_enclosure_rule_list)
// {
//   for (auto& enclosure_rule : below_enclosure_rule_list) {
//     int layer_require_overhang1 = enclosure_rule->getOverhang1();
//     int layer_require_overhang2 = enclosure_rule->getOverhang2();
//     std::vector<RTreeBox> query_box_list;
//     getEnclosureQueryBox(target_rect, layer_require_overhang1, layer_require_overhang2, query_box_list);
//     for (auto& query_box : query_box_list) {
//       std::vector<std::pair<RTreeBox, DrcRect*>> query_result_below = getQueryResult(cutLayerId, query_box, enclosure_rule, false, true);
//       if (checkEnclosureFromQueryResult(cutLayerId, target_rect, enclosure_rule, query_result_below)) {
//         return;
//       }
//     }
//   }
//   storeViolationResult(cutLayerId, target_rect);
// }

// /**
//  * @brief Check the target rectangle for top bounding and report violations
//  *
//  * @param cutLayerId
//  * @param target_rect
//  * @param above_enclosure_rule_list
//  */
// void EnclosureCheck::checkAboveEnclosureRule(int cutLayerId, DrcRect* target_rect, std::vector<EnclosureRule*> above_enclosure_rule_list)
// {
//   for (auto& enclosure_rule : above_enclosure_rule_list) {
//     int layer_require_overhang1 = enclosure_rule->getOverhang1();
//     int layer_require_overhang2 = enclosure_rule->getOverhang2();
//     std::vector<RTreeBox> query_box_list;
//     getEnclosureQueryBox(target_rect, layer_require_overhang1, layer_require_overhang2, query_box_list);
//     for (auto& query_box : query_box_list) {
//       std::vector<std::pair<RTreeBox, DrcRect*>> query_result_above = getQueryResult(cutLayerId, query_box, enclosure_rule, true, false);
//       // return true,met this rule
//       if (checkEnclosureFromQueryResult(cutLayerId, target_rect, enclosure_rule, query_result_above)) {
//         return;
//       }
//     }
//   }
//   storeViolationResult(cutLayerId, target_rect);
// }

void EnclosureCheck::getAboveMetalRectList(DrcRect* target_cut_rect, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result)
{
  int layer_id = target_cut_rect->get_layer_id();
  // cut上层金属的layer_id与其自身layer_id相同
  _region_query->queryEnclosureInRoutingLayer(layer_id, DRCUtil::getRTreeBox(target_cut_rect), query_result);
}

void EnclosureCheck::getBelowMetalRectList(DrcRect* target_cut_rect, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result)
{
  int layer_id = target_cut_rect->get_layer_id();
  // cut上层金属的layer_id与其自身layer_id相同
  _region_query->queryEnclosureInRoutingLayer(layer_id - 1, DRCUtil::getRTreeBox(target_cut_rect), query_result);
}

bool EnclosureCheck::checkOverhang_below(std::vector<std::pair<RTreeBox, DrcRect*>>& below_metal_rect_list, DrcRect* target_cut_rect)
{
  int cut_left = target_cut_rect->get_left();
  int cut_right = target_cut_rect->get_right();
  int cut_top = target_cut_rect->get_top();
  int cut_bottom = target_cut_rect->get_bottom();
  for (auto [rtree_box, above_metal_rect] : below_metal_rect_list) {
    int metal_left = above_metal_rect->get_left();
    int metal_right = above_metal_rect->get_right();
    int metal_bottom = above_metal_rect->get_bottom();
    int metal_top = above_metal_rect->get_top();

    if (cut_left - metal_left > _left_overhang_below) {
      _left_overhang_below = cut_left - metal_left;
      _left_overhang_rect_below = above_metal_rect;
    }
    if (metal_right - cut_right > _right_overhang_below) {
      _right_overhang_below = metal_right - cut_right;
      _right_overhang_rect_below = above_metal_rect;
    }
    if (cut_bottom - metal_bottom > _bottom_overhang_below) {
      _bottom_overhang_below = cut_bottom - metal_bottom;
      _bottom_overhang_rect_below = above_metal_rect;
    }
    if (metal_top - cut_top > _top_overhang_below) {
      _top_overhang_below = metal_top - cut_top;
      _top_overhang_rect_below = above_metal_rect;
    }
  }

  for (auto enclosure_rule : _lef58_enclosure_list) {
    auto target_cut_class_name = getCutClassName(target_cut_rect);
    auto rule_cut_class_name = enclosure_rule->get_class_name();
    if (target_cut_class_name.compare(rule_cut_class_name) != 0) {
      continue;
    }
    if (enclosure_rule->get_overhang1().has_value()) {
      int overhang1 = enclosure_rule->get_overhang1().value();
      int overhang2 = enclosure_rule->get_overhang2().value();
      if ((_left_overhang_below >= overhang1 && _right_overhang_below >= overhang1 && _top_overhang_below >= overhang2
           && _bottom_overhang_below >= overhang2)
          || (_left_overhang_below >= overhang2 && _right_overhang_below >= overhang2 && _top_overhang_below >= overhang1
              && _bottom_overhang_below >= overhang1)) {
        return true;
      }
    }
    if (enclosure_rule->get_end_overhang1().has_value()) {
      int end_overhang1 = enclosure_rule->get_end_overhang1().value();
      int side_overhang2 = enclosure_rule->get_side_overhang2().value();
      if (target_cut_rect->isHorizontal()) {
        if (_left_overhang_below >= end_overhang1 && _right_overhang_below >= end_overhang1 && _top_overhang_below >= side_overhang2
            && _bottom_overhang_below >= side_overhang2) {
          return true;
        }
      }
      if (target_cut_rect->isVertical()) {
        if (_left_overhang_below >= side_overhang2 && _right_overhang_below >= side_overhang2 && _top_overhang_below >= end_overhang1
            && _bottom_overhang_below >= end_overhang1) {
          return true;
        }
      }
    }
  }
  return false;
}

bool EnclosureCheck::checkOverhang_above(std::vector<std::pair<RTreeBox, DrcRect*>>& above_metal_rect_list, DrcRect* target_cut_rect)
{
  int cut_left = target_cut_rect->get_left();
  int cut_right = target_cut_rect->get_right();
  int cut_top = target_cut_rect->get_top();
  int cut_bottom = target_cut_rect->get_bottom();
  for (auto [rtree_box, above_metal_rect] : above_metal_rect_list) {
    int metal_left = above_metal_rect->get_left();
    int metal_right = above_metal_rect->get_right();
    int metal_bottom = above_metal_rect->get_bottom();
    int metal_top = above_metal_rect->get_top();

    if (cut_left - metal_left > _left_overhang_above) {
      _left_overhang_above = cut_left - metal_left;
      _left_overhang_rect_above = above_metal_rect;
    }
    if (metal_right - cut_right > _right_overhang_above) {
      _right_overhang_above = metal_right - cut_right;
      _right_overhang_rect_above = above_metal_rect;
    }
    if (cut_bottom - metal_bottom > _bottom_overhang_above) {
      _bottom_overhang_above = cut_bottom - metal_bottom;
      _bottom_overhang_rect_above = above_metal_rect;
    }
    if (metal_top - cut_top > _top_overhang_above) {
      _top_overhang_above = metal_top - cut_top;
      _top_overhang_rect_above = above_metal_rect;
    }
  }

  for (auto enclosure_rule : _lef58_enclosure_list) {
    auto target_cut_class_name = getCutClassName(target_cut_rect);
    auto rule_cut_class_name = enclosure_rule->get_class_name();
    if (target_cut_class_name.compare(rule_cut_class_name) != 0) {
      continue;
    }
    if (enclosure_rule->get_overhang1().has_value()) {
      int overhang1 = enclosure_rule->get_overhang1().value();
      int overhang2 = enclosure_rule->get_overhang2().value();
      if ((_left_overhang_above >= overhang1 && _right_overhang_above >= overhang1 && _top_overhang_above >= overhang2
           && _bottom_overhang_above >= overhang2)
          || (_left_overhang_above >= overhang2 && _right_overhang_above >= overhang2 && _top_overhang_above >= overhang1
              && _bottom_overhang_above >= overhang1)) {
        return true;
      }
    }
    if (enclosure_rule->get_end_overhang1().has_value()) {
      int end_overhang1 = enclosure_rule->get_end_overhang1().value();
      int side_overhang2 = enclosure_rule->get_side_overhang2().value();
      if (target_cut_rect->isHorizontal()) {
        if (_left_overhang_above >= end_overhang1 && _right_overhang_above >= end_overhang1 && _top_overhang_above >= side_overhang2
            && _bottom_overhang_above >= side_overhang2) {
          return true;
        }
      }
      if (target_cut_rect->isVertical()) {
        if (_left_overhang_above >= side_overhang2 && _right_overhang_above >= side_overhang2 && _top_overhang_above >= end_overhang1
            && _bottom_overhang_above >= end_overhang1) {
          return true;
        }
      }
    }
  }
  return false;
}

std::string EnclosureCheck::getCutClassName(DrcRect* cut_rect)
{
  int size = _lef58_cut_class_list.size();
  for (int index = 0; index < size; index++) {
    if (cut_rect->getWidth() == _lef58_cut_class_list[index]->get_via_width()
        && cut_rect->getLength() == _lef58_cut_class_list[index]->get_via_length()) {
      return _lef58_cut_class_list[index]->get_class_name();
    }
  }
  std::cout << "[DRC Enclosure Check Warning] : Unkown Cut Class!" << std::endl;
  return std::string("");
}

/**
 * @brief Check the bounding of the target rectangle
 *
 * @param target_rect
 */
void EnclosureCheck::checkEnclosure(DrcRect* target_cut_rect)
{
  int layer_id = target_cut_rect->get_layer_id();
  reFresh();
  _lef58_enclosure_list = _tech->get_drc_cut_layer_list()[layer_id]->get_lef58_enclosure_list();
  _lef58_cut_class_list = _tech->get_drc_cut_layer_list()[layer_id]->get_lef58_cut_class_list();
  _lef58_enclosure_edge_list = _tech->get_drc_cut_layer_list()[layer_id]->get_lef58_enclosure_edge_list();

  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  // check above metal;
  getAboveMetalRectList(target_cut_rect, query_result);
  if (!query_result.empty()) {
    std::vector<std::pair<RTreeSegment, DrcEdge*>> edges_query_result;
    int layer_id = target_cut_rect->get_layer_id();
    int size = query_result.size();
    for (int i = 0; i < size; i++) {
      _region_query->queryEdgeInRoutingLayer(layer_id, query_result[i].first, edges_query_result);
      if (!edges_query_result.empty()) {
        break;
      }
    }
    if (!edges_query_result.empty()) {
      int size = edges_query_result.size();
      for (int i = 0; i < size; i++) {
        //避免取到short的poly
        if (edges_query_result[i].second->get_owner_polygon()->getNetId() == target_cut_rect->get_net_id()) {
          _cut_below_poly = edges_query_result[i].second->get_owner_polygon();
          break;
        }
      }
    } else {
      std::cout << "[DRC CutEnclosure Warning]: Get cut below poly failed!" << std::endl;
    }
  } else {
    std::cout << "[DRC CutEnclosure Warning]: Get cut below rect failed!" << std::endl;
  }

  // if (!above_metal_rect) {
  //   std::cout << "[DRC CutEolSpacingCheck Warning]:cut has no above metal" << std::endl;
  // }
  if (!checkOverhang_above(query_result, target_cut_rect)) {
    _check_result = false;
    // std::cout << "above_enclosure vio!!!!" << std::endl;
    _region_query->addViolation(ViolationType::kEnclosure);
    addSpot(target_cut_rect);
    // return;
  }

  // check below metal
  query_result.clear();
  getBelowMetalRectList(target_cut_rect, query_result);
  if (!query_result.empty()) {
    std::vector<std::pair<RTreeSegment, DrcEdge*>> edges_query_result;
    int layer_id = target_cut_rect->get_layer_id();
    int size = query_result.size();
    for (int i = 0; i < size; i++) {
      _region_query->queryEdgeInRoutingLayer(layer_id - 1, query_result[i].first, edges_query_result);
      if (!edges_query_result.empty()) {
        break;
      }
    }
    if (!edges_query_result.empty()) {
      int size = edges_query_result.size();
      for (int i = 0; i < size; i++) {
        //避免取到short的poly
        if (edges_query_result[i].second->get_owner_polygon()->getNetId() == target_cut_rect->get_net_id()) {
          _cut_below_poly = edges_query_result[i].second->get_owner_polygon();
          break;
        }
      }
    } else {
      std::cout << "[DRC CutEnclosure Warning]: Get cut below poly failed!" << std::endl;
    }
  } else {
    std::cout << "[DRC CutEnclosure Warning]: Get cut below rect failed!" << std::endl;
  }

  // if (!above_metal_rect) {
  //   std::cout << "[DRC CutEolSpacingCheck Warning]:cut has no above metal" << std::endl;
  // }
  if (!checkOverhang_below(query_result, target_cut_rect)) {
    _check_result = false;
    // std::cout << "below_enclosure vio!!!!" << std::endl;
    _region_query->addViolation(ViolationType::kEnclosure);
    addSpot(target_cut_rect);
    // return;
  }

  if (!checkEdgeEnclosure(target_cut_rect)) {
    _check_result = false;
    // std::cout << "edge enclosure vio!!" << std::endl;
    _region_query->addViolation(ViolationType::kEnclosureEdge);
    addEdgeEnclosureSpot(target_cut_rect);
  }
}

void EnclosureCheck::addSpot(DrcRect* target_cut_rect)
{
  auto box = DRCUtil::getRTreeBox(target_cut_rect);
  DrcViolationSpot* spot = new DrcViolationSpot();
  int layer_id = target_cut_rect->get_layer_id();
  spot->set_layer_id(layer_id);
  spot->set_layer_name(_tech->getCutLayerNameById(layer_id));
  spot->set_net_id(target_cut_rect->get_net_id());
  spot->set_vio_type(ViolationType::kEnclosure);
  spot->setCoordinate(box.min_corner().x(), box.min_corner().y(), box.max_corner().x(), box.max_corner().y());
  _region_query->_cut_enclosure_spot_list.emplace_back(spot);
}

void EnclosureCheck::addEdgeEnclosureSpot(DrcRect* target_cut_rect)
{
  auto box = DRCUtil::getRTreeBox(target_cut_rect);
  DrcViolationSpot* spot = new DrcViolationSpot();
  int layer_id = target_cut_rect->get_layer_id();
  spot->set_layer_id(layer_id);
  spot->set_layer_name(_tech->getCutLayerNameById(layer_id));
  spot->set_net_id(target_cut_rect->get_net_id());
  spot->set_vio_type(ViolationType::kEnclosureEdge);
  spot->setCoordinate(box.min_corner().x(), box.min_corner().y(), box.max_corner().x(), box.max_corner().y());
  _region_query->_cut_enclosure_edge_spot_list.emplace_back(spot);
}

bool EnclosureCheck::checkParWithin(DrcRect* target_cut_rect, DrcEdge* drc_edge)
{
  int layer_id = drc_edge->get_layer_id();
  RTreeBox within_query_box;
  getWithinQueryBox(within_query_box, target_cut_rect, drc_edge);
  std::vector<std::pair<RTreeSegment, DrcEdge*>> query_result;
  _region_query->queryEdgeInRoutingLayer(layer_id, within_query_box, query_result);
  for (auto [rtree_seg, result_edge] : query_result) {
    if ((result_edge->isHorizontal() && drc_edge->isHorizontal()) || (result_edge->isVertical() && drc_edge->isVertical())) {
      if (result_edge->get_owner_polygon() != _cut_above_poly) {
        return true;
      }
    }
  }
  return false;
}

void EnclosureCheck::getWithinQueryBox(RTreeBox& within_query_box, DrcRect* target_cut_rect, DrcEdge* drc_edge)
{
  int par_within = _lef58_enclosure_edge_list[_edge_rule_index]->get_convex_corners().value().get_par_within();
  if (drc_edge->isVertical()) {
    bg::set<bg::min_corner, 1>(within_query_box, target_cut_rect->get_bottom());
    bg::set<bg::max_corner, 1>(within_query_box, target_cut_rect->get_top());
    if (drc_edge->get_edge_dir() == EdgeDirection::kNorth) {
      bg::set<bg::min_corner, 0>(within_query_box, drc_edge->get_begin_x());
      bg::set<bg::max_corner, 0>(within_query_box, drc_edge->get_begin_x() + par_within);
    }
    if (drc_edge->get_edge_dir() == EdgeDirection::kSouth) {
      bg::set<bg::min_corner, 0>(within_query_box, drc_edge->get_begin_x() - par_within);
      bg::set<bg::max_corner, 0>(within_query_box, drc_edge->get_begin_x());
    }
  } else {
    bg::set<bg::min_corner, 0>(within_query_box, target_cut_rect->get_left());
    bg::set<bg::max_corner, 0>(within_query_box, target_cut_rect->get_right());
    if (drc_edge->get_edge_dir() == EdgeDirection::kEast) {
      bg::set<bg::min_corner, 1>(within_query_box, drc_edge->get_begin_y() - par_within);
      bg::set<bg::max_corner, 1>(within_query_box, drc_edge->get_begin_y());
    }
    if (drc_edge->get_edge_dir() == EdgeDirection::kWest) {
      bg::set<bg::min_corner, 1>(within_query_box, drc_edge->get_begin_y());
      bg::set<bg::max_corner, 1>(within_query_box, drc_edge->get_begin_y() + par_within);
    }
  }
}

bool EnclosureCheck::checkConvexCons(DrcEdge* drc_edge, DrcRect* target_cut_rect)
{
  auto convex_corner_field = _lef58_enclosure_edge_list[_edge_rule_index]->get_convex_corners().value();
  int convex_length = convex_corner_field.get_convex_length();
  int adjacent_length = convex_corner_field.get_adjacent_length();
  int length = convex_corner_field.get_length();

  auto pre_edge = drc_edge->getPreEdge();
  auto next_edge = drc_edge->getNextEdge();

  if (drc_edge->getLength() <= convex_length && DRCUtil::isCornerConVex(drc_edge) && DRCUtil::isCornerConVex(pre_edge)) {
    if (pre_edge->getLength() <= adjacent_length && next_edge->getLength() >= length) {
      if (DRCUtil::isCornerConVex(pre_edge->getPreEdge()) && DRCUtil::isCornerConVex(pre_edge)) {
        if (checkParWithin(target_cut_rect, next_edge)) {
          return false;
        }
      }
    }
    if (next_edge->getLength() <= adjacent_length && pre_edge->getLength() >= length) {
      if (DRCUtil::isCornerConVex(drc_edge) && DRCUtil::isCornerConVex(next_edge)) {
        if (checkParWithin(target_cut_rect, pre_edge)) {
          return false;
        }
      }
    }
  }
  return true;
}

bool EnclosureCheck::checkEdgeEnclosure(DrcRect* target_cut_rect)
{
  int layer_id = target_cut_rect->get_layer_id();

  int size = _lef58_enclosure_edge_list.size();

  for (_edge_rule_index = 0; _edge_rule_index < size; _edge_rule_index++) {
    auto target_cut_class_name = getCutClassName(target_cut_rect);
    auto rule_cut_class_name = _lef58_enclosure_edge_list[_edge_rule_index]->get_class_name();
    if (target_cut_class_name.compare(rule_cut_class_name) != 0) {
      continue;
    }
    int required_overhang = _lef58_enclosure_edge_list[_edge_rule_index]->get_overhang();

    if (_lef58_enclosure_edge_list[_edge_rule_index]->get_convex_corners().has_value()) {
      // query edge
      RTreeBox query_box;
      getTriggerEdgeQueryBox(query_box, target_cut_rect, required_overhang);
      std::vector<std::pair<RTreeSegment, DrcEdge*>> query_result;
      if (_lef58_enclosure_edge_list[_edge_rule_index]->get_direction() == idb::cutlayer::Lef58EnclosureEdge::Direction::kNone) {
        _region_query->queryEdgeInRoutingLayer(layer_id, query_box, query_result);
        _region_query->queryEdgeInRoutingLayer(layer_id - 1, query_box, query_result);
      } else if (_lef58_enclosure_edge_list[_edge_rule_index]->get_direction() == idb::cutlayer::Lef58EnclosureEdge::Direction::kAbove) {
        _region_query->queryEdgeInRoutingLayer(layer_id, query_box, query_result);
      } else if (_lef58_enclosure_edge_list[_edge_rule_index]->get_direction() == idb::cutlayer::Lef58EnclosureEdge::Direction::kBelow) {
        _region_query->queryEdgeInRoutingLayer(layer_id - 1, query_box, query_result);
      }
      for (auto [rtree_seg, drc_edge] : query_result) {
        if (drc_edge->get_owner_polygon() == _cut_above_poly) {
          if (!checkConvexCons(drc_edge, target_cut_rect)) {
            return false;
          }
        }
      }

    } else {
      if (!_lef58_enclosure_edge_list[_edge_rule_index]->get_par_within().has_value()) {
        continue;
      }

      int width = _lef58_enclosure_edge_list[_edge_rule_index]->get_min_width().value();
      // rect

      if (_lef58_enclosure_edge_list[_edge_rule_index]->get_direction() == idb::cutlayer::Lef58EnclosureEdge::Direction::kNone) {
        if (_left_overhang_above <= required_overhang && _left_overhang_rect_above->getWidth() >= width) {
          RTreeBox query_box;
          getPrlQueryBox_above(query_box, EdgeDirection::kWest);
          if (!checkParallelCons(query_box, _left_overhang_rect_above)) {
            return false;
          }
        }
        if (_right_overhang_above <= required_overhang && _right_overhang_rect_above->getWidth() >= width) {
          RTreeBox query_box;
          getPrlQueryBox_above(query_box, EdgeDirection::kEast);
          if (!checkParallelCons(query_box, _left_overhang_rect_above)) {
            return false;
          }
        }

        if (_top_overhang_above <= required_overhang && _top_overhang_rect_above->getWidth() >= width) {
          RTreeBox query_box;
          getPrlQueryBox_above(query_box, EdgeDirection::kNorth);
          if (!checkParallelCons(query_box, _left_overhang_rect_above)) {
            return false;
          }
        }
        if (_bottom_overhang_above <= required_overhang && _bottom_overhang_rect_above->getWidth() >= width) {
          RTreeBox query_box;
          getPrlQueryBox_above(query_box, EdgeDirection::kSouth);
          if (!checkParallelCons(query_box, _left_overhang_rect_above)) {
            return false;
          }
        }
        if (_left_overhang_below <= required_overhang && _left_overhang_rect_below->getWidth() >= width) {
          RTreeBox query_box;
          getPrlQueryBox_below(query_box, EdgeDirection::kWest);
          if (!checkParallelCons(query_box, _left_overhang_rect_above)) {
            return false;
          }
        }

        if (_right_overhang_below <= required_overhang && _right_overhang_rect_below->getWidth() >= width) {
          RTreeBox query_box;
          getPrlQueryBox_below(query_box, EdgeDirection::kEast);
          if (!checkParallelCons(query_box, _left_overhang_rect_above)) {
            return false;
          }
        }
        if (_top_overhang_below <= required_overhang && _top_overhang_rect_below->getWidth() >= width) {
          RTreeBox query_box;
          getPrlQueryBox_below(query_box, EdgeDirection::kNorth);
          if (!checkParallelCons(query_box, _left_overhang_rect_above)) {
            return false;
          }
        }

        if (_bottom_overhang_below <= required_overhang && _bottom_overhang_rect_below->getWidth() >= width) {
          RTreeBox query_box;
          getPrlQueryBox_below(query_box, EdgeDirection::kSouth);
          if (!checkParallelCons(query_box, _left_overhang_rect_above)) {
            return false;
          }
        }

      } else if (_lef58_enclosure_edge_list[_edge_rule_index]->get_direction() == idb::cutlayer::Lef58EnclosureEdge::Direction::kAbove) {
        if (_left_overhang_above <= required_overhang && _left_overhang_rect_above->getWidth() >= width) {
          RTreeBox query_box;
          getPrlQueryBox_above(query_box, EdgeDirection::kWest);
          if (!checkParallelCons(query_box, _left_overhang_rect_above)) {
            return false;
          }
        }
        if (_right_overhang_above <= required_overhang && _right_overhang_rect_above->getWidth() >= width) {
          RTreeBox query_box;
          getPrlQueryBox_above(query_box, EdgeDirection::kEast);
          if (!checkParallelCons(query_box, _left_overhang_rect_above)) {
            return false;
          }
        }
        if (_top_overhang_above <= required_overhang && _top_overhang_rect_above->getWidth() >= width) {
          RTreeBox query_box;
          getPrlQueryBox_above(query_box, EdgeDirection::kNorth);
          if (!checkParallelCons(query_box, _left_overhang_rect_above)) {
            return false;
          }
        }
        if (_bottom_overhang_above <= required_overhang && _bottom_overhang_rect_above->getWidth() >= width) {
          RTreeBox query_box;
          getPrlQueryBox_above(query_box, EdgeDirection::kSouth);
          if (!checkParallelCons(query_box, _left_overhang_rect_above)) {
            return false;
          }
        }

      } else if (_lef58_enclosure_edge_list[_edge_rule_index]->get_direction() == idb::cutlayer::Lef58EnclosureEdge::Direction::kBelow) {
        if (_left_overhang_below <= required_overhang && _left_overhang_rect_below->getWidth() >= width) {
          RTreeBox query_box;
          getPrlQueryBox_below(query_box, EdgeDirection::kWest);
          if (!checkParallelCons(query_box, _left_overhang_rect_above)) {
            return false;
          }
        }
        if (_right_overhang_below <= required_overhang && _right_overhang_rect_below->getWidth() >= width) {
          RTreeBox query_box;
          getPrlQueryBox_below(query_box, EdgeDirection::kEast);
          if (!checkParallelCons(query_box, _left_overhang_rect_above)) {
            return false;
          }
        }
        if (_top_overhang_below <= required_overhang && _top_overhang_rect_below->getWidth() >= width) {
          RTreeBox query_box;
          getPrlQueryBox_below(query_box, EdgeDirection::kNorth);
          if (!checkParallelCons(query_box, _left_overhang_rect_above)) {
            return false;
          }
        }
        if (_bottom_overhang_below <= required_overhang && _bottom_overhang_rect_below->getWidth() >= width) {
          RTreeBox query_box;
          getPrlQueryBox_below(query_box, EdgeDirection::kSouth);
          if (!checkParallelCons(query_box, _left_overhang_rect_above)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

void EnclosureCheck::getPrlQueryBox_below(RTreeBox& query_box, EdgeDirection edge_dir)
{
  int par_within = _lef58_enclosure_edge_list[_edge_rule_index]->get_par_within().value();

  if (edge_dir == EdgeDirection::kEast) {
    bg::set<bg::min_corner, 0>(query_box, _right_overhang_rect_below->get_right());
    bg::set<bg::min_corner, 1>(query_box, _right_overhang_rect_below->get_bottom());
    bg::set<bg::max_corner, 0>(query_box, _right_overhang_rect_below->get_right() + par_within);
    bg::set<bg::max_corner, 1>(query_box, _right_overhang_rect_below->get_top());
  }
  if (edge_dir == EdgeDirection::kWest) {
    bg::set<bg::min_corner, 0>(query_box, _left_overhang_rect_below->get_left());
    bg::set<bg::min_corner, 1>(query_box, _left_overhang_rect_below->get_bottom());
    bg::set<bg::max_corner, 0>(query_box, _left_overhang_rect_below->get_left() - par_within);
    bg::set<bg::max_corner, 1>(query_box, _left_overhang_rect_below->get_top());
  }
  if (edge_dir == EdgeDirection::kNorth) {
    bg::set<bg::min_corner, 0>(query_box, _top_overhang_rect_below->get_left());
    bg::set<bg::min_corner, 1>(query_box, _top_overhang_rect_below->get_top());
    bg::set<bg::max_corner, 0>(query_box, _top_overhang_rect_below->get_right());
    bg::set<bg::max_corner, 1>(query_box, _top_overhang_rect_below->get_top() + par_within);
  }
  if (edge_dir == EdgeDirection::kSouth) {
    bg::set<bg::min_corner, 0>(query_box, _bottom_overhang_rect_below->get_left());
    bg::set<bg::min_corner, 1>(query_box, _bottom_overhang_rect_below->get_bottom());
    bg::set<bg::max_corner, 0>(query_box, _bottom_overhang_rect_below->get_right());
    bg::set<bg::max_corner, 1>(query_box, _bottom_overhang_rect_below->get_bottom() - par_within);
  }
}

void EnclosureCheck::getPrlQueryBox_above(RTreeBox& query_box, EdgeDirection edge_dir)
{
  int par_within = _lef58_enclosure_edge_list[_edge_rule_index]->get_par_within().value();

  if (edge_dir == EdgeDirection::kEast) {
    bg::set<bg::min_corner, 0>(query_box, _right_overhang_rect_above->get_right());
    bg::set<bg::min_corner, 1>(query_box, _right_overhang_rect_above->get_bottom());
    bg::set<bg::max_corner, 0>(query_box, _right_overhang_rect_above->get_right() + par_within);
    bg::set<bg::max_corner, 1>(query_box, _right_overhang_rect_above->get_top());
  }
  if (edge_dir == EdgeDirection::kWest) {
    bg::set<bg::min_corner, 0>(query_box, _left_overhang_rect_above->get_left());
    bg::set<bg::min_corner, 1>(query_box, _left_overhang_rect_above->get_bottom());
    bg::set<bg::max_corner, 0>(query_box, _left_overhang_rect_above->get_left() - par_within);
    bg::set<bg::max_corner, 1>(query_box, _left_overhang_rect_above->get_top());
  }
  if (edge_dir == EdgeDirection::kNorth) {
    bg::set<bg::min_corner, 0>(query_box, _top_overhang_rect_above->get_left());
    bg::set<bg::min_corner, 1>(query_box, _top_overhang_rect_above->get_top());
    bg::set<bg::max_corner, 0>(query_box, _top_overhang_rect_above->get_right());
    bg::set<bg::max_corner, 1>(query_box, _top_overhang_rect_above->get_top() + par_within);
  }
  if (edge_dir == EdgeDirection::kSouth) {
    bg::set<bg::min_corner, 0>(query_box, _bottom_overhang_rect_above->get_left());
    bg::set<bg::min_corner, 1>(query_box, _bottom_overhang_rect_above->get_bottom());
    bg::set<bg::max_corner, 0>(query_box, _bottom_overhang_rect_above->get_right());
    bg::set<bg::max_corner, 1>(query_box, _bottom_overhang_rect_above->get_bottom() - par_within);
  }
}

bool EnclosureCheck::checkParallelCons(RTreeBox& query_box, DrcRect* target_enclosure_rect)
{
  if (!_lef58_enclosure_edge_list[_edge_rule_index]->get_par_length().has_value()) {
    return true;
  }
  int par_length = _lef58_enclosure_edge_list[_edge_rule_index]->get_par_length().value();
  int layer_id = target_enclosure_rect->get_layer_id();
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  _region_query->queryInRoutingLayer(layer_id, query_box, query_result);
  for (auto [rtree_box, result_rect] : query_result) {
    if (DRCUtil::intersection(result_rect, target_enclosure_rect)) {
      continue;
    }
    int prl = DRCUtil::getPRLRunLength(result_rect, target_enclosure_rect);
    if (prl >= par_length) {
      // std::cout << "edge enclosure vio!!!" << std::endl;
      _check_result = false;
      return false;
    }
  }
  return true;
}

void EnclosureCheck::getTriggerEdgeQueryBox(RTreeBox& query_box, DrcRect* target_cut_rect, int required_overhang)
{
  bg::set<bg::min_corner, 0>(query_box, target_cut_rect->get_left() - required_overhang);
  bg::set<bg::min_corner, 1>(query_box, target_cut_rect->get_bottom() - required_overhang);
  bg::set<bg::max_corner, 0>(query_box, target_cut_rect->get_right() + required_overhang);
  bg::set<bg::max_corner, 1>(query_box, target_cut_rect->get_top() + required_overhang);
}

/**
 * @brief Check whether the Enclosure violation exists in the area query result
 *
 * @param layer_id
 * @param target_rect
 * @param enclosure_rule
 * @param query_result
 * @return true ：If the query result is not empty, the shape covering the target
 *  rectangle exists in the corresponding metal layer and there is no violation.
 * @return false : If the query result is empty, it indicates that
 * the shape of the target rectangle is not covered by the corresponding metal layer,
 * which indicates that there is a violation.
 */
bool EnclosureCheck::checkEnclosureFromQueryResult(int layer_id, DrcRect* target_rect, EnclosureRule* enclosure_rule,
                                                   std::vector<std::pair<RTreeBox, DrcRect*>> query_result)
{
  if (!query_result.empty()) {
    return true;
  } else {
    return false;
  }
}

/**
 * @brief Obtain two target rectangles for region query:
 * One is the horizontal direction is overhang1 direction, the vertical direction is overhang2 direction;
 * The other is the horizontal direction is the overhang2 direction, the vertical direction is the overhang1 direction;
 *
 * @param target_rect
 * @param overhang1
 * @param overhang2
 * @param query_box_list
 */
void EnclosureCheck::getEnclosureQueryBox(DrcRect* target_rect, int overhang1, int overhang2, std::vector<RTreeBox>& query_box_list)
{
  int lb_x = target_rect->get_left() - overhang1;
  int lb_y = target_rect->get_bottom() - overhang2;
  int rt_x = target_rect->get_right() + overhang1;
  int rt_y = target_rect->get_top() + overhang2;
  query_box_list.push_back(RTreeBox(RTreePoint(lb_x, lb_y), RTreePoint(rt_x, rt_y)));
  lb_x = target_rect->get_left() - overhang2;
  lb_y = target_rect->get_bottom() - overhang1;
  rt_x = target_rect->get_right() + overhang2;
  rt_y = target_rect->get_top() + overhang1;
  query_box_list.push_back(RTreeBox(RTreePoint(lb_x, lb_y), RTreePoint(rt_x, rt_y)));
}

/**
 * @brief Performing a regional query
 *
 * @param CutLayerId
 * @param query_box
 * @param enclosure_rule
 * @param get_above_result Flag Queries the upper metal layer
 * @param get_below_result Flag Queries the metal layer below
 * @return std::vector<std::pair<RTreeBox, DrcRect*>>
 */
std::vector<std::pair<RTreeBox, DrcRect*>> EnclosureCheck::getQueryResult(int CutLayerId, RTreeBox& query_box,
                                                                          EnclosureRule* enclosure_rule, bool get_above_result,
                                                                          bool get_below_result)
{
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  if (get_below_result) {
    _region_query->queryEnclosureInRoutingLayer(CutLayerId - 1, query_box, query_result);
  }
  if (get_above_result) {
    _region_query->queryEnclosureInRoutingLayer(CutLayerId, query_box, query_result);
  }
  return query_result;
}

/**
 * @brief Save the violation result
 *
 * @param cutLayerId
 * @param target_rect
 */
void EnclosureCheck::storeViolationResult(int cutLayerId, DrcRect* target_rect)
{
  if (_interact_with_irt) {
    // _violation_rect_pair_list.push_back(std::make_pair(target_rect, result_rect));
  } else {
    _layer_to_violation_box_tree[cutLayerId].insert(DRCUtil::getRTreeBox(target_rect));
  }
}

/**
 * @brief Reset to clear all violation markers
 *
 */
void EnclosureCheck::reset()
{
  for (auto& [LayerId, enclosure_spot_list] : _cut_layer_to_enclosure_spots_list) {
    for (auto& enclosure_spot : enclosure_spot_list) {
      enclosure_spot.clearSpotRects();
    }
  }
  _cut_layer_to_enclosure_spots_list.clear();
  _layer_to_violation_box_tree.clear();
  _violation_rect_list.clear();
}

/**
 * @brief Gets the number of enclosure violations
 *
 * @return int
 */
int EnclosureCheck::get_enclosure_violation_num()
{
  int count = 0;
  for (auto& [layerId, spot_list] : _cut_layer_to_enclosure_spots_list) {
    count += spot_list.size();
  }
  return count;
}

}  // namespace idrc