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
#include "CutSpacingCheck.hpp"

#include "DRCUtil.h"
#include "DrcConflictGraph.h"
#include "DrcDesign.h"
#include "RegionQuery.h"
#include "Tech.h"

namespace idrc {

bool CutSpacingCheck::check(DrcRect* target_rect)
{
  _check_result = true;
  checkCutSpacing(target_rect);
  return _check_result;
}

/**
 * @brief Initializes the violation tag list from the R-tree
 */
void CutSpacingCheck::initSpacingSpotListFromRtree()
{
  for (auto& [layerId, rtree] : _layer_to_violation_box_tree) {
    for (auto it = rtree.begin(); it != rtree.end(); ++it) {
      RTreeBox box = *it;
      DrcRectangle<int> rect = DRCUtil::getRectangleFromRTreeBox(box);
      DrcSpot spot;
      spot.set_violation_type(ViolationType::kCutSpacing);
      DrcRect* drc_rect = new DrcRect();
      drc_rect->set_owner_type(RectOwnerType::kSpotMark);
      drc_rect->set_layer_id(layerId);
      drc_rect->set_rectangle(rect);
      spot.add_spot_rect(drc_rect);
      _cut_layer_to_spacing_spots_list[layerId].emplace_back(spot);
    }
  }
}

/**
 * @brief Initialize the CutSpacing check module
 *
 * @param config
 * @param tech
 */
void CutSpacingCheck::init(DrcConfig* config, Tech* tech, RegionQuery* region_query)
{
  _config = config;
  _tech = tech;
  _region_query = region_query;
}

/**
 * @brief Check the target line network for Cut Spacing
 *
 * @param target_net
 */
void CutSpacingCheck::checkCutSpacing(DrcNet* target_net)
{
  for (auto& [layerId, routing_rect_list] : target_net->get_layer_to_cut_rects_map()) {
    for (auto target_rect : routing_rect_list) {
      checkCutSpacing(target_rect);
    }
  }
  // initSpacingSpotListFromRtree();
  // // Clear each net, or the next net will repeat the record
  // _layer_to_violation_box_tree.clear();
}

void CutSpacingCheck::getSpacing1QueryBox_PrlNeg(RTreeBox& query_box, QueryBoxDir dir, DrcRect* drc_rect)
{
  // cut spacing2 is a single value in t28, which simplifies the build query_box operation. If encounter a spacing table with no single
  // value in cut spacing2, need to change the following code.
  int spacing1 = _lef58_spacing_table_list[_rule_index]->get_cutclass().get_cut_spacing(1, 1).get_cut_spacing1().value();
  int prl = _lef58_spacing_table_list[_rule_index]->get_prl()->get_prl();
  if (dir == QueryBoxDir::kNE) {
    bg::set<bg::min_corner, 0>(query_box, drc_rect->get_right() - prl);
    bg::set<bg::min_corner, 1>(query_box, drc_rect->get_top() - prl);
    bg::set<bg::max_corner, 0>(query_box, drc_rect->get_right() + spacing1);
    bg::set<bg::max_corner, 1>(query_box, drc_rect->get_top() + spacing1);
  } else if (dir == QueryBoxDir::kNW) {
    bg::set<bg::min_corner, 0>(query_box, drc_rect->get_left() - spacing1);
    bg::set<bg::min_corner, 1>(query_box, drc_rect->get_top() - prl);
    bg::set<bg::max_corner, 0>(query_box, drc_rect->get_left() + prl);
    bg::set<bg::max_corner, 1>(query_box, drc_rect->get_top() + spacing1);
  } else if (dir == QueryBoxDir::kSE) {
    bg::set<bg::min_corner, 0>(query_box, drc_rect->get_right() - prl);
    bg::set<bg::min_corner, 1>(query_box, drc_rect->get_bottom() - spacing1);
    bg::set<bg::max_corner, 0>(query_box, drc_rect->get_right() + spacing1);
    bg::set<bg::max_corner, 1>(query_box, drc_rect->get_bottom() + prl);
  } else if (dir == QueryBoxDir::kSW) {
    bg::set<bg::min_corner, 0>(query_box, drc_rect->get_left() - spacing1);
    bg::set<bg::min_corner, 1>(query_box, drc_rect->get_bottom() - spacing1);
    bg::set<bg::max_corner, 0>(query_box, drc_rect->get_left() + prl);
    bg::set<bg::max_corner, 1>(query_box, drc_rect->get_bottom() + prl);
  }
}

void CutSpacingCheck::getSpacing2QueryBox_PrlNeg(RTreeBox& query_box, QueryBoxDir dir, DrcRect* drc_rect)
{
  // cut spacing2 is a single value in t28, which simplifies the build query_box operation. If encounter a spacing table with no single
  // value in cut spacing2, need to change the following code.
  int spacing2 = _lef58_spacing_table_list[_rule_index]->get_cutclass().get_cut_spacing(0, 0).get_cut_spacing2().value();
  int prl = _lef58_spacing_table_list[_rule_index]->get_prl()->get_prl();
  if (dir == QueryBoxDir::kUp) {
    bg::set<bg::min_corner, 0>(query_box, drc_rect->get_left() + prl);
    bg::set<bg::min_corner, 1>(query_box, drc_rect->get_top());
    bg::set<bg::max_corner, 0>(query_box, drc_rect->get_right() - prl);
    bg::set<bg::max_corner, 1>(query_box, drc_rect->get_top() + spacing2);
  } else if (dir == QueryBoxDir::kDown) {
    bg::set<bg::min_corner, 0>(query_box, drc_rect->get_left() + prl);
    bg::set<bg::min_corner, 1>(query_box, drc_rect->get_bottom() - spacing2);
    bg::set<bg::max_corner, 0>(query_box, drc_rect->get_right() - prl);
    bg::set<bg::max_corner, 1>(query_box, drc_rect->get_bottom());
  } else if (dir == QueryBoxDir::kLeft) {
    bg::set<bg::min_corner, 0>(query_box, drc_rect->get_left() - spacing2);
    bg::set<bg::min_corner, 1>(query_box, drc_rect->get_bottom() + prl);
    bg::set<bg::max_corner, 0>(query_box, drc_rect->get_left());
    bg::set<bg::max_corner, 1>(query_box, drc_rect->get_top() - prl);
  } else if (dir == QueryBoxDir::kRight) {
    bg::set<bg::min_corner, 0>(query_box, drc_rect->get_right());
    bg::set<bg::min_corner, 1>(query_box, drc_rect->get_bottom() + prl);
    bg::set<bg::max_corner, 0>(query_box, drc_rect->get_right() + spacing2);
    bg::set<bg::max_corner, 1>(query_box, drc_rect->get_top() - prl);
  }
}

void CutSpacingCheck::getSpacing1QueryBoxList_PrlNeg(std::vector<RTreeBox>& query_box_list, DrcRect* drc_rect)
{
  getSpacing1QueryBox_PrlNeg(query_box_list[0], QueryBoxDir::kNE, drc_rect);
  getSpacing1QueryBox_PrlNeg(query_box_list[1], QueryBoxDir::kNW, drc_rect);
  getSpacing1QueryBox_PrlNeg(query_box_list[2], QueryBoxDir::kSE, drc_rect);
  getSpacing1QueryBox_PrlNeg(query_box_list[3], QueryBoxDir::kSW, drc_rect);
}

void CutSpacingCheck::getSpacing2QueryBoxList_PrlNeg(std::vector<RTreeBox>& query_box_list, DrcRect* drc_rect)
{
  getSpacing2QueryBox_PrlNeg(query_box_list[0], QueryBoxDir::kUp, drc_rect);
  getSpacing2QueryBox_PrlNeg(query_box_list[1], QueryBoxDir::kDown, drc_rect);
  getSpacing2QueryBox_PrlNeg(query_box_list[2], QueryBoxDir::kLeft, drc_rect);
  getSpacing2QueryBox_PrlNeg(query_box_list[3], QueryBoxDir::kRight, drc_rect);
}

void CutSpacingCheck::checkSpacing_TwoRect_SingleValue(DrcRect* target_rect, DrcRect* result_rect)
{
  if (DRCUtil::intersection(target_rect, result_rect)) {
    _check_result = false;
    // std::cout << "short vio" << std::endl;
    _region_query->addViolation(ViolationType::kCutShort);
  }
  int layer_id = target_rect->get_layer_id();
  int required_spacing = _tech->get_drc_cut_layer_list()[layer_id]->get_cut_spacing();
  if (DRCUtil::isParallelOverlap(target_rect, result_rect)) {
    checkMaxXYSpacing(target_rect, result_rect, required_spacing);
  } else {
    checkCornerSpacing(target_rect, result_rect, required_spacing);
  }
}

void CutSpacingCheck::checkQueryResult_SingleValue(std::vector<std::pair<RTreeBox, DrcRect*>>& query_result, DrcRect* target_rect,
                                                   RTreeBox query_box)
{
  DrcRect* query_rect = new DrcRect();
  query_rect->set_coordinate(query_box.min_corner().x(), query_box.min_corner().y(), query_box.max_corner().x(),
                             query_box.max_corner().y());
  for (auto& [rt_box, result_rect] : query_result) {
    if (intersectionExceptJustEdgeTouch(query_rect, result_rect)) {
      if (target_rect == result_rect) {
        continue;
      }
      checkSpacing_TwoRect_SingleValue(target_rect, result_rect);
    }
  }
}

void CutSpacingCheck::checkCutSpacing_SingleValue(DrcRect* target_rect)
{
  int layer_id = target_rect->get_layer_id();
  int required_spacing = _tech->get_drc_cut_layer_list()[layer_id]->get_cut_spacing();
  RTreeBox query_box;
  getQueryBox(query_box, target_rect, required_spacing);
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  _region_query->queryInCutLayer(layer_id, query_box, query_result);
  checkQueryResult_SingleValue(query_result, target_rect, query_box);
}

/**
 * @brief Check the target rectangle for Cut Spacing
 *
 * @param target_rect
 */
void CutSpacingCheck::checkCutSpacing(DrcRect* target_rect)
{
  int layer_id = target_rect->get_layer_id();
  _lef58_cut_class_list = _tech->get_drc_cut_layer_list()[layer_id]->get_lef58_cut_class_list();
  _lef58_spacing_table_list = _tech->get_drc_cut_layer_list()[layer_id]->get_lef58_spacing_table_list();
  // int layer_max_require_spacing = _tech->getCutSpacing(cutLayerId);
  // RTreeBox query_box = getSpacingQueryBox(target_rect, layer_max_require_spacing);
  // std::vector<std::pair<RTreeBox, DrcRect*>> query_result = getQueryResult(cutLayerId, query_box);
  // checkSpacingFromQueryResult(cutLayerId, target_rect, query_result);
  if (_lef58_spacing_table_list.empty()) {
    checkCutSpacing_SingleValue(target_rect);
  } else {
    int size = _lef58_spacing_table_list.size();
    for (_rule_index = 0; _rule_index < size; _rule_index++) {
      if (_lef58_spacing_table_list[_rule_index]->get_prl()->get_prl() < 0) {
        if (_lef58_spacing_table_list[_rule_index]->get_prl()->is_maxxy()) {
          // SPACINGTABLE PRL -0.04 MAXXY
          // CUTCLASS	VSINGLECUT	VDOUBLECUT
          // VSINGLECUT	0.070 0.080	0.075 0.080
          // VDOUBLECUT	0.075 0.080	0.080 0.080 ;" ;
          checkSpacing2_PrlNeg(target_rect);
          checkSpacing1_PrlNeg(target_rect);
        } else {
          std::cout << "[DRC::CutSpacingCheck] Sorry, Euclidean spacing check with positive prl is not supported, check skipped"
                    << std::endl;
        }
      } else {
        if (_lef58_spacing_table_list[_rule_index]->get_prl()->get_prl() > 0) {
          if (_lef58_spacing_table_list[_rule_index]->get_second_layer().has_value()) {
            // PROPERTY LEF58_SPACINGTABLE "
            // SPACINGTABLE LAYER VIA5 PRL 0.02
            // CUTCLASS    VSINGLECUT  VDOUBLECUT
            // VSINGLECUT  0.000 0.060 0.000 0.060
            // VDOUBLECUT  0.000 0.060 0.000 0.060 ;" ;
            checkSpacing_PrlPos(target_rect);
          }
        } else {
          std::cout << "[DRC::CutSpacingCheck] Sorry, SpanLayer spacing check with negative prl is not supported, check skipped"
                    << std::endl;
        }
      }
    }
  }
}

int CutSpacingCheck::getQuerySpacing_PrlPos(int cut_class_index, DrcRect* target_rect)
{
  // 该Cut类型的最大间隔
  int max_index = _lef58_spacing_table_list[_rule_index]->get_cutclass().get_class_name1_list().size() - 1;
  return _lef58_spacing_table_list[_rule_index]->get_cutclass().get_cut_spacing(cut_class_index, max_index).get_cut_spacing2().value();
}

void CutSpacingCheck::getQueryBox(RTreeBox& query_box, DrcRect* drc_rect, int cut_class_query_spacing)
{
  bg::set<bg::min_corner, 0>(query_box, drc_rect->get_left() - cut_class_query_spacing);
  bg::set<bg::min_corner, 1>(query_box, drc_rect->get_bottom() - cut_class_query_spacing);
  bg::set<bg::max_corner, 0>(query_box, drc_rect->get_right() + cut_class_query_spacing);
  bg::set<bg::max_corner, 1>(query_box, drc_rect->get_top() + cut_class_query_spacing);
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

int CutSpacingCheck::getQueryLayerId_PrlPos()
{
  auto layer_name = _lef58_spacing_table_list[_rule_index]->get_second_layer()->get_second_layer_name();
  int layer_id = _tech->getLayerIdByLayerName(layer_name);
  return layer_id;
}

bool CutSpacingCheck::skipCheck(DrcRect* target_rect, DrcRect* result_rect)
{
  if (_lef58_spacing_table_list[_rule_index]->get_second_layer().has_value()) {
    if (target_rect->get_net_id() == result_rect->get_net_id()) {
      return true;
    }
  }
  return target_rect == result_rect;
}

int CutSpacingCheck::getPrlOfTwoRect(DrcRect* target_rect, DrcRect* result_rect)
{
  return DRCUtil::getPRLRunLength(target_rect, result_rect);
}

int CutSpacingCheck::getRequiredSpacingOfTwoRect(DrcRect* target_rect, DrcRect* result_rect)
{
  int prl = getPrlOfTwoRect(target_rect, result_rect);
  int required_prl = _lef58_spacing_table_list[_rule_index]->get_prl()->get_prl();
  int target_rect_class_index = getCutClassIndex(target_rect);
  int result_rect_class_index = getCutClassIndex(result_rect);
  int required_spacing;
  if (prl > required_prl) {
    required_spacing = _lef58_spacing_table_list[_rule_index]
                           ->get_cutclass()
                           .get_cut_spacing(target_rect_class_index, result_rect_class_index)
                           .get_cut_spacing2()
                           .value();
  } else {
    required_spacing = _lef58_spacing_table_list[_rule_index]
                           ->get_cutclass()
                           .get_cut_spacing(target_rect_class_index, result_rect_class_index)
                           .get_cut_spacing1()
                           .value();
  }
  return required_spacing;
}

void CutSpacingCheck::checkMaxXYSpacing(DrcRect* target_rect, DrcRect* result_rect, int required_spacing)
{
  RTreeBox span_box = DRCUtil::getSpanBoxBetweenTwoRects(target_rect, result_rect);
  // bool isHorizontalParallelOverlap = false;
  int spacing = -1;
  int lb_x = span_box.min_corner().get<0>();
  int lb_y = span_box.min_corner().get<1>();
  int rt_x = span_box.max_corner().get<0>();
  int rt_y = span_box.max_corner().get<1>();
  if (DRCUtil::isHorizontalParallelOverlap(target_rect, result_rect)) {
    // isHorizontalParallelOverlap = true;
    spacing = std::abs(rt_y - lb_y);
  } else {
    spacing = std::abs(rt_x - lb_x);
  }
  if (spacing < required_spacing) {
    _check_result = false;
    // std::cout << "spacing2 vio" << std::endl;
    if (target_rect->get_layer_id() == result_rect->get_layer_id()) {
      if (_region_query->addCutSpacingViolation(target_rect, result_rect)) {
        addCutSpacingSpot(target_rect, result_rect);
      }

    } else {
      if (_region_query->addCutDiffLayerSpacingViolation(target_rect, result_rect)) {
        addDiffLayerSpot(target_rect, result_rect);
      }
    }
    // _region_query->addViolation(ViolationType::kCutSpacing);
  }
}

void CutSpacingCheck::addCutSpacingSpot(DrcRect* target_rect, DrcRect* result_rect)
{
  auto box = DRCUtil::getSpanBoxBetweenTwoRects(target_rect, result_rect);
  DrcViolationSpot* spot = new DrcViolationSpot();
  int layer_id = target_rect->get_layer_id();
  spot->set_layer_id(layer_id);
  spot->set_layer_name(_tech->getCutLayerNameById(layer_id));
  spot->set_net_id(target_rect->get_net_id());
  spot->set_vio_type(ViolationType::kCutSpacing);
  spot->setCoordinate(box.min_corner().x(), box.min_corner().y(), box.max_corner().x(), box.max_corner().y());
  _region_query->_cut_spacing_spot_list.emplace_back(spot);
}

void CutSpacingCheck::addDiffLayerSpot(DrcRect* target_rect, DrcRect* result_rect)
{
  auto box = DRCUtil::getSpanBoxBetweenTwoRects(target_rect, result_rect);
  DrcViolationSpot* spot = new DrcViolationSpot();
  int layer_id = target_rect->get_layer_id();
  spot->set_layer_id(layer_id);
  spot->set_layer_name(_tech->getCutLayerNameById(layer_id));
  spot->set_net_id(target_rect->get_net_id());
  spot->set_vio_type(ViolationType::kCutDiffLayerSpacing);
  spot->setCoordinate(box.min_corner().x(), box.min_corner().y(), box.max_corner().x(), box.max_corner().y());
  _region_query->_cut_diff_layer_spacing_spot_list.emplace_back(spot);
}

void CutSpacingCheck::checkSpacing_TwoRect_PrlPos(DrcRect* target_rect, DrcRect* result_rect)
{
  int required_spacing = getRequiredSpacingOfTwoRect(target_rect, result_rect);
  if (DRCUtil::intersection(target_rect, result_rect)) {
    _check_result = false;
    // std::cout << "short vio" << std::endl;
    _region_query->addViolation(ViolationType::kCutShort);
  }
  if (DRCUtil::isParallelOverlap(target_rect, result_rect)) {
    checkMaxXYSpacing(target_rect, result_rect, required_spacing);
  } else {
    checkCornerSpacing(target_rect, result_rect, required_spacing);
  }
}

void CutSpacingCheck::checkQueryResult_PrlPos(std::vector<std::pair<RTreeBox, DrcRect*>>& query_result, DrcRect* target_rect,
                                              RTreeBox& query_box)
{
  DrcRect* query_rect = new DrcRect();
  query_rect->set_coordinate(query_box.min_corner().x(), query_box.min_corner().y(), query_box.max_corner().x(),
                             query_box.max_corner().y());
  for (auto& [rt_box, result_rect] : query_result) {
    if (intersectionExceptJustEdgeTouch(query_rect, result_rect)) {
      if (skipCheck(target_rect, result_rect)) {
        continue;
      }
      checkSpacing_TwoRect_PrlPos(target_rect, result_rect);
    }
  }
}

void CutSpacingCheck::checkSpacing_PrlPos(DrcRect* target_rect)
{
  int cut_class_index = getCutClassIndex(target_rect);
  int cut_class_query_spacing = getQuerySpacing_PrlPos(cut_class_index, target_rect);
  RTreeBox query_box;
  getQueryBox(query_box, target_rect, cut_class_query_spacing);
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  int query_layer_id = getQueryLayerId_PrlPos();
  _region_query->queryInCutLayer(query_layer_id, query_box, query_result);
  checkQueryResult_PrlPos(query_result, target_rect, query_box);
}

int CutSpacingCheck::getCutClassIndex(DrcRect* cut_rect)
{
  int size = _lef58_cut_class_list.size();
  for (int index = 0; index < size; index++) {
    if (cut_rect->getWidth() == _lef58_cut_class_list[index]->get_via_width()
        && cut_rect->getLength() == _lef58_cut_class_list[index]->get_via_length()) {
      return index;
    }
  }
  std::cout << "[DRC::CutSpacingCheck] Warning: Unkown Cut Class!" << std::endl;
  return -1;
}

int CutSpacingCheck::getRequiredSpacing1(DrcRect* result_rect, DrcRect* target_rect)
{
  // int layer_id = result_rect->get_layer_id();
  int index1 = getCutClassIndex(result_rect);
  int index2 = getCutClassIndex(target_rect);
  if (_lef58_spacing_table_list[_rule_index]->get_cutclass().get_cut_spacing(index1, index2).get_cut_spacing1().has_value()) {
    return _lef58_spacing_table_list[_rule_index]->get_cutclass().get_cut_spacing(index1, index2).get_cut_spacing1().value();
  } else {
    std::cout << "[DRC Cut Spacing Check Warning] : get tech data failed!" << std::endl;
  }
  return -1;
}

void CutSpacingCheck::checkCornerSpacing(DrcRect* result_rect, DrcRect* target_rect, int required_spacing)
{
  int distanceX = std::min(std::abs(target_rect->get_left() - result_rect->get_right()),
                           std::abs(target_rect->get_right() - result_rect->get_left()));
  int distanceY = std::min(std::abs(target_rect->get_bottom() - result_rect->get_top()),
                           std::abs(target_rect->get_top() - result_rect->get_bottom()));

  if (required_spacing * required_spacing > distanceX * distanceX + distanceY * distanceY) {
    // add_spot();
    _check_result = false;
    // std::cout << "spacing1 vio" << std::endl;
    // _region_query->addViolation(ViolationType::kCutSpacing);
    if (target_rect->get_layer_id() == result_rect->get_layer_id()) {
      if (_region_query->addCutSpacingViolation(target_rect, result_rect)) {
        addCutSpacingSpot(target_rect, result_rect);
      }
    } else {
      if (_region_query->addCutDiffLayerSpacingViolation(target_rect, result_rect)) {
        addDiffLayerSpot(target_rect, result_rect);
      }
    }
  }
}

void CutSpacingCheck::checkSpacing1_TwoRect_PrlNeg(DrcRect* result_rect, DrcRect* target_rect)
{
  int required_spacing = getRequiredSpacing1(result_rect, target_rect);
  checkCornerSpacing(result_rect, target_rect, required_spacing);
}

void CutSpacingCheck::checkSpacing1_PrlNeg(DrcRect* target_rect)
{
  std::vector<RTreeBox> query_box_list(4);
  getSpacing1QueryBoxList_PrlNeg(query_box_list, target_rect);
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  for (auto& query_box : query_box_list) {
    _region_query->queryInCutLayer(target_rect->get_layer_id(), query_box, query_result);
    for (auto& [boost_rect, cut_rect] : query_result) {
      DrcRect query_rect;
      query_rect.set_coordinate(query_box.min_corner().x(), query_box.min_corner().y(), query_box.max_corner().x(),
                                query_box.max_corner().y());
      if (intersectionExceptJustEdgeTouch(&query_rect, cut_rect)) {
        // std::cout << "check PRL NEG" << std::endl;
        checkSpacing1_TwoRect_PrlNeg(cut_rect, target_rect);
      }
    }
    query_result.clear();
  }
}

void CutSpacingCheck::checkSpacing2_PrlNeg(DrcRect* target_rect)
{
  std::vector<RTreeBox> query_box_list(4);
  getSpacing2QueryBoxList_PrlNeg(query_box_list, target_rect);
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  for (auto& query_box : query_box_list) {
    query_result.clear();
    _region_query->queryInCutLayer(target_rect->get_layer_id(), query_box, query_result);
    for (auto& [boost_rect, cut_rect] : query_result) {
      DrcRect query_rect;
      query_rect.set_coordinate(query_box.min_corner().x(), query_box.min_corner().y(), query_box.max_corner().x(),
                                query_box.max_corner().y());
      if (intersectionExceptJustEdgeTouch(&query_rect, cut_rect)) {
        // TODO
        // addSpot();
        _check_result = false;
        // std::cout << "spacing2 vio" << std::endl;
        // _region_query->addViolation(ViolationType::kCutSpacing);
        if (_region_query->addCutSpacingViolation(target_rect, cut_rect)) {
          addCutSpacingSpot(target_rect, cut_rect);
        }
      }
    }
  }
}

/**
 * @brief Gets the query rectangle for the interval check
 *
 * @param target_rect
 * @param spacing
 * @return RTreeBox
 */
RTreeBox CutSpacingCheck::getSpacingQueryBox(DrcRect* target_rect, int spacing)
{
  int lb_x = target_rect->get_left() - spacing;
  int lb_y = target_rect->get_bottom() - spacing;
  int rt_x = target_rect->get_right() + spacing;
  int rt_y = target_rect->get_top() + spacing;
  return RTreeBox(RTreePoint(lb_x, lb_y), RTreePoint(rt_x, rt_y));
}

/**
 * @brief Performing a regional query
 *
 * @param CutLayerId
 * @param query_box
 * @return std::vector<std::pair<RTreeBox, DrcRect*>>
 */
std::vector<std::pair<RTreeBox, DrcRect*>> CutSpacingCheck::getQueryResult(int CutLayerId, RTreeBox& query_box)
{
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  _region_query->queryInCutLayer(CutLayerId, query_box, query_result);
  return query_result;
}

/**
 * @brief Check for a spacing violation from the query results and save
 *
 * @param cutLayerId
 * @param target_rect
 * @param query_result
 */
void CutSpacingCheck::checkSpacingFromQueryResult(int cutLayerId, DrcRect* target_rect,
                                                  std::vector<std::pair<RTreeBox, DrcRect*>>& query_result)
{
  for (auto rect_pair : query_result) {
    DrcRect* result_rect = rect_pair.second;
    // Skip some cases that don't need to be checked
    if (skipCheck(target_rect, result_rect)) {
      continue;
    }
    // Check for spacing violations
    if (checkSpacingViolation(cutLayerId, target_rect, result_rect, query_result)) {
      storeViolationResult(cutLayerId, target_rect, result_rect, ViolationType::kCutSpacing);
    }
  }
}

/**
 * @brief Determine whether to skip the check
 *
 * @param target_rect
 * @param result_rect
 * @return true
 * @return false
 */
// bool CutSpacingCheck::skipCheck(DrcRect* target_rect, DrcRect* result_rect)
// {
//   return (target_rect == result_rect);
// }

/**
 * @brief Check the query result for violations
 *
 * @param cutLayerId
 * @param target_rect
 * @param result_rect
 * @param query_result
 * @return true
 * @return false
 */
bool CutSpacingCheck::checkSpacingViolation(int cutLayerId, DrcRect* target_rect, DrcRect* result_rect,
                                            std::vector<std::pair<RTreeBox, DrcRect*>>& query_result)
{
  int require_spacing = _tech->getCutSpacing(cutLayerId);
  DrcRect query_rect = DRCUtil::enlargeRect(target_rect, require_spacing);
  // The rectangle that just intersects the edge of the search box is not violation
  if (intersectionExceptJustEdgeTouch(&query_rect, result_rect)) {
    if (!isParallelOverlap(target_rect, result_rect)) {
      // case no Parallel Overlap between two rect ,need check corner spacing
      // if corner spacing is not meet require_spacing,it is a violation
      return checkCornerSpacingViolation(target_rect, result_rect, require_spacing);
    } else {
      // There is  Parallel Overlap between two rect
      // need check span box is covered by exited rect
      return checkXYSpacingViolation(target_rect, result_rect, require_spacing, query_result);
    }
  }
  return false;
}

/**
 * @brief Determine if only the boundaries intersect
 *
 * @param query_rect
 * @param result_rect
 * @return true
 * @return false
 */
bool CutSpacingCheck::intersectionExceptJustEdgeTouch(DrcRect* query_rect, DrcRect* result_rect)
{
  return DRCUtil::intersection(query_rect, result_rect, false);
}

/**
 * @brief Judge whether there is overlap of parallel edge
 *
 * @param target_rect
 * @param result_rect
 * @return true
 * @return false
 */
bool CutSpacingCheck::isParallelOverlap(DrcRect* target_rect, DrcRect* result_rect)
{
  return DRCUtil::isParallelOverlap(target_rect, result_rect);
}

/**
 * @brief Check for angular spacing violations
 *
 * @param target_rect
 * @param result_rect
 * @param require_spacing
 * @return true
 * @return false
 */
bool CutSpacingCheck::checkCornerSpacingViolation(DrcRect* target_rect, DrcRect* result_rect, int require_spacing)
{
  int distanceX = std::min(std::abs(target_rect->get_left() - result_rect->get_right()),
                           std::abs(target_rect->get_right() - result_rect->get_left()));
  int distanceY = std::min(std::abs(target_rect->get_bottom() - result_rect->get_top()),
                           std::abs(target_rect->get_top() - result_rect->get_bottom()));
  return require_spacing * require_spacing > distanceX * distanceX + distanceY * distanceY;
}

/**
 * @brief Check for parallel edge spacing violations
 *
 * @param target_rect
 * @param result_rect
 * @param require_spacing
 * @param query_result
 * @return true
 * @return false
 */
bool CutSpacingCheck::checkXYSpacingViolation(DrcRect* target_rect, DrcRect* result_rect, const int require_spacing,
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
    // If the spacing between the two rectangles does not meet the spacing requirements and it is
    // necessary to determine whether the span rectangle of the two rectangles is
    // penetrated by the third rectangle, if not, there is a violation.
    return !checkSpanBoxCoveredByExistedRect(span_box, isHorizontalParallelOverlap, query_result);
  }
  return false;
}

/**
 * @brief Check that the span rectangle is overwritten by an existing rectangle
 *
 * @param span_box
 * @param isHorizontalParallelOverlap
 * @param query_result
 * @return true
 * @return false
 */
bool CutSpacingCheck::checkSpanBoxCoveredByExistedRect(const RTreeBox& span_box, bool isHorizontalParallelOverlap,
                                                       std::vector<std::pair<RTreeBox, DrcRect*>>& query_result)
{
  for (auto& rect_pair : query_result) {
    // TEST
    // RTreeBox result_box = rect_pair.first;
    DrcRect* result_rect = rect_pair.second;
    if (DRCUtil::isBoxPenetratedByRect(span_box, result_rect, isHorizontalParallelOverlap)) {
      return true;
    }
  }
  return false;
}

/**
 * @brief Save the violation result
 *
 * @param cutLayerId
 * @param target_rect
 * @param result_rect
 * @param type
 */
void CutSpacingCheck::storeViolationResult(int cutLayerId, DrcRect* target_rect, DrcRect* result_rect, ViolationType type)
{
  if (type == ViolationType::kCutSpacing) {
    if (_interact_with_op) {
      _violation_rect_pair_list.push_back(std::make_pair(target_rect, result_rect));
    } else {
      addViolationBox(cutLayerId, target_rect, result_rect);
    }
  }
}

/**
 * @brief Add the violation to the result
 *
 * @param layerId
 * @param target_rect
 * @param result_rect
 */
void CutSpacingCheck::addViolationBox(int layerId, DrcRect* target_rect, DrcRect* result_rect)
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
 * @brief Checks whether the current violation rectangle
 *  intersects an existing violation rectangle
 *
 * @param routingLayerId
 * @param query_box
 * @param result
 */
void CutSpacingCheck::searchIntersectedViolationBox(int routingLayerId, const RTreeBox& query_box, std::vector<RTreeBox>& result)
{
  _layer_to_violation_box_tree[routingLayerId].query(bgi::intersects(query_box), std::back_inserter(result));
}

void CutSpacingCheck::reset()
{
  for (auto& [LayerId, spacing_spot_list] : _cut_layer_to_spacing_spots_list) {
    for (auto& spacing_spot : spacing_spot_list) {
      spacing_spot.clearSpotRects();
    }
  }
  _cut_layer_to_spacing_spots_list.clear();
  _layer_to_violation_box_tree.clear();
  _violation_rect_pair_list.clear();
}

int CutSpacingCheck::get_spacing_violation_num()
{
  int count = 0;
  for (auto& [layerId, spacing_spot_list] : _cut_layer_to_spacing_spots_list) {
    count += spacing_spot_list.size();
  }
  return count;
}

}  // namespace idrc